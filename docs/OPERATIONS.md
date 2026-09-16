# Operations, recovery, and rollback

## The symptoms we actually encountered

| Symptom | Cause in this deployment | Resolution |
| --- | --- | --- |
| Client 404, model does not exist | Client sent an obsolete model ID | Select `DeepSeek-v4.1-Flash-EXL3` in the provider and current session |
| HTTP 200 stream with no tokens | Head remained alive after worker reboot | Check both ranks; recover the worker dependencies and restart the pair |
| No route to the worker fabric IP | Reboot removed a temporary IPv4 assignment | Configure the dedicated fabric profile persistently |
| Temporary IP worked briefly, then disappeared | NetworkManager still owned the interface with a DHCP profile | Correct that profile instead of repeatedly running `ip addr add` |
| NCCL initialization failed after IP restoration | Worker GID 3 was IPv6; head GID 3 was IPv4 | Restore the original IPv4-only fabric or select the correct per-node IPv4 RoCE v2 GIDs |
| Worker cannot restart after reboot | Launcher bind sources under `/tmp` were gone | Re-stage all startup files with the pinned launcher |
| `/engram-src/config.json` not found | Native Engram staging omitted the native config | Download the native config alongside shards 47/48 and the index, then rebuild/sync the slim source |
| Hook image loads but hook has no effect | Host bind mount shadows baked `exl3.py` | Set `EXL3_OVERLAY_HOST` to the prepared hook overlay |
| Parameter load tries an empty filename | Launcher passes an empty `ABLIT_PARAMS` | Use explicit path; deployed code also handles empty values with `get(...) or default` |
| Hook announces installation but may be bypassed | Installation is not evidence that the serving path executes it | Require 26 unique fired-layer logs on each rank and real inference |

## Inspect both ranks first

On each node, with the appropriate container name:

```bash
docker ps -a --filter name=dsv41-exl3
docker inspect -f '{{json .State}}' dsv41-exl3-head
docker logs --tail 100 dsv41-exl3-head
free -h
nvidia-smi
```

Use `dsv41-exl3-worker` on the worker. A listening head, `/health` 200, or an HTTP 200 streaming header is insufficient. The failed head accepted requests while waiting indefinitely for its missing tensor-parallel partner.

When diagnosing a new launch, filter logs by the current `StartedAt`; old hook lines and errors remain in reused containers. Verify the image and source hashes on both nodes. Do not confuse `docker logs` output from a prior successful start with proof of the current process.

## Fabric address and GID after a reboot

These are example commands for an existing **dedicated fabric** NetworkManager profile. Inspect its interface binding before making changes. Keep management/LAN/VPN profiles unchanged.

```bash
nmcli -f NAME,UUID,DEVICE connection show
PROFILE='<UUID of the dedicated CX7 fabric connection>'
nmcli -f connection.interface-name,ipv4.method,ipv4.addresses,ipv6.method connection show "$PROFILE"
```

The actual recovery saved the old values, changed that worker profile to manual IPv4 with no default route, and disabled IPv6 on that one link to restore its original IPv4 RoCE GID index. On a stopped model pair, substituting that node's address:

```bash
read -r -p 'This node dedicated fabric IPv4 address: ' FABRIC_IP
sudo nmcli connection modify "$PROFILE" \
  ipv4.method manual ipv4.addresses "$FABRIC_IP/24" ipv4.never-default yes \
  ipv6.method disabled
sudo nmcli connection down "$PROFILE"
sudo nmcli connection up "$PROFILE"
```

Recheck the IP, peer ping/SSH, and GID table before serving. You can also retain IPv6 and select each node's correct IPv4 GID in `HEAD_GID`/`WORKER_GID`; that is a different valid configuration, not the exact recovery used here. Never assume the same numeric index represents the same address family on both nodes.

## Missing `/tmp` startup files

The pinned kit copies the worker startup script, chat template, EXL3 overlay, and auxiliary patches into `/tmp`, then bind-mounts them into its container. A reboot can remove those host files. Running `docker start` without restoring them is not a complete recovery.

Use the pinned, configured launcher to stage all required files and launch the pair again after the fabric is healthy. Preserve your hook image, `EXL3_OVERLAY_HOST`, parameters, data mounts, and configuration-specific caches. Do not copy only the hook file and assume the other bind sources survived. Compare staged source hashes before treating the recovery as complete.

The working deployment retains Docker restart policy `no`; no watchdog or auto-reboot behavior was added. Therefore this guide does **not** promise unattended recovery across node reboots. Making that durable requires persistent startup-file locations plus ordered network/storage/service startup and a separately tested restart procedure. Enabling a restart policy alone does not solve the missing-file problem.

## Cache and graph boundaries

Install the hook before warmup and CUDA graph capture. Keep the direction buffer stable for that process. Never hot-toggle alpha, layer selection, direction, or `ABLIT`: generated KV/prefix state belongs to the original configuration, and captured execution may still contain the old operation.

The preparation manifest supplies a `config_id` derived from the hook code and parameter bytes. Use fresh cache directories when changing configurations. Keep Engram source paths outside those cache directories so a cache change cannot silently change the model's data mounts. Do not delete caches while a serving process is using them.

## Revert to the stock checkpoint

Because the checkpoint was never edited, rollback is a serving change. Finish requests and stop the pair using `./start.sh stop`. Set `CACHE_ROOT` and `WORKER_VLLM_CACHE` in `.env` to new stock-specific directories. Keep data mounts fixed and ensure `.env` contains none of the hook-only overrides. Then:

```bash
cd "$HOME/dsv41-kit"
env -u EXL3_OVERLAY_HOST -u ABLIT -u ABLIT_ALPHA -u ABLIT_PARAMS \
  IMAGE=dsv41-flash-exl3:stock-pinned \
  SKIP_PULL=1 SKIP_BUILD=1 SKIP_SHIP=1 SKIP_SYNC=1 TAIL=0 ./start.sh
```

The launcher defaults to the unchanged `overlay/exl3.py`, and the stock image has no added NPZ. Verify generation and absence of hook installation in logs from the new process. The environment-passthrough patch is inert for stock and may remain in the launcher. A full source rollback can restore `start.sh.before-runtime-hook` after checking that backup and preserving any later launcher edits.

## Quality and memory limits

Keep a frozen baseline prompt set and identical inference settings for comparisons. Score complete answers contextually; a qualification is not automatically a refusal, and a token-limit response is not a success. The published counts are a local experiment, not a universal result.

The 131,072-token configuration value was not a full-context validation. Test long prefill and concurrent use separately before increasing limits. We did not run heavy uploads, training, or new packaging builds on the serving GPUs while writing this guide.
