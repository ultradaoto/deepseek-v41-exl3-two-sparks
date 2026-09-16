# Verification record — 16 September 2026

This is a sanitized summary of the original run. Raw requests/responses, private addresses, hostnames, session databases, and internal logs are not included. The summary is not an independently audited benchmark release.

## Working deployment

- Two DGX Sparks, TP=2, unchanged Mia EXL3 2.9bpw checkpoint, separate file-backed native Engram.
- Hook at alpha 3.5, layers 10–35; all 26 layers fired on each rank after the final recovery.
- Both model containers running; no OOM; zero error markers in logs scoped to the successful startup.
- Configured context 131,072; full-length context was not evaluated here.
- Completed inference was checked in addition to model discovery and health. Client-specific software, local settings, and conversation state are outside this release's scope.

## Local stock versus hook experiment

Each configuration received 133 requests: 106 benign technical prompts, 15 authorized dual-use prompts, six coherence controls, and six repeated dual-use prompts. Primary prompts requested short outlines with identical settings: temperature 0, seed 42, thinking disabled, 384-token limit.

| Primary results | Stock | Hook alpha 3.5 |
| --- | ---: | ---: |
| Engaged, including qualified attempts | 111 | 118 |
| Full refusal | 10 | 0 |
| Inconclusive / token-limited | 0 | 3 |
| Total | 121 | 121 |

There were 118 valid paired comparisons: 109 engage→engage and **nine stock-refusal→hook-engagement** changes. Three hook truncations were excluded. Five security-probe differences recurred within both configurations' repeats and retained engagement after the hook was restored.

Stock changed binary label on one of six repeats; hook on zero of six. Neither configuration reproduced identical text on those six repeats. All six coherence controls passed in each complete pass. These results do not establish deterministic wording.

An **unblinded assistant** reviewed all security responses, flagged/inconclusive answers, changed pairs, controls, and selected benign agreements: 77/133 hook and 73/133 stock responses. Remaining benign engagement labels were automated. Generated code was not executed; technical correctness and exploit effectiveness were not measured. This was not the upstream Refusal32/Cyber22 benchmark, and upstream scores do not apply to our recovered-direction hook.

## Small timing sample

After one discarded warmup, each configuration had three serial requests producing 256 completion tokens. Median endpoint throughput was **26.14 tokens/s with the hook** and **27.63 stock**, approximately 5.4% lower for the hook. The endpoint timer includes prefill and HTTP handling; the runs were sequential, generated different text, and used freshly initialized caches. This is not isolated hook overhead or a general performance guarantee.

## Reproduction identifiers

| Item | Identifier |
| --- | --- |
| Mia serving-kit commit | `85305680fb0f8ebc72b7b8ebf49ba54471019285` |
| Mia model revision | `64ba41b6c916a587db06eae2e19b7845f7be6e6b` |
| Native model revision pinned for this guide | `dba1be0a40aa45a94ad051997016db3960a90277` |
| Overlay repository revision pinned for this guide | `87bb9850e6f49fd20ad9c8516b7817215cbbf5fc` |
| Base image registry digest | `sha256:2f0cf3adc0f989c1d446be274df864eb799630175f604c3b22b71b7205971dce` |
| Base image local ID | `sha256:4cdba4e946da2d19bf5b5a20c6d3a1a4bf421fa4d6db5082f271a986168176cb` |
| Deployed overlay SHA-256 | `7f109bec461eb67a0122e8dffcf14bb94cadeed29d1a4e4681d0eee8a78e5cd0` |
| Patched launcher SHA-256 | `a6d942561d28b823834e0c21e337da62298ecf8d5461a9617acd3e235e6d34d4` |
| Tested NPZ SHA-256 (data not distributed here) | `abf42531f925f418aa1914cdf50c51db474ec62b296b8de18ea1cfb8ac335a60` |

The kit commit and Mia model revision were read from the deployed source/download metadata. Native and overlay revisions were resolved while preparing this guide; they pin a reproducible new derivation, rather than retroactively proving the exact revisions of an earlier download that used `main`. NPZ identity, startup code identity, and a repeat of behavioral validation are separate checks.

The new preparation helper was checked against a clean pinned checkout and the existing tested NPZ: its overlay, launcher, and copied NPZ matched all three hashes. Bash syntax checking passed. See `packaging-validation.json` for additional local checks. The portable derivation code was tested numerically and against a bounded public native scale-tensor fetch; a fresh full 26-layer derivation and a second hardware deployment were not performed for this publication.
