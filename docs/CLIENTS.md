# Connect an OpenAI-compatible client

The deployment exposes a chat-completions API. It does not require a particular desktop application. Follow [the full reproduction guide](../README.md) before connecting a client.

| Setting | Value |
| --- | --- |
| API root | `http://<your-reachable-head>:8888/v1` |
| Model ID | `DeepSeek-v4.1-Flash-EXL3` |
| Context limit | At most 131072 for the documented profile |
| Initial test | Short, non-streaming completion with thinking disabled |

Some clients expect only a host URL and append `/v1/chat/completions`; others expect an API root ending in `/v1`. Follow that client's convention. Verify the actual request path if you see a route-level 404. A model-level 404 means the requested model identifier does not match `/v1/models`; a friendly display name does not create a model alias.

Update the model selected in an existing session, or start a new session, if an older model identifier persists after changing defaults. There is no need to edit historical conversation databases.

## Reach the API privately

The fabric address used between the Sparks is not necessarily reachable from your client. Use your own trusted routed network or a tunnel. This Bash example runs on the **client machine**, prompts for your existing head SSH destination, and binds a local port only to loopback:

```bash
read -r -p 'Head SSH destination (your user@host or SSH alias): ' HEAD_SSH
ssh -N -o ExitOnForwardFailure=yes \
  -L 127.0.0.1:18888:127.0.0.1:8888 "$HEAD_SSH"
```

Leave the tunnel running. In another client terminal, from this guide's checkout:

```bash
python tools/check_api.py --base-url http://127.0.0.1:18888/v1
```

Expected: reply `323` and `passed: true`. Configure the client's API root as `http://127.0.0.1:18888/v1`, adjusting the `/v1` convention as required. Stop the tunnel with Ctrl+C when finished.

The supplied helper assumes an unauthenticated private server. It does not configure a public gateway or add authentication headers. If you choose to secure your endpoint with authentication, configure the client accordingly and keep real credentials in its local secret store. Never commit them or paste them into support conversations.

Model discovery, a working stream, tool calling, and completion of a real application task are separate checks. Start with a complete plain-text response, then test the specific capabilities your client uses.
