# Connect Goose to the current DeepSeek model

Verified with Goose desktop/runtime **1.49.0**. Your desktop needs a route to the head's API, through a trusted LAN, VPN, or SSH tunnel. The example unauthenticated endpoint is for that private network; the guide does not configure a public API gateway.

## Provider settings

Create or edit an OpenAI-compatible custom provider in Goose:

| Field | Value |
| --- | --- |
| Display name | `DGX Spark` |
| Provider engine | `openai` |
| Base URL | `http://<desktop-reachable-head>:8888` |
| Model | **`DeepSeek-v4.1-Flash-EXL3`** |
| Context limit | **131072** |
| Streaming | Enabled |
| Reasoning model | Enabled |
| Requires authentication | Disabled for the tested private endpoint |

For this custom-provider path, Goose appends `/v1/chat/completions`. The saved base URL therefore has **no trailing `/v1`**. By contrast, `tools/check_api.py --base-url` takes the API root **including `/v1`**. Match the convention of the client you are configuring instead of adding the suffix twice.

Use `/v1/models` to discover the real model ID. A friendly name in an old client configuration does not create a server alias. If an old session still requests an obsolete ID, select the current model for that session or create a new chat. Changing a global default may not replace a model already stored with an existing session.

## Configuration files on Windows

Use `goose info` to confirm paths for your installation. The tested paths were:

```text
%APPDATA%\Block\goose\config\config.yaml
%APPDATA%\Block\goose\config\custom_providers\custom_dgx_spark.json
%APPDATA%\Goose\settings.json
```

If editing files directly, quit Goose first and back up the originals. Merge the following selected-provider fields into the existing YAML; do not replace the whole file or remove extensions and unrelated settings:

```yaml
active_provider: custom_dgx_spark
providers:
  custom_dgx_spark:
    model: DeepSeek-v4.1-Flash-EXL3
    enabled: true
    configured: true
```

In the existing custom-provider JSON, set `base_url` as above and update the intended `models` entry's `name` and `context_limit`. Preserve other fields. If the desktop's `recentModels` entry still holds the obsolete model name for this provider, update that entry too. Environment overrides, if configured separately, can override file defaults.

Our recovery also updated only the failed current session's saved model configuration after taking a SQLite backup. Readers can avoid database edits by starting a new chat with the correct model selected. Do not bulk-rewrite historical conversations.

## Prove the full request path

First test the server from the desktop using this repository:

```bash
python tools/check_api.py --base-url http://spark-head.example:8888/v1
```

Then send a short prompt in Goose: `Reply with only GOOSE LINK OK.` An actual answer proves more than a model-list or health response.

For an isolated check through the installed Windows runtime, run from an empty working directory in PowerShell:

```powershell
$gooseExe = Join-Path $env:LOCALAPPDATA 'Programs\Goose\resources\bin\goose.exe'
& $gooseExe run --no-profile --no-session --max-turns 1 --quiet `
  -t 'This is a connection check. Reply with only the exact text: GOOSE LINK OK'
```

This uses the saved provider and model without loading extensions or creating a persistent chat. Our check additionally set `GOOSE_MAX_TOKENS=512` only in that process's environment. It returned exactly `GOOSE LINK OK`, exit code 0, no stderr, in **3.297 seconds**. The desktop then worked for the user as well.

That verifies model connectivity and the Goose provider path. It does not verify every extension or completion of a coding task.

References: [Goose provider configuration](https://github.com/aaif-goose/goose/blob/main/documentation/docs/getting-started/providers.md), [Goose configuration files](https://github.com/aaif-goose/goose/blob/main/documentation/docs/guides/config-files.md).
