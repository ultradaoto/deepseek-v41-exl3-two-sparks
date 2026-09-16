# Could the weights be uploaded to Hugging Face?

**Technically yes, but copying the current weights does not bake in the hook.** The served checkpoint is still Mia's original EXL3 pack. Without the runtime code and parameter file, another loader gets the unchanged checkpoint's behavior.

## Distribution choices

| Package | Approximate model-data size | What a reader still needs |
| --- | --- | --- |
| Guide + runtime code + locally derived parameters | Small code repository; parameters about 21 KiB | Referenced EXL3 pack, Engram, pinned runtime, upstream overlay access for derivation |
| Full EXL3 mirror + hook package | 196.14 GiB plus small hook artifacts | Native Engram separately, runtime integration, source notices/terms |
| Full deployment data bundle | 385.26 GiB plus code/images | Correct loader, two-node setup, and hook activation |
| Newly edited EXL3 checkpoint | Another conversion and evaluation project | Compatible re-encoding, accuracy checks, model card and distribution terms |

The first option is the efficient starting point. The base files are already hosted on Hugging Face. A small model-card repository could link their pinned revisions and explain the runtime integration. This repository currently publishes the recipe and code, while readers obtain the gated source artifact and derive the vector locally.

Publishing the exact recovered NPZ is a separate data-release decision: record its derivation, preserve source notices, and reconcile the overlay's additional agreement with the intended access and distribution. The absence of that NPZ here is a conservative packaging choice, not a claim that an MIT license categorically forbids redistribution.

## Licensing and access checked on 16 September 2026

The [native DeepSeek model](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/LICENSE) and [Mia quantized model](https://huggingface.co/Mia-AiLab/DeepSeek-V4.1-Flash-EXL3-2.9bpw/tree/main) contain MIT license notices. Retain their copyright and permission notices when redistributing covered material.

The [edited sidecar](https://huggingface.co/drowzeys/DeepSeek-V4.1-Flash-Abliterated-Cybersecurity-Unleashed) lists MIT in its model-card metadata **and** requires an additional gated agreement. Do not treat a metadata label as the entire set of terms. The [serving kit](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks/blob/85305680fb0f8ebc72b7b8ebf49ba54471019285/LICENSE) is AGPL-3.0; it is distinct from the checkpoint's license. Redistributing a modified serving image requires attention to its corresponding source and other bundled component licenses too.

## A practical upload plan, if a data release is chosen

1. Choose a Hub namespace, visibility/access policy, and storage plan. Build a dedicated upload directory containing only the intended release files and notices.
2. Write a model card that calls this an EXL3 runtime-hook deployment, identifies all source revisions and parameter hashes, and states the verification limits. It is not a normal PEFT/LoRA adapter or a generic `transformers.from_pretrained()` model.
3. Prefer references to upstream files. If a mirror is needed, preserve shard names, config/tokenizer files, and the corresponding index. Keep native Engram in a separate documented tree.
4. Use the current Hugging Face upload tooling from a separate machine or while serving is stopped. Account for disk reads, memory, CPU, and bandwidth; do not start a 200–400 GiB upload alongside this tightly budgeted live model.
5. After upload, verify remote manifests and hashes, then test a clean download and inference. An upload completion message alone is not deployment validation.

Hugging Face's current documentation recommends `hf upload` / `HfApi.upload_folder()` with Xet support for resumable large-folder uploads. The older `upload_large_folder` path is now deprecated. Its `copy_files()` API can also copy between Hub repositories without downloading and re-uploading large files, subject to access and source/destination support. That is worth checking before transferring a duplicate through the Sparks. [Official upload and copy documentation](https://huggingface.co/docs/huggingface_hub/guides/upload)

Example, **after** the destination and release directory have been deliberately prepared:

```bash
hf auth login
hf upload YOUR_NAMESPACE/YOUR_REPOSITORY ./hf-upload . --repo-type model
```

A full mirror counts against account storage. Free public capacity is best-effort; private free capacity is 100 GB, below either full package here. Current guidance recommends files below 200 GB and has a 500 GB single-file hard limit; the existing EXL3 and Engram shard sizes fit those per-file limits. Check current account entitlements before starting; no storage purchase or weight upload was performed for this guide. [Hub storage limits](https://huggingface.co/docs/hub/storage-limits)
