# Attribution and release scope

This is an independent community guide and integration patch. No affiliation or endorsement by the upstream projects is claimed.

This integration connects drowzeys' original abliteration edit with Mia AI Lab's EXL3 two-Spark stack. Our contribution is the offline characterization and approximate rank-1 runtime implementation, plus the deployment and Goose verification. The upstream model, edit, quantization, and serving foundation are credited below.

- **DeepSeek**: DeepSeek-V4.1-Flash architecture and native model data. The MIT notice is retained in `licenses/DeepSeek-MIT.txt` for provenance; no model tensors are included.
- **[Mia AI Lab](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks)**: EXL3 2.9bpw checkpoint and the two-Spark serving kit. The pinned kit uses AGPL-3.0. Its retained historical MIT notice is copied in `licenses/Mia-Legacy-MIT.txt`. Follow the upstream license when redistributing the generated overlay or modified launcher.
- **[drowzeys](https://huggingface.co/drowzeys/DeepSeek-V4.1-Flash-Abliterated-Cybersecurity-Unleashed)**: the original abliteration edit, supplied as the edited FP8 sidecar used in the offline characterization. Its model-card metadata says MIT and its gated repository presents an additional use agreement. This guide links that access process and does not redistribute the sidecar or recovered parameter data.
- **ExLlamaV3, vLLM, NVIDIA and other runtime dependencies**: supplied by the referenced serving image and upstream source. Their respective terms continue to apply; this repository's license does not relicense them.

The root `LICENSE` covers this guide and its code under GNU Affero General Public License version 3. Generated files contain or modify AGPL-covered upstream source. The pinned upstream repository plus this repository provide the source used by the recipe; an image distributor must also account for the rest of the image's components.

The exact runtime suffix was extracted from the deployed overlay, including an encoding blemish in a comment, to preserve byte-identical reconstruction. Its legacy comment uses “decoder” informally; the actual scope is backbone layer indices 10–35, as implemented and documented here. It is not the earlier standalone prototype or the earlier generator that mishandled an empty `ABLIT_PARAMS` variable.

No private endpoints, original hostnames, account credentials, session databases, or raw evaluation conversations are part of this release. Public source references and deliberately fictional configuration examples are included so readers can supply their own environment.
