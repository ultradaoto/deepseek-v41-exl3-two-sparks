# Reproduction checklist

The complete, canonical procedure now lives directly in the [main README](../README.md), including a handoff prompt for another assistant, commands, prerequisites, source pins, and verification gates. Follow its numbered sections in order.

1. Inspect both Sparks and identify the dedicated fabric interfaces, HCAs, IPv4 addresses, and RoCE v2 GIDs.
2. Clone the guide and pinned Mia kit. Fill a private local profile and merge it with the kit's defaults.
3. Download the pinned EXL3 checkpoint and native Engram source, including native config and index.
4. Obtain the original FP8 sidecar through its access process and derive the small parameter file while the model is stopped.
5. Ship the immutable stock image, sync both data replicas, and record completed stock generation.
6. Stop the model pair. Prepare the guarded launcher patch and hook image, then ship it to the worker.
7. Select fresh caches and start both ranks with explicit hook variables and overlay bind source.
8. Compare artifact hashes, require all 26 target layers on each rank, and run completed inference and your own behavioral comparisons.
9. Connect an OpenAI-compatible client using the exact advertised model ID.
10. Keep a stock rollback available and preserve manifests and evaluation results privately.

Supporting references:

- [Design and parameter recovery](DESIGN.md)
- [Operations, network recovery, and rollback](OPERATIONS.md)
- [Generic client access](CLIENTS.md)
- [Historical evidence and its limitations](../evidence/RESULTS.md)
- [Publication privacy review](PRIVACY_REVIEW.md)

The original deployment was verified on hardware. The published helpers were checked against its deployed source; a fresh installation on another pair of Sparks and a fresh full 26-layer parameter derivation have not been performed for this guide. Preserve that distinction when reporting reproduction results.
