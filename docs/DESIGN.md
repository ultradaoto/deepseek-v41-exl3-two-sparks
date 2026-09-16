# What was done, and why it works

## The storage mismatch

The original sidecar replaces 26 `attn.wo_b` matrices, layers 10–35. Each matrix has 5,120 output rows and 8,192 input columns. It stores FP8 E4M3 values with UE8M0 scales for 32×32 blocks. Mia's corresponding matrices use EXL3 K=5 mul1 trellis encoding, with packed `trellis`, `suh`, `svh`, and `mul1` tensors.

This is a mismatch in **weight representation and loading**, not a fundamentally incompatible attention mechanism. An FP8 matrix cannot simply replace the packed tensor collection. The alternatives were a full re-quantization, re-encoding only the affected matrices, a mixed-precision loader, or a runtime transform of the ordinary output. We chose the runtime transform.

The upstream release now also offers a Mia-specific EXL3 sidecar and helper. That prepared weight-graft route is distinct from the original FP8 artifact and was not independently tested here.

## Offline characterization

The original process ran on a laptop CPU with NumPy:

1. Parse the edited safetensors header and fetch corresponding native tensor byte ranges.
2. Decode E4M3 using a 256-value lookup table. Expand each UE8M0 byte with `2^(byte−127)` over its 32×32 block.
3. Form `ΔW = W_edited − W_native` for each layer.
4. Run power iteration to estimate the dominant left singular direction, and inspect its energy relative to the whole edit.
5. Align vector signs, normalize each vector, average across layers, and normalize the result.
6. Compare fitted strengths and quantizer replay before fixing the deployed strength at 3.5.

The leading component accounted for about 85–90% of delta energy. Cross-layer direction cosines were at least about 0.9997. A naive full-matrix fit gave alpha near 3.35, while a fit restricted to changes of at least two FP8 ULPs recovered about 3.504 ± 0.004. Quantizer dead zones dilute the whole-matrix fit, motivating the deployed 3.5.

That does not establish exact recovery. Quantized matrices can share a systematic bias, so high cross-layer agreement alone does not remove the uncertainty. The portable derivation script implements the direction-recovery stage and explicitly sets the previously selected deployment strength. It records the naive fit; it does not pretend to rerun the separate dead-zone and byte-replay investigation.

The recovered approximation is:

```text
W_edited ≈ T W_native
T = I − α r rᵀ, with ||r|| = 1
```

## Move the operation to the output

For the deployed compressed matrix `Wq`, its ordinary output is `y = Wq x`. Applying `T` gives:

```text
T(Wq x) = y − α r (rᵀ y)
```

The hook therefore operates on activations, after the compressed linear operation. It avoids a second weight quantization pass. It is not a generic checkpoint adapter, and an arbitrary inference engine will not discover or execute it by finding the NPZ next to a model.

At alpha 3.5, the component along `r` has gain −2.5. Orthogonal components are unchanged in exact arithmetic. If `Wq = W + E`, the hook produces `TWx + TEx`: it transforms existing quantization error too. The dot product and correction are performed in FP32, then cast back to the output dtype.

## Tensor parallelism

In this pinned serving path, `wo_b` is row-parallel: the two ranks compute partial outputs that the parent sums. Both ranks have the same full output-space direction. Linearity gives:

```text
T(y0) + T(y1) = T(y0 + y1)
```

This requires no new collective. It assumes the matching output coordinates, insertion before the parent sum, and appropriate bias handling. The target projections in this deployment are bias-free. Do not reuse the hook blindly for a biased or differently sharded layer.

The actual deployed suffix wraps `Exl3LinearMethod.apply` once, matches `language_model.model.layers.N.attn.wo_b`, and caches the direction tensor per device before graph replay. It emits one fired-layer message per target layer per rank. The wrapper is installed during module import, before warmup/capture.

## What makes two Sparks possible

Mia's compression and file-backed Engram are the main memory enablers. The EXL3 shards occupy 196.14 GiB on disk; separate native Engram files add 189.13 GiB. Those tables are not all resident simultaneously. The head's successful boot reported 98.93 GiB for model loading, leaving room for its caches, workspaces, and host processes.

The rank-1 edit is carried by a 21,534-byte NPZ plus a small amount of runtime code. It avoids distributing another edited checkpoint, but has nonzero runtime cost. Our small sequential timing sample measured 26.14 versus 27.63 endpoint tokens/s (hook versus stock), about 5.4% lower. This includes prefill and HTTP handling and is not an isolated kernel-overhead measurement.

Neither disk-size arithmetic nor this deployment proves that every other quantization needs four Sparks. The defensible claim is narrower: this particular EXL3 + file-backed Engram + runtime-hook configuration ran on two.

References: [Mia serving source](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks/tree/85305680fb0f8ebc72b7b8ebf49ba54471019285), [upstream overlay and compatibility](https://huggingface.co/drowzeys/DeepSeek-V4.1-Flash-Abliterated-Cybersecurity-Unleashed), [Arditi et al., 2024](https://arxiv.org/abs/2406.11717).
