"""Recover the shared output direction from an authorized local FP8 sidecar.

Portable version of the offline NumPy procedure used for this deployment.
Only selected native tensors are fetched with checked HTTP byte ranges. The
gated edited sidecar must already have been obtained through upstream access.
No GPU, EXL3 rewriting, or automatic download of the edited sidecar is involved.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import urllib.request
import numpy as np

NATIVE_REPO = 'deepseek-ai/DeepSeek-V4.1-Flash'
NATIVE_REVISION = 'dba1be0a40aa45a94ad051997016db3960a90277'


def fp8_lut():
    result = np.zeros(256, np.float32)
    for b in range(256):
        sign, exponent, mantissa = b >> 7, (b >> 3) & 15, b & 7
        value = (mantissa / 8) * 2**-6 if exponent == 0 else (1 + mantissa / 8) * 2**(exponent - 7)
        if exponent == 15 and mantissa == 7:
            value = np.nan
        result[b] = (-1 if sign else 1) * value
    return result


def dequantize(weight_bytes, scale_bytes, shape, scale_shape):
    shape, scale_shape = tuple(shape), tuple(scale_shape)
    if len(shape) != 2 or len(scale_shape) != 2 or any(a != b * 32 for a, b in zip(shape, scale_shape)):
        raise ValueError('Expected 32x32 UE8M0 blocks')
    weights = fp8_lut()[np.frombuffer(weight_bytes, np.uint8)].reshape(shape)
    scale_raw = np.frombuffer(scale_bytes, np.uint8)
    if np.any(scale_raw == 255):
        raise ValueError('Invalid UE8M0 scale')
    scales = np.exp2(scale_raw.astype(np.float32) - 127).reshape(scale_shape)
    result = weights * np.repeat(np.repeat(scales, 32, axis=0), 32, axis=1)
    if not np.isfinite(result).all():
        raise ValueError('Nonfinite FP8 tensor')
    return result


def byte_range(url, start, end):
    expected = end - start + 1
    if start < 0 or not 1 <= expected <= 64 * 1024 * 1024:
        raise ValueError('Unexpected tensor range size')
    req = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}', 'User-Agent': 'dsv41-rank1-reproduction'})
    with urllib.request.urlopen(req, timeout=180) as response:
        content_range = response.headers.get('Content-Range', '')
        if response.status != 206 or not content_range.startswith(f'bytes {start}-{end}/'):
            raise ValueError('Server did not honor the exact byte range; refusing a whole-shard download')
        data = response.read(expected + 1)
    if len(data) != expected:
        raise ValueError('Truncated or oversized byte-range response')
    return data


class LocalTensors:
    def __init__(self, path):
        self.path = path
        with path.open('rb') as f:
            length = struct.unpack('<Q', f.read(8))[0]
            if not 2 <= length <= 32 * 1024 * 1024:
                raise ValueError('Unexpected safetensors header size')
            self.header = json.loads(f.read(length))
        self.base = 8 + length

    def read(self, key):
        entry = self.header[key]
        start, end = entry['data_offsets']
        if not 0 <= start < end <= self.path.stat().st_size - self.base:
            raise ValueError('Invalid tensor offsets')
        with self.path.open('rb') as f:
            f.seek(self.base + start)
            data = f.read(end - start)
        if len(data) != end - start:
            raise ValueError('Truncated tensor')
        return data, entry['shape']


class NativeTensors:
    def __init__(self, repo, revision):
        self.base = f'https://huggingface.co/{repo}/resolve/{revision}'
        with urllib.request.urlopen(self.base + '/model.safetensors.index.json', timeout=60) as response:
            self.index = json.load(response)['weight_map']
        self.headers = {}

    def read(self, key):
        shard = self.index[key]
        url = self.base + '/' + shard
        if shard not in self.headers:
            length = struct.unpack('<Q', byte_range(url, 0, 7))[0]
            if not 2 <= length <= 32 * 1024 * 1024:
                raise ValueError('Unexpected safetensors header size')
            self.headers[shard] = (json.loads(byte_range(url, 8, 7 + length)), 8 + length)
        header, base = self.headers[shard]
        entry = header[key]
        start, end = entry['data_offsets']
        return byte_range(url, base + start, base + end - 1), entry['shape']


def top_direction(matrix, iterations=100):
    rng = np.random.default_rng(0)
    v = rng.standard_normal(matrix.shape[1]).astype(np.float32)
    v /= np.linalg.norm(v)
    for _ in range(iterations):
        u = matrix @ v
        unorm = np.linalg.norm(u)
        if not np.isfinite(unorm) or unorm == 0:
            raise ValueError('Zero or invalid edit')
        u /= unorm
        v = matrix.T @ u
        singular_value = np.linalg.norm(v)
        v /= singular_value
    return u, float(singular_value)


def read_matrix(source, layer):
    wb, shape = source.read(f'layers.{layer}.attn.wo_b.weight')
    sb, scale_shape = source.read(f'layers.{layer}.attn.wo_b.scale')
    if tuple(shape) != (5120, 8192):
        raise ValueError('Wrong matrix shape or sidecar; use the original FP8 wo_b overlay')
    return dequantize(wb, sb, shape, scale_shape)


def derive(overlay, output, native_repo=NATIVE_REPO, native_revision=NATIVE_REVISION):
    report_path = output.with_suffix('.derivation.json')
    if output.exists() or report_path.exists():
        raise ValueError('Output exists; preserve the previous derivation')
    edited, native = LocalTensors(overlay), NativeTensors(native_repo, native_revision)
    directions, measurements = [], []
    for layer in range(10, 36):
        w0, we = read_matrix(native, layer), read_matrix(edited, layer)
        delta = we - w0
        direction, singular_value = top_direction(delta)
        projected_w0, projected_delta = direction @ w0, direction @ delta
        fit_alpha = float(-(projected_delta @ projected_w0) / (projected_w0 @ projected_w0))
        energy = float(singular_value**2 / np.linalg.norm(delta)**2)
        measurements.append({'layer': layer, 'full_matrix_fit_alpha': fit_alpha, 'top_component_energy': energy})
        directions.append(direction)
        print(json.dumps(measurements[-1]), flush=True)
        del w0, we, delta
    aligned = np.stack([u * (1 if u @ directions[0] >= 0 else -1) for u in directions])
    aligned /= np.linalg.norm(aligned, axis=1, keepdims=True)
    r = aligned.mean(axis=0)
    r /= np.linalg.norm(r)
    output.parent.mkdir(parents=True, exist_ok=True)
    # The original workflow first estimated ~3.35 from quantized matrices, then
    # set the deployment strength to 3.5 after dead-zone-aware quantizer analysis.
    # Do not mistake the biased full-matrix fit for the intended strength.
    with output.open('xb') as f:
        np.savez(f, r=r.astype(np.float32), alpha=np.full(26, 3.5, np.float32), layers=np.arange(10, 36))
    digest = hashlib.sha256()
    with overlay.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    report = {'native_repo': native_repo, 'native_revision': native_revision,
              'overlay_sha256': digest.hexdigest(), 'params_sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
              'method': '100 power iterations; seed 0; sign-aligned mean of per-layer directions; float32',
              'deployment_alpha': 3.5, 'min_cross_layer_cosine': float(np.abs(aligned @ r).min()),
              'layers': measurements, 'limitation': 'Recovered approximation, not exact upstream direction; evaluate this artifact separately.'}
    report_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--overlay', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--native-repo', default=NATIVE_REPO)
    parser.add_argument('--native-revision', default=NATIVE_REVISION)
    args = parser.parse_args()
    print(json.dumps(derive(args.overlay, args.out, args.native_repo, args.native_revision), indent=2))
