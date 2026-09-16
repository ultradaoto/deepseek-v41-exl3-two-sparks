"""Prepare the byte-identical tested EXL3 hook and guarded launcher patch.

Does not start containers, access a GPU, download weights, or edit checkpoints.
The build directory is new; start.sh is backed up and patched only after all
source and parameter validation passes. Run against the pinned Mia checkout.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
KIT_COMMIT = '85305680fb0f8ebc72b7b8ebf49ba54471019285'
STOCK_OVERLAY_SHA = 'ccdc69bfa04bff4870c3e555736a990fde6448ddb329441c4e0a27d6fc41078d'
HOOK_OVERLAY_SHA = '7f109bec461eb67a0122e8dffcf14bb94cadeed29d1a4e4681d0eee8a78e5cd0'
STOCK_START_SHA = 'e9e390237e98fb42d217e426ee3118ff2592592d7da0c04a9a0ea266b04af13b'
HOOK_START_SHA = 'a6d942561d28b823834e0c21e337da62298ecf8d5461a9617acd3e235e6d34d4'
TESTED_PARAMS_SHA = 'abf42531f925f418aa1914cdf50c51db474ec62b296b8de18ea1cfb8ac335a60'
BASE_IMAGE = 'ghcr.io/miaai-lab/deepseek-v4.1-flash-exl3-2x-dgx-sparks@sha256:2f0cf3adc0f989c1d446be274df864eb799630175f604c3b22b71b7205971dce'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def validate_params(path):
    if path.stat().st_size > 1024 * 1024:
        raise ValueError('Unexpectedly large parameter file; expected about 21 KiB')
    with np.load(path, allow_pickle=False) as data:
        r, alpha, layers = data['r'], data['alpha'], data['layers']
        if r.shape != (5120,) or r.dtype != np.float32 or not np.isfinite(r).all():
            raise ValueError('r must be finite float32[5120]')
        if not np.isclose(np.linalg.norm(r.astype(np.float64)), 1, atol=1e-5, rtol=0):
            raise ValueError('Direction must have unit norm')
        if not np.array_equal(layers, np.arange(10, 36)):
            raise ValueError('Expected ordered layers 10 through 35')
        if alpha.shape != (26,) or not np.isfinite(alpha).all() or not np.all(alpha == 3.5):
            raise ValueError('This reproduction profile requires 26 alpha values of 3.5')
    return sha(path.read_bytes())


def patch_launcher(original):
    if sha(original) == HOOK_START_SHA:
        return original
    if sha(original) != STOCK_START_SHA:
        raise ValueError('Unexpected start.sh: use the pinned clean checkout; do not patch a different version')
    old1 = b'             LONG_PREFILL_TOKEN_THRESHOLD; do'
    new1 = b'             LONG_PREFILL_TOKEN_THRESHOLD \\\n             ABLIT ABLIT_ALPHA ABLIT_PARAMS; do'
    old2 = b'        --entrypoint bash "$IMAGE" /start.sh >/dev/null'
    new2 = (b'        -e ABLIT="${ABLIT:-}" -e ABLIT_ALPHA="${ABLIT_ALPHA:-}" '
            b'-e ABLIT_PARAMS="${ABLIT_PARAMS:-}" \\\n' + old2)
    if original.count(old1) != 1 or original.count(old2) != 1:
        raise ValueError('Launcher anchors are not unique')
    result = original.replace(old1, new1).replace(old2, new2)
    if sha(result) != HOOK_START_SHA:
        raise ValueError('Patched launcher does not match the deployed source')
    return result


def prepare(kit, params, out):
    kit, params, out = kit.resolve(), params.resolve(), out.resolve()
    if out.exists():
        raise ValueError('Build destination exists; use a new directory and preserve the previous build')
    original = (kit / 'overlay/exl3.py').read_bytes()
    if sha(original) != STOCK_OVERLAY_SHA:
        raise ValueError('Unexpected EXL3 overlay; use the pinned Mia checkout')
    overlay = original + (ROOT / 'runtime/wo_b_hook_append.py').read_bytes()
    if sha(overlay) != HOOK_OVERLAY_SHA:
        raise ValueError('Generated overlay does not match the deployed source')
    start_path = kit / 'start.sh'
    start_original = start_path.read_bytes()
    patched = patch_launcher(start_original)
    params_sha = validate_params(params)
    backup = kit / 'start.sh.before-runtime-hook'
    if backup.exists() and sha(backup.read_bytes()) != STOCK_START_SHA:
        raise ValueError('Existing launcher backup differs; preserve and inspect it')
    identity = sha(overlay + params.read_bytes() + b'ABLIT=1;alpha=npz;layers=10:35')[:16]
    out.mkdir(parents=True)
    (out / 'exl3.py').write_bytes(overlay)
    shutil.copyfile(params, out / 'wob_hook_params.npz')
    (out / 'Dockerfile').write_text(
        f'FROM {BASE_IMAGE}\nCOPY exl3.py /opt/dsv41/exl3.py\n'
        'COPY wob_hook_params.npz /opt/dsv41/wob_hook_params.npz\n', encoding='utf-8')
    manifest = {'kit_commit': KIT_COMMIT, 'base_image': BASE_IMAGE, 'config_id': identity,
                'overlay_sha256': sha(overlay), 'launcher_sha256': sha(patched),
                'params_sha256': params_sha, 'matches_tested_params': params_sha == TESTED_PARAMS_SHA,
                'alpha': 3.5, 'layers': list(range(10, 36)),
                'note': 'Freshly derived parameters require their own behavioral validation.'}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    if start_original != patched:
        if not backup.exists():
            shutil.copy2(start_path, backup)
        staged = kit / 'start.sh.runtime-hook-new'
        with staged.open('xb') as f:
            f.write(patched)
        shutil.copymode(start_path, staged)
        staged.replace(start_path)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kit', type=Path, required=True)
    parser.add_argument('--params', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.kit, args.params, args.out), indent=2))
