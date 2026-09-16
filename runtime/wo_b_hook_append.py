

# ============================================================================
# ablit rank-1 wo_b hook (operator, 2026-09-16) — runtime, gated on ABLIT=1.
#   y' = y - alpha * r * (r . y)   on the attention output projection wo_b,
#   decoder layers 10-35. r: fp32[5120] shared direction; alpha per-layer (~3.5).
# NOT a projection: at alpha=3.5 the r-component gain is (1-alpha)=-2.5 (flip+amplify).
# Params baked at /opt/dsv41/wob_hook_params.npz. No effect unless ABLIT=1.
# TP-safe: wo_b is row-parallel; apply() returns the per-rank PARTIAL and the parent
# all-reduces. The op is linear and r is replicated, so sum_rank[y_r - a r(r.y_r)]
# = y_full - a r(r.y_full). Reduction/subtraction in fp32. Idempotent install.
# ============================================================================
import os as _ablit_os
import re as _ablit_re

_ABLIT_TARGET_LAYERS = frozenset(range(10, 36))
_ABLIT_PREFIX_RE = _ablit_re.compile(r"layers\.(\d+)\.[A-Za-z_]*attn\.wo_b\b")


def _ablit_install():
    if _ablit_os.environ.get("ABLIT", "0") != "1":
        return
    if getattr(Exl3LinearMethod, "_ablit_installed", False):
        return
    import numpy as _np
    import torch as _th
    _path = _ablit_os.environ.get("ABLIT_PARAMS") or "/opt/dsv41/wob_hook_params.npz"
    _d = _np.load(_path)
    _r_np = _d["r"].astype("float32")
    _alpha = {int(_l): float(_a) for _l, _a in zip(_d["layers"], _d["alpha"])}
    _ov = _ablit_os.environ.get("ABLIT_ALPHA")
    _alpha_ov = float(_ov) if _ov not in (None, "") else None
    _cache = {}
    _logged = set()

    def _r_for(dev):
        t = _cache.get(dev)
        if t is None:
            t = _th.from_numpy(_r_np).to(device=dev, dtype=_th.float32)
            _cache[dev] = t
        return t

    def _layer_of(prefix):
        if not prefix:
            return None
        m = _ABLIT_PREFIX_RE.search(prefix)
        return int(m.group(1)) if m else None

    _orig_apply = Exl3LinearMethod.apply

    def apply(self, layer, x, *args, **kwargs):
        y = _orig_apply(self, layer, x, *args, **kwargs)
        prefix = getattr(layer, "_exl3_prefix", None) or getattr(self, "prefix", None)
        L = _layer_of(prefix)
        if L in _ABLIT_TARGET_LAYERS:
            al = _alpha_ov if _alpha_ov is not None else _alpha.get(L)
            if al is not None:
                r = _r_for(y.device)
                yf = y.float()
                s = yf @ r
                y = (yf - (al * s).unsqueeze(-1) * r).to(y.dtype)
                if L not in _logged:
                    _logged.add(L)
                    print("[ablit] fired L%d prefix=%s alpha=%.3f n_layers=%d"
                          % (L, prefix, al, len(_logged)), flush=True)
        return y

    Exl3LinearMethod.apply = apply
    Exl3LinearMethod._ablit_installed = True
    _msg = ("override %.3f" % _alpha_ov) if _alpha_ov is not None else "per-layer(~3.5)"
    print("[ablit] rank-1 wo_b hook INSTALLED: layers 10-35 alpha=%s params=%s"
          % (_msg, _path), flush=True)


_ablit_install()
