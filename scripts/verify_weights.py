#!/usr/bin/env python3
"""Verify a weights file by building the model, loading, and running a dummy forward."""
import os, sys, types, math
import tensorflow as tf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Minimal Sionna stub
try:
    import sionna as sn  # type: ignore
    assert hasattr(sn, "utils") and hasattr(sn.utils, "log10") and hasattr(sn, "PI")
except Exception:
    sn = types.ModuleType("sionna")
    utils = types.SimpleNamespace()
    def _log10(x: tf.Tensor) -> tf.Tensor:
        x = tf.convert_to_tensor(x)
        return tf.math.log(x) / tf.math.log(tf.constant(10.0, x.dtype))
    utils.log10 = _log10
    sn.utils = utils
    sn.PI = math.pi
    sys.modules["sionna"] = sn

from scripts.generate_weights import _build_dummy_gen, _input_len
from scripts.load_weights import load_weights_flex
from synch import DeepNSynch

def main(path: str) -> int:
    tf.random.set_seed(7)
    dummy = _build_dummy_gen()
    T = _input_len(dummy.config)

    model = DeepNSynch(dummy)
    n = load_weights_flex(model, path)

    y = tf.complex(tf.zeros([2, T], tf.float32), tf.zeros([2, T], tf.float32))
    out = model(y)  # just to ensure graph runs
    print("[OK] Forward ran. Output types/shapes:")
    if isinstance(out, (tuple, list)):
        for i, o in enumerate(out):
            print(f"  [{i}] {o.dtype} {o.shape}")
    else:
        print(f"  {out}")
    return 0

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python scripts/verify_weights.py <weights.(npz|h5|dat)>")
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1]))
