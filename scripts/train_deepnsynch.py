#!/usr/bin/env python3
"""
Train (or warm-build) DeepNSynch and save weights in three formats:
- weights.dat  (pickle list)
- weights.npz  (name->array)
- weights.h5   (Keras)

This script tolerates missing Sionna by stubbing minimal pieces so that
model can be constructed for shape-compatibility when needed.
"""
import os, sys, types, math, pickle
import numpy as np
import tensorflow as tf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Sionna stub if needed
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

from scripts.generate_weights import _build_dummy_gen, _input_len  # reuse
from synch import DeepNSynch

def _save_all_formats(model: tf.keras.Model, root: str) -> None:
    weights_list = model.get_weights()
    # dat
    with open(os.path.join(root, "weights.dat"), "wb") as f:
        pickle.dump(weights_list, f)
    # npz
    name_to_arr = {v.name: w for w, v in zip(weights_list, model.weights)}
    np.savez_compressed(os.path.join(root, "weights.npz"), **name_to_arr)
    # h5
    model.save_weights(os.path.join(root, "weights.h5"))
    print("[OK] Saved weights as .dat, .npz, and .h5")

def main() -> int:
    tf.random.set_seed(123)
    dummy = _build_dummy_gen()
    T = _input_len(dummy.config)

    # Build model
    model = DeepNSynch(dummy)

    # You can plug in your real training pipeline here.
    # For now, we just do a few warm-up steps to ensure variables exist.
    y = tf.complex(tf.zeros([8, T], tf.float32), tf.zeros([8, T], tf.float32))
    _ = model(y)

    # TODO: replace with real training loop and optimizer if needed
    _save_all_formats(model, ROOT)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
