#!/usr/bin/env python3
"""
Generate weights for DeepNSynch in three formats:
- weights.dat : pickle(list of numpy arrays)  [legacy-compatible]
- weights.npz : name->array mapping            [safer / future-proof]
- weights.h5  : Keras save_weights format      [standard]

The script stubs a minimal `sionna` if missing so that model build succeeds
without a full PHY install. It also builds a dummy NPRACH generator with
correct shapes (format-0: DFT=48, 4 SG, 5 seq/SG).
"""
from __future__ import annotations
import os, sys, types, pickle, math, json
import numpy as np
import tensorflow as tf

# Project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Try import sionna, else stub
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

# Dummy NPRACH generator (minimal API used by DeepNSynch)
class _Cfg:
    @property
    def nprach_dft_size(self) -> int: return 48
    @property
    def nprach_seq_per_sg(self) -> int: return 5
    @property
    def nprach_sg_per_rep(self) -> int: return 4
    @property
    def nprach_num_rep(self) -> int: return 1
    @property
    def nprach_num_sc(self) -> int: return 48

class _DummyGen:
    def __init__(self, cfg: _Cfg, seq_indices: tf.Tensor, freq_patterns: tf.Tensor):
        self._config = cfg
        self._seq_indices = seq_indices
        self._freq_patterns = freq_patterns
    @property
    def config(self) -> _Cfg: return self._config
    @property
    def seq_indices(self) -> tf.Tensor: return self._seq_indices
    @property
    def freq_patterns(self) -> tf.Tensor: return self._freq_patterns

def _build_dummy_gen() -> _DummyGen:
    cfg = _Cfg()
    samples_per_seq = 48
    samples_per_cp  = 12
    samples_per_sg  = samples_per_cp + cfg.nprach_seq_per_sg * samples_per_seq  # 12 + 5*48 = 252
    num_sg = cfg.nprach_sg_per_rep * cfg.nprach_num_rep  # 4

    base = tf.range(samples_per_seq, dtype=tf.int32)[tf.newaxis, :]  # [1,48]
    shifts = tf.range(num_sg, dtype=tf.int32) * samples_per_sg + samples_per_cp  # [4]
    shifts = tf.repeat(shifts, cfg.nprach_seq_per_sg)  # 20
    seq_indices = base + shifts[:, tf.newaxis]  # [20,48]

    num_sc = cfg.nprach_num_sc
    dft = cfg.nprach_dft_size
    k = tf.range(num_sc, dtype=tf.int32)[:, tf.newaxis]        # [48,1]
    m = tf.range(num_sg, dtype=tf.int32)[tf.newaxis, :]        # [1,4]
    freq_patterns = (k + m) % dft                              # [48,4]

    return _DummyGen(cfg, seq_indices, freq_patterns)

def _input_len(cfg: _Cfg) -> int:
    samples_per_seq = 48
    samples_per_cp  = 12
    samples_per_sg  = samples_per_cp + cfg.nprach_seq_per_sg * samples_per_seq
    num_sg = cfg.nprach_sg_per_rep * cfg.nprach_num_rep
    return num_sg * samples_per_sg  # 1008

def main() -> int:
    tf.random.set_seed(42)

    dummy = _build_dummy_gen()

    # Import after stub is set
    from synch import DeepNSynch

    model = DeepNSynch(dummy)

    # One forward pass to build variables
    T = _input_len(dummy.config)
    y = tf.complex(tf.zeros([1, T], tf.float32), tf.zeros([1, T], tf.float32))
    _ = model(y)

    # Collect weights
    weights_list = model.get_weights()
    assert len(weights_list) > 0, "No weights collected from model."

    # (A) Legacy pickle(list) -> weights.dat
    dat_path = os.path.join(ROOT, "weights.dat")
    with open(dat_path, "wb") as f:
        pickle.dump(weights_list, f)

    # (B) Name-based -> weights.npz  (safer for future)
    # Use tf.Variable names for stable mapping (e.g., '.../kernel:0')
    name_to_arr = {}
    for w, v in zip(weights_list, model.weights):
        name_to_arr[v.name] = w
    npz_path = os.path.join(ROOT, "weights.npz")
    np.savez_compressed(npz_path, **name_to_arr)

    # (C) Keras H5 -> weights.h5
    h5_path = os.path.join(ROOT, "weights.h5")
    model.save_weights(h5_path)

    # Small meta (for debugging offline)
    meta = {
        "num_tensors": len(weights_list),
        "input_len": T,
        "npz_keys": sorted(list(name_to_arr.keys()))[:10],
    }
    with open(os.path.join(ROOT, "weights_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[OK] Wrote:\n - {dat_path}\n - {npz_path}\n - {h5_path}")
    print(f"[INFO] Tensors: {len(weights_list)}, input_len={T}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
