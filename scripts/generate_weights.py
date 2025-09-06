#!/usr/bin/env python3
"""
Generate a valid weights.dat (pickle of Keras weights) for DeepNSynch without
requiring Sionna/Scipy/NumPy at generation time.

This script:
- Stubs a minimal `sionna` module if missing (utils.log10 and PI)
- Builds a dummy NPRACH generator exposing `config`, `seq_indices`, `freq_patterns`
  with correct shapes for preamble format 0 (nprach_dft_size=48, 4 SG, 5 seq/SG)
- Instantiates `synch.DeepNSynch`, runs a single forward pass on zeros to
  initialize variables, then pickles `model.get_weights()` into weights.dat
  at the project root.

Resulting weights are random-initialized and only intended to satisfy
loading/shape requirements (NOT trained).
"""
from __future__ import annotations

import os
import sys
import types
import pickle
import math
from typing import Any

import tensorflow as tf

# 0) Make project root importable if script is run from elsewhere
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# 1) Stub minimal sionna if not installed
try:
    import sionna as sn  # type: ignore
    # Validate required attributes
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

# 2) Dummy NPRACH generator with correct shapes for DeepNSynch
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
    # Numeric lengths at NPRACH bandwidth discretization (not fs):
    # samples_per_seq = int(nprach_seq_duration * bandwidth) = 48
    # samples_per_cp  = int(nprach_cp_duration  * bandwidth) = 12
    samples_per_seq = 48
    samples_per_cp  = 12
    samples_per_sg  = samples_per_cp + cfg.nprach_seq_per_sg * samples_per_seq  # 12 + 5*48 = 252
    num_sg = cfg.nprach_sg_per_rep * cfg.nprach_num_rep  # 4

    # seq_indices: [num_seq, samples_per_seq] == [20, 48]
    base = tf.range(samples_per_seq, dtype=tf.int32)[tf.newaxis, :]  # [1,48]
    shifts = tf.range(num_sg, dtype=tf.int32) * samples_per_sg + samples_per_cp  # [4]
    shifts = tf.repeat(shifts, cfg.nprach_seq_per_sg)  # 20
    seq_indices = base + shifts[:, tf.newaxis]  # [20,48]

    # freq_patterns: [num_sc, num_sg] in [0, dft_size)
    num_sc = cfg.nprach_num_sc
    dft = cfg.nprach_dft_size
    k = tf.range(num_sc, dtype=tf.int32)[:, tf.newaxis]        # [48,1]
    m = tf.range(num_sg, dtype=tf.int32)[tf.newaxis, :]        # [1,4]
    freq_patterns = (k + m) % dft                              # [48,4]

    return _DummyGen(cfg, seq_indices, freq_patterns)


def main() -> int:
    # Set deterministic seed for reproducibility of initial weights
    tf.random.set_seed(42)

    dummy = _build_dummy_gen()

    # Import model after sionna stub is in place
    from synch import DeepNSynch  # type: ignore

    model = DeepNSynch(dummy)

    # Compute input length in NPRACH bandwidth samples: T = 4 * (12 + 5*48) = 1008
    cfg = dummy.config
    samples_per_seq = 48
    samples_per_cp  = 12
    samples_per_sg  = samples_per_cp + cfg.nprach_seq_per_sg * samples_per_seq
    num_sg = cfg.nprach_sg_per_rep * cfg.nprach_num_rep
    T = num_sg * samples_per_sg

    # One forward pass on zeros to build variables
    y = tf.complex(tf.zeros([1, T], tf.float32), tf.zeros([1, T], tf.float32))
    _ = model(y)

    # Serialize weights to project root
    weights = model.get_weights()
    out_path = os.path.join(ROOT, "weights.dat")
    with open(out_path, "wb") as f:
        pickle.dump(weights, f)

    print(f"Wrote {out_path} with {len(weights)} arrays; input_len={T}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
