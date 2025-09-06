#!/usr/bin/env python3
"""
Train DeepNSynch and periodically write weights.dat at the project root.

This script builds the end-to-end model in training mode (E2E('dl', training=True)),
optimizes the three losses (BCE for activity, MSE for ToA and CFO weighted by SNR),
and saves DeepNSynch weights (NOT the whole E2E) as a pickle list into weights.dat
for later evaluation in notebooks.

Usage examples:
  # CPU-only quick sanity run
  python scripts/train_deepnsynch.py --steps 2000 --batch 64 --save-every 500

  # Longer run (recommend GPU/WSL2-CUDA)
  python scripts/train_deepnsynch.py --steps 800000 --batch 64 --save-every 2000

Notes:
- weights.dat contains ONLY the DeepNSynch (sys.synch) weights to be loaded via:
    sys = E2E('dl', False, nprach_num_rep=1, nprach_num_sc=48)
    _ = sys(1, max_cfo_ppm=10., ue_prob=0.5)   # build
    with open('weights.dat','rb') as f: w = pickle.load(f)
    sys.synch.set_weights(w)
- For reproducibility, a fixed seed is set unless --no-seed is provided.
- Uses runtime/auto_config if available to apply safe TF threading settings.
"""
from __future__ import annotations

import os
import sys
import time
import math
import argparse
import pickle
from typing import Optional

import tensorflow as tf

# Make project root importable if invoked from elsewhere
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Optional runtime auto-config
try:
    from runtime.auto_config import get_system_profile, recommend_settings, apply_tf_settings
except Exception:
    get_system_profile = recommend_settings = apply_tf_settings = None  # type: ignore

from e2e import E2E
from parameters import DEEPNSYNCH_WEIGHTS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DeepNSynch and write weights.dat")
    p.add_argument('--steps', type=int, default=20000, help='Total training steps (iterations)')
    p.add_argument('--batch', type=int, default=64, help='Batch size')
    p.add_argument('--lr', type=float, default=1e-3, help='Adam learning rate')
    p.add_argument('--save-every', type=int, default=1000, help='Save weights every N steps')
    p.add_argument('--nprach-num-sc', type=int, default=48, help='Number of NPRACH subcarriers')
    p.add_argument('--nprach-num-rep', type=int, default=1, help='Number of NPRACH repetitions')
    p.add_argument('--no-seed', action='store_true', help='Do not set deterministic seed')
    return p.parse_args()


def save_weights(sys_model: E2E, step: int, out_path: str) -> None:
    """Serialize DeepNSynch weights only (sys.synch)."""
    weights = sys_model.synch.get_weights()
    tmp = out_path + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(weights, f)
    os.replace(tmp, out_path)
    # Also keep a checkpoint in results/
    ckpt_dir = os.path.join(ROOT, 'results')
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f'weights_step{step}.dat')
    try:
        with open(ckpt_path, 'wb') as f:
            pickle.dump(weights, f)
    except Exception:
        pass


def main() -> int:
    args = parse_args()

    if not args.no_seed:
        tf.random.set_seed(42)

    # Apply runtime settings if available
    if get_system_profile and recommend_settings and apply_tf_settings:
        prof = get_system_profile()
        rec = recommend_settings(prof, mode='train')
        apply_tf_settings(rec)
        print("[auto_config] using:")
        print(f"  batch_size_train={rec.batch_size_train}, inter_op={rec.inter_op_threads}, intra_op={rec.intra_op_threads}")

    # Build trainable E2E with DL synchronizer
    sys_model = E2E('dl', True, nprach_num_rep=args.nprach_num_rep, nprach_num_sc=args.nprach_num_sc)

    # Optimizer
    opt = tf.keras.optimizers.Adam(learning_rate=args.lr)

    @tf.function(jit_compile=False)
    def train_step(batch_size: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        with tf.GradientTape() as tape:
            loss_tx, loss_toa, loss_cfo = sys_model(batch_size)
            # Sum of the three heads as in the paper (no extra lambdas)
            loss_total = loss_tx + loss_toa + loss_cfo
        grads = tape.gradient(loss_total, sys_model.trainable_variables)
        opt.apply_gradients(zip(grads, sys_model.trainable_variables))
        return loss_tx, loss_toa, loss_cfo

    # Warm-up build (ensures variables exist)
    _ = sys_model(1)

    batch = tf.constant(args.batch, dtype=tf.int32)
    out_path = os.path.join(ROOT, DEEPNSYNCH_WEIGHTS)

    ema = None  # exponential moving average of losses
    alpha = 0.01
    t0 = time.time()

    for step in range(1, args.steps + 1):
        l1, l2, l3 = train_step(batch)
        l1f, l2f, l3f = float(l1.numpy()), float(l2.numpy()), float(l3.numpy())
        if ema is None:
            ema = [l1f, l2f, l3f]
        else:
            ema = [alpha*l1f + (1-alpha)*ema[0],
                   alpha*l2f + (1-alpha)*ema[1],
                   alpha*l3f + (1-alpha)*ema[2]]

        if step % 50 == 0:
            dt = time.time() - t0
            print(f"step {step:7d} | loss_tx={ema[0]:.6f} loss_toa={ema[1]:.6f} loss_cfo={ema[2]:.6f} | {dt:.1f}s")

        if step % args.save_every == 0:
            save_weights(sys_model, step, out_path)
            print(f"[save] wrote {out_path} (and checkpoint) at step {step}")

    # Final save
    save_weights(sys_model, args.steps, out_path)
    print(f"[done] wrote final {out_path} at step {args.steps}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
