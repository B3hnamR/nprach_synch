#!/usr/bin/env python3
"""Flexible loader for DeepNSynch weights: .npz, .h5, .dat"""
from __future__ import annotations
import os, sys, pickle
import numpy as np
import tensorflow as tf

def load_weights_flex(model: tf.keras.Model, path: str) -> int:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        data = np.load(path, allow_pickle=False)
        name_to_var = {v.name: v for v in model.weights}
        assigned = 0
        for k in data.files:
            arr = data[k]
            if k in name_to_var and tuple(arr.shape) == tuple(name_to_var[k].shape):
                name_to_var[k].assign(arr); assigned += 1
        print(f"[NPZ] Assigned {assigned}/{len(model.weights)} tensors")
        return assigned

    if ext in (".h5", ".hdf5"):
        model.load_weights(path)
        print("[H5] Loaded via Keras save_weights")
        return len(model.weights)

    if ext == ".dat":
        with open(path, "rb") as f:
            lst = pickle.load(f)
        model.set_weights(lst)
        print("[DAT] Loaded via set_weights(list-of-arrays)")
        return len(lst)

    raise ValueError("Unsupported weights format (use .npz, .h5, or .dat)")
