<!-- SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: LicenseRef-NvidiaProprietary

NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
property and proprietary rights in and to this material, related
documentation and any modifications thereto. Any use, reproduction,
disclosure or distribution of this material and related documentation
without an express license agreement from NVIDIA CORPORATION or
its affiliates is strictly prohibited. -->

# Deep Learning-Based Synchronization for Uplink NB-IoT

[راهنمای فارسی (Persian README)](./README.fa.md)

Implementation of the NPRACH detection algorithm from
[[A]](https://arxiv.org/abs/2205.10805) using the
[Sionna link-level simulator](https://nvlabs.github.io/sionna/).

This repository implements two synchronization methods for NB-IoT NPRACH:
- A deep learning-based approach (DeepNSynch)
- A strong analytical baseline (NPRACHSynch)

It also provides end-to-end simulation of NPRACH generation, channel modeling (3GPP UMi), and synchronization/evaluation.

## Table of Contents
- Overview
- Features
- Requirements
- Quickstart
- Runtime Auto-Configuration (new)
- Project Structure
- Notebooks: Train and Evaluate
- CPU/GPU notes, XLA, and plotting
- Smoke Test (baseline-only)
- Troubleshooting
- References
- Credits and License

## Overview
We propose a neural network (NN)-based algorithm for device detection and time of arrival (ToA) and carrier frequency offset (CFO) estimation for the narrowband physical random-access channel (NPRACH) of NB-IoT. The NN architecture leverages residual convolutional networks as well as knowledge of the preamble structure of the 5G NR specifications. The method is benchmarked against a strong analytical baseline.

## Features
- NPRACH waveform implementation (preamble configuration 0)
- End-to-end simulation with 3GPP UMi channel (Sionna)
- Two synchronization methods: deep learning and baseline
- Reproducible environment via pinned requirements
- New: Runtime auto-configuration that adapts execution to system capabilities (CPU-only vs GPU)

## Requirements
We recommend Ubuntu 20.04 (or WSL2 on Windows), Python 3.8, and TensorFlow 2.8.

Pinned dependencies are provided in `requirements.txt` (Python 3.8 target):
- tensorflow==2.8.4
- sionna==0.13.0
- numpy==1.22.4
- scipy==1.8.1
- matplotlib==3.5.3
- jupyter==1.0.0
- ipympl==0.9.3 (optional for interactive plots)
- protobuf==3.19.6
- gdown==4.7.1 (optional; not required if generating weights locally)

## Quickstart
1) Create and activate a virtual environment (Python 3.8):
```
# Linux/WSL
python3.8 -m venv .venv && source .venv/bin/activate

# Windows PowerShell
py -3.8 -m venv .venv && .venv\Scripts\activate
```

2) Install dependencies:
```
pip install --upgrade pip
pip install -r requirements.txt
```

3) (Optional, for DeepNSynch) Prepare model weights:
```
# Option A: Use your own weights file (place at project root as weights.dat)
# Option B: Generate a shape‑compatible weights file locally
python scripts/generate_weights.py
# Optionally verify weights load and a dummy forward pass
python scripts/verify_weights.py weights.dat
```

4) Launch Jupyter:
```
jupyter notebook
```
Open `Evaluate.ipynb` or `Train.ipynb`.

## Runtime Auto-Configuration (new)
We added `runtime/auto_config.py` to detect system resources (GPU/CPU/RAM/OS) and recommend safe, efficient settings: batch sizes, mixed precision, TF threads, tf.data autotune, and plotting backend.

Paste at the top of your notebook:
```
from runtime.auto_config import get_system_profile, recommend_settings, apply_tf_settings, summarize

prof = get_system_profile()
rec  = recommend_settings(prof, mode='eval')  # or 'train'
apply_tf_settings(rec)
print(summarize(prof, rec))

# Optionally propagate suggested sizes and jit flag
BATCH_SIZE_TRAIN = rec.batch_size_train
BATCH_SIZE_EVAL  = rec.batch_size_eval
USE_XLA = rec.jit_compile  # defaults False for broad compatibility
```
Bind your tf.function decorators to the recommended flag if desired:
```
@tf.function(jit_compile=USE_XLA)
def sample_sys_snr(...):
    ...
```
Heuristics:
- GPU: mixed precision on, XLA off by default, batch sizes ~64, conservative threads
- CPU-only / low-RAM: mixed precision off, XLA off, smaller batch sizes (4–16), conservative threads
- tf.data: AUTOTUNE for map/parallels and prefetch
- Matplotlib: inline backend to avoid hard dependency on ipympl

## Project Structure
- `nprach/`: NPRACH waveform implementation
- `synch/`: Synchronization algorithms (DeepNSynch, NPRACHSynch)
- `e2e/`: End-to-end system modeling (NPRACH generation, channel, synchronization)
- `runtime/`: Runtime helpers
  - `auto_config.py`: auto detection and recommended TF/runtime settings (new)
- `parameters.py`: Global parameters (batch sizes, CFO ranges, etc.)
- `results/`: Output artifacts created by Evaluate.ipynb
- `Train.ipynb`: Training loop for DeepNSynch
- `Evaluate_prepared.ipynb`: Ready-to-run baseline/DL curves with saved figures
- `Evaluate.ipynb`: Benchmarks DeepNSynch vs baseline and reproduces paper plots
- `scripts/generate_weights.py`: Builds a minimal DeepNSynch and writes weights in .dat/.npz/.h5
- `scripts/verify_weights.py`: Loads weights and runs a dummy forward for sanity
- `scripts/train_deepnsynch.py`: Warm‑build and save weights (extendable to real training)
- `CHANGELOG.md`: All changes and rationales

## Notebooks: Train and Evaluate
- Training (`Train.ipynb`):
  - Default `jit_compile=False` for portability; enable only if your TF/XLA/CUDA stack is compatible.
  - Consider using `runtime/auto_config.py` to pick batch sizes and threads.
- Evaluation (`Evaluate.ipynb`):
  - You need `weights.dat` in project root to evaluate the deep model.
  - Baseline can be evaluated without weights.
  - Plotting uses inline backend by default; install `ipympl` if you want interactive widgets and switch back to `%matplotlib widget`.

## CPU/GPU notes, XLA, and plotting
- XLA is disabled by default in notebooks for maximum compatibility (Windows/CPU-only can fail with XLA).
- On Windows, prefer WSL2 (Ubuntu 20.04). For native Windows, use TF 2.8 CPU-only and keep XLA disabled.
- If `ipympl` is not installed, the notebooks use `%matplotlib inline`.

To re-enable XLA in a compatible environment:
```
USE_XLA = True
@tf.function(jit_compile=USE_XLA)
def my_fn(...):
    ...
```

## Smoke Test (baseline-only)
You can quickly verify TF/Sionna compatibility without weights using the baseline path in `Evaluate.ipynb`:
- Set system to baseline (`E2E('baseline', False, ...)`) with `pfa=0.999`
- Use a small batch (`BATCH_SIZE_EVAL`) and `max_cfo_ppm=10., ue_prob=0.5`
If there is a version mismatch or Sionna/TF setup issue, this test will fail early.

Alternatively, run the CLI smoke test:
```
python scripts/smoke_test.py
```

## Troubleshooting
- FileNotFoundError: `weights.dat`
  - Place your weights at project root, or run `python scripts/generate_weights.py`, then verify with `python scripts/verify_weights.py weights.dat`.
- Sionna import or API errors
  - Use the pinned `sionna==0.13.0` with `tensorflow==2.8.4` as specified.
- XLA Unimplemented/Unsupported
  - Keep `jit_compile=False` (default). XLA is optional and environment-dependent.
- Matplotlib widget errors
  - Install `ipympl` or keep `%matplotlib inline`.
- GPU not detected
  - Verify correct TF build, CUDA/cuDNN versions, and driver installation. Otherwise run in CPU mode with default settings.

## References
[A] F. Aït Aoudia, J. Hoydis, S. Cammerer, M. Van Keirsbilck, and A. Keller, "Deep Learning-Based Synchronization for Uplink NB-IoT", 2022. https://arxiv.org/abs/2205.10805

[B] H. Chougrani, S. Kisseleff and S. Chatzinotas, "Efficient Preamble Detection and Time-of-Arrival Estimation for Single-Tone Frequency Hopping Random Access in NB-IoT," in IEEE IoT Journal, 8(9):7437-7449, 2021. https://ieeexplore.ieee.org/abstract/document/9263250/

## Credits
- Project maintenance, updates, and documentation: Behnam
- Original implementation and paper references as cited above

## License
Copyright © 2022, NVIDIA Corporation. All rights reserved.

This work is made available under the [Nvidia License](LICENSE.txt).
