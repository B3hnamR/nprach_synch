"""
Runtime auto-configuration for nprach_synch

This module detects available system resources (GPU/CPU/RAM/OS) and derives
recommended TensorFlow and runtime settings for stable and efficient execution
across a wide range of machines (from CPU-only laptops to high-end GPUs).

Usage patterns (high-level):
- Import and call get_system_profile() and recommend_settings().
- Optionally call apply_tf_settings() before building models to apply thread and
  mixed-precision settings.
- Consume recommended flags (e.g., jit_compile, batch sizes) in notebooks or
  scripts.

Notes:
- This module avoids hard failures if optional libraries (e.g., psutil) are not
  present by using best-effort detection.
- XLA is not forced on here; enabling it should be an explicit choice based on
  your TF/XLA/CUDA stack.
"""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

try:
    import psutil  # type: ignore
except Exception:
    psutil = None  # Optional

import tensorflow as tf


@dataclass
class SystemProfile:
    has_gpu: bool
    gpu_names: Optional[list[str]]
    cpu_count: int
    ram_gb: Optional[float]
    os: str
    tf_version: str


@dataclass
class RuntimeSettings:
    use_gpu: bool
    mixed_precision: bool
    jit_compile: bool
    batch_size_train: int
    batch_size_eval: int
    inter_op_threads: int
    intra_op_threads: int
    tfdata_parallel_calls: Any  # typically tf.data.AUTOTUNE
    tfdata_prefetch: Any        # typically tf.data.AUTOTUNE
    matplotlib_backend: str     # 'inline' or 'widget'


def _detect_gpu_names() -> list[str]:
    names = []
    try:
        for d in tf.config.list_physical_devices('GPU'):
            # Some TF builds expose device details via experimental API
            try:
                details = tf.config.experimental.get_device_details(d)  # type: ignore
                name = details.get('device_name') or details.get('gpu_model') or str(d)
            except Exception:
                name = str(d)
            names.append(name)
    except Exception:
        pass
    return names


def get_system_profile() -> SystemProfile:
    gpu_devices = tf.config.list_physical_devices('GPU')
    has_gpu = len(gpu_devices) > 0
    cpu_count = os.cpu_count() or 1
    ram_gb = None
    if psutil is not None:
        try:
            ram_gb = psutil.virtual_memory().total / (1024 ** 3)
        except Exception:
            ram_gb = None

    return SystemProfile(
        has_gpu=has_gpu,
        gpu_names=_detect_gpu_names() if has_gpu else None,
        cpu_count=int(cpu_count),
        ram_gb=ram_gb,
        os=platform.system().lower(),
        tf_version=tf.__version__,
    )


def recommend_settings(profile: SystemProfile, mode: str = 'eval') -> RuntimeSettings:
    """
    Provide practical defaults depending on system capabilities.

    Heuristics:
    - CPU-only or low-RAM: small batch sizes, disable mixed precision & XLA.
    - GPU present: enable mixed precision; keep XLA off by default unless the user
      explicitly enables it in notebooks.
    - Threads: conservative choice based on CPU count.
    - tf.data: use AUTOTUNE for parallel calls & prefetch.
    - Matplotlib: prefer 'inline' to avoid ipympl dependency.
    """
    use_gpu = bool(profile.has_gpu)

    # Batch sizes
    # Conservative defaults that should work broadly. Adjust up/down if needed.
    if use_gpu:
        bs_train = 64
        bs_eval = 64
    else:
        # CPU-only defaults
        bs_train = 8
        bs_eval = 16
        # Further scale down if RAM is low
        if profile.ram_gb is not None and profile.ram_gb < 8:
            bs_train = 4
            bs_eval = 8

    # Mixed precision typically benefits modern NVIDIA GPUs
    mixed_precision = use_gpu

    # XLA: default False for maximum compatibility
    jit_compile = False

    # Threads
    # Simple heuristic: inter = max(1, cpu//2), intra = max(1, cpu-1)
    inter = max(1, profile.cpu_count // 2)
    intra = max(1, profile.cpu_count - 1)

    settings = RuntimeSettings(
        use_gpu=use_gpu,
        mixed_precision=mixed_precision,
        jit_compile=jit_compile,
        batch_size_train=bs_train,
        batch_size_eval=bs_eval,
        inter_op_threads=inter,
        intra_op_threads=intra,
        tfdata_parallel_calls=tf.data.AUTOTUNE,
        tfdata_prefetch=tf.data.AUTOTUNE,
        matplotlib_backend='inline',
    )
    return settings


def apply_tf_settings(settings: RuntimeSettings) -> None:
    """
    Apply a subset of settings to the active TF runtime.

    - Sets TF threading to recommended values
    - Enables GPU memory growth when GPU is present
    - Applies mixed precision policy if requested
    """
    # Threads
    try:
        tf.config.threading.set_inter_op_parallelism_threads(settings.inter_op_threads)
        tf.config.threading.set_intra_op_parallelism_threads(settings.intra_op_threads)
    except Exception:
        pass

    # GPU memory growth (when GPU present)
    if settings.use_gpu:
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                tf.config.experimental.set_memory_growth(gpus[0], True)  # type: ignore
        except Exception:
            pass

    # Mixed precision
    if settings.mixed_precision:
        try:
            from tensorflow.keras import mixed_precision
            mixed_precision.set_global_policy('mixed_float16')
        except Exception:
            # Silently ignore if mixed precision is unsupported
            pass


def to_dict(settings: RuntimeSettings) -> Dict[str, Any]:
    return asdict(settings)


def summarize(profile: SystemProfile, settings: RuntimeSettings) -> str:
    lines = []
    lines.append("System profile:")
    lines.append(f"  OS: {profile.os}")
    lines.append(f"  TF version: {profile.tf_version}")
    lines.append(f"  CPU count: {profile.cpu_count}")
    lines.append(f"  RAM (GB): {profile.ram_gb:.1f}" if profile.ram_gb else "  RAM (GB): n/a")
    lines.append(f"  Has GPU: {profile.has_gpu}")
    if profile.gpu_names:
        for i, n in enumerate(profile.gpu_names):
            lines.append(f"    GPU[{i}]: {n}")
    lines.append("")
    lines.append("Recommended settings:")
    lines.append(f"  use_gpu: {settings.use_gpu}")
    lines.append(f"  mixed_precision: {settings.mixed_precision}")
    lines.append(f"  jit_compile: {settings.jit_compile}")
    lines.append(f"  batch_size_train: {settings.batch_size_train}")
    lines.append(f"  batch_size_eval: {settings.batch_size_eval}")
    lines.append(f"  inter_op_threads: {settings.inter_op_threads}")
    lines.append(f"  intra_op_threads: {settings.intra_op_threads}")
    lines.append(f"  tfdata_parallel_calls: AUTOTUNE")
    lines.append(f"  tfdata_prefetch: AUTOTUNE")
    lines.append(f"  matplotlib_backend: {settings.matplotlib_backend}")
    return "\n".join(lines)


if __name__ == "__main__":
    prof = get_system_profile()
    rec = recommend_settings(prof)
    print(summarize(prof, rec))
