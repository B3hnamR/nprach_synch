"""Global experiment parameters for nprach_synch.

These constants centralize numerical settings shared across the notebooks and
the end-to-end simulator. All units are SI unless otherwise noted. Adjust them
if you need to model a different deployment scenario.
"""

from __future__ import annotations

from typing import Any, Dict

# ---------------------------------------------------------------------------
# Radio configuration
# ---------------------------------------------------------------------------
# NB-IoT uplink typically occupies sub-GHz licensed spectrum. We pick 920 MHz as
# a representative carrier frequency that lines up with the assumptions used in
# the documentation and notebooks.
CARRIER_FREQ: float = 920e6  # Hz

# NPRACH spans 180 kHz with a 3.75 kHz subcarrier spacing. The simulator uses
# this value as the sampling frequency for all time-domain operations.
SAMPLING_FREQUENCY: float = 180e3  # Hz

# ---------------------------------------------------------------------------
# Link budget / waveform normalization
# ---------------------------------------------------------------------------
# Transmit power is handled in dB relative to a unitless baseband waveform.
# Selecting 0 dB keeps the average symbol energy near unity and avoids numerical
# overflow. Increase/decrease to simulate stronger or weaker UEs.
TX_POWER_DB: float = 0.0  # dB

# Noise spectral density (linear value computed via 10 ** (N0_DB / 10)). A value
# around -10 dB yields a moderate SNR regime for both baseline and DL models.
N0_DB: float = -10.0  # dB

# ---------------------------------------------------------------------------
# Channel / mobility parameters
# ---------------------------------------------------------------------------
# Maximum UE speed used when sampling the 3GPP UMi channel topology.
MAX_SPEED: float = 30.0  # m/s (~108 km/h)

# Bounds for the tapped-delay-line representation passed to ApplyTimeChannel.
MIN_L: int = 0
MAX_L: int = 4

# ---------------------------------------------------------------------------
# Synchronization-specific ranges
# ---------------------------------------------------------------------------
# CFO range (training) in parts-per-million.
MAX_CFO_PPM_TRAIN: float = 20.0  # +/- ppm

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def as_dict() -> Dict[str, Any]:
    """Expose the parameter set as a plain dictionary (useful for logging)."""
    return {
        "CARRIER_FREQ": CARRIER_FREQ,
        "SAMPLING_FREQUENCY": SAMPLING_FREQUENCY,
        "TX_POWER_DB": TX_POWER_DB,
        "N0_DB": N0_DB,
        "MAX_SPEED": MAX_SPEED,
        "MIN_L": MIN_L,
        "MAX_L": MAX_L,
        "MAX_CFO_PPM_TRAIN": MAX_CFO_PPM_TRAIN,
    }

__all__ = [
    "CARRIER_FREQ",
    "SAMPLING_FREQUENCY",
    "TX_POWER_DB",
    "N0_DB",
    "MAX_SPEED",
    "MIN_L",
    "MAX_L",
    "MAX_CFO_PPM_TRAIN",
    "as_dict",
]
