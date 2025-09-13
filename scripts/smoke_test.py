#!/usr/bin/env python3
"""Quick smoke test for baseline path (CPU-only friendly).

Runs a tiny E2E baseline simulation to catch obvious env/version issues.
"""
from __future__ import annotations
import sys

try:
    from e2e import E2E
except Exception as e:
    print("[ERR] Failed to import E2E:", e)
    raise SystemExit(1)

def main() -> int:
    BATCH = 8
    try:
        sysm = E2E('baseline', False, nprach_num_rep=1, nprach_num_sc=24, fft_size=256, pfa=0.999)
        out = sysm(BATCH, max_cfo_ppm=10.0, ue_prob=0.5)
        if not isinstance(out, (tuple, list)) or len(out) != 8:
            print("[ERR] Unexpected output shape from E2E baseline:", type(out))
            return 2
        print("[OK] Baseline smoke test passed. Outputs:")
        for i, t in enumerate(out):
            try:
                print(f"  [{i}] {t.shape} {t.dtype}")
            except Exception:
                print(f"  [{i}] {type(t)}")
        return 0
    except Exception as e:
        print("[ERR] Baseline smoke test failed:", e)
        return 3

if __name__ == "__main__":
    raise SystemExit(main())

