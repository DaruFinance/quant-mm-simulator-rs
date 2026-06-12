#!/usr/bin/env python3
"""Parity / determinism gate for the markout + ledger research layer (H1/H3).

SCOPE NOTE: the markout/ledger layer is currently Python-only (numba
production kernel + pure-Python reference). The Rust port of the markout
kernel is not yet implemented. This script therefore pins the contract that is
live today:

  1. numba markout kernel == pure-Python reference, BIT-IDENTICAL
     (NaN-aware exact) on the 60-min fixture's fills.
  2. permutation-null numba kernel == pure-Python reference, bit-identical
     for a fixed seed.
  3. the markout decomposition identity holds to f64 noise:
     quoted_half_spread == realised_spread + adverse_selection.

When the Rust markout kernel lands, extend this script to diff the Rust
output against the Python reference at <=1e-9 (the deterministic-sim
tolerance), matching the parity_fills.py pattern.

Single-threaded.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

REPO_RUST = Path(__file__).resolve().parent.parent
REPO_PY = Path(os.environ.get(
    "MMSIM_PY_DIR", REPO_RUST.parent / "quant-mm-simulator"))


def main() -> int:
    sys.path.insert(0, str(REPO_PY))
    sys.path.insert(0, str(REPO_PY / "tests"))
    from mmsim.ingest.lob import load_lob
    from mmsim.sim.loop import run_sim
    from mmsim.sim.fills import QueueAwareFillModel
    from mmsim.ledger.writer import build_mid_timeline
    from mmsim.markout.engine import compute_markout, compute_markout_reference
    from mmsim.research.perm_null import _perm_null_reference, _perm_null_numba
    from test_sim_fills import BracketQuoter

    snap = REPO_PY / "tests" / "fixtures" / "lob_btcusdt_60min_snapshots.parquet"
    trade = REPO_PY / "tests" / "fixtures" / "lob_btcusdt_60min_trades.parquet"
    if not snap.exists():
        sys.stderr.write(f"missing fixture {snap}\n"); return 2

    stream = load_lob(str(snap), str(trade))
    res = run_sim(stream, BracketQuoter(taker_every=500), QueueAwareFillModel())
    snap_ts, snap_mid = build_mid_timeline(stream)

    failures = []

    # 1. markout numba vs reference, bit-identical.
    mo_ref = compute_markout_reference(res.fills, snap_ts, snap_mid)
    mo_nb = compute_markout(res.fills, snap_ts, snap_mid, use_numba=True)
    cols = ["mid0", "mid_1s", "mid_10s", "mid_60s",
            "markout_1s", "markout_10s", "markout_60s",
            "realised_spread_1s", "realised_spread_10s", "realised_spread_60s",
            "adverse_1s", "adverse_10s", "adverse_60s"]
    for c in cols:
        eq = np.array_equal(getattr(mo_ref, c), getattr(mo_nb, c), equal_nan=True)
        print(f"  markout.{c}: bit-identical={eq}")
        if not eq:
            failures.append(f"markout.{c}")

    # 2. perm-null numba vs reference, bit-identical for fixed seed.
    mk = mo_nb.markout_10s[~np.isnan(mo_nb.markout_10s)]
    a = _perm_null_reference(mk, 200, 4242)
    b = _perm_null_numba(mk, 200, np.uint64(4242))
    eq = np.array_equal(a, b)
    print(f"  perm_null kernel: bit-identical={eq} max|Δ|={np.max(np.abs(a-b)):.3e}")
    if not eq:
        failures.append("perm_null")

    # 3. decomposition identity to f64 noise.
    fill_px = np.array([f.price for f in res.fills])
    fill_side = np.array([f.side for f in res.fills])
    qhs = fill_side * (fill_px - mo_nb.mid0) / mo_nb.mid0
    ident = mo_nb.realised_spread_10s + mo_nb.adverse_10s
    maxd = float(np.nanmax(np.abs(qhs - ident)))
    print(f"  decomposition identity max|Δ|={maxd:.3e} (tol 1e-9)")
    if maxd > 1e-9:
        failures.append("decomposition_identity")

    if failures:
        print(f"\nMARKOUT PARITY FAILED: {failures}")
        return 1
    print("\nMARKOUT PARITY OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
