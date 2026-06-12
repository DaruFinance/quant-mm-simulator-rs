#!/usr/bin/env python3
"""Cross-language parity for quote-shape primitives.

Runs all 5 shape primitives in Python and Rust on a battery of
hand-picked (spec, ref_price, inv) cases — including the 5 G3
hand-reconciled arrays — and diffs the produced QuoteRequest lists
field-by-field.

Each primitive is pure of (spec, ref_price, inv), so parity is
EXACT for every field; no aggregate tolerance needed.

Single-threaded.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

REPO_RUST = Path(__file__).resolve().parent.parent
REPO_PY = Path(os.environ.get(
    "MMSIM_PY_DIR", REPO_RUST.parent / "quant-mm-simulator"))


RUST_DRIVER = r'''
//! Cross-language parity harness binary.
//! Generated at runtime by tools/parity_shapes.py.

#![cfg(feature = "quoter")]

use quant_mm_simulator_rs::quoter::shapes::{
    dynamic_depth, geometric, ladder, paired, single,
    DynamicDepthSpec, GeometricSpec, LadderSpec, PairedSpec, SingleSpec,
};

fn dump(label: &str, quotes: &[quant_mm_simulator_rs::sim::sim_loop::QuoteRequest]) {
    println!("{}_n={}", label, quotes.len());
    for (i, q) in quotes.iter().enumerate() {
        println!("{}_{}_side={}", label, i, q.side);
        println!("{}_{}_price={:.12}", label, i, q.price);
        println!("{}_{}_size={:.12}", label, i, q.size);
    }
}

fn main() {
    // Case 1 — single
    dump("single@100",
        &single(&SingleSpec { size: 0.001, half_spread: 0.5 }, 100.0, 0.0));

    // Case 2 — paired_2x2
    dump("paired_2x2",
        &paired(&PairedSpec {
            levels_bid: vec![(0.5, 0.001), (1.0, 0.002)],
            levels_ask: vec![(0.5, 0.001), (1.0, 0.002)],
        }, 50.0, 0.0));

    // Case 3 — ladder_3x_step_0.2
    dump("ladder_3x",
        &ladder(&LadderSpec {
            half_spread: 1.0, step: 0.2, n_levels: 3, size_per_level: 0.005,
        }, 200.0, 0.0));

    // Case 4 — geometric_ratio_1.5
    dump("geometric_1.5",
        &geometric(&GeometricSpec {
            half_spread: 1.0, ratio: 1.5, n_levels: 3, size_per_level: 0.005,
        }, 10.0, 0.0));

    // Case 5 — dynamic_depth tapered at 2x threshold
    dump("dyn_depth_tapered",
        &dynamic_depth(&DynamicDepthSpec {
            half_spread: 0.5, step: 0.1, max_levels: 5,
            inv_taper_threshold: 1.0, size_per_level: 0.001,
        }, 100.0, 3.0));

    // Case 6 — dynamic_depth at zero inv (full depth)
    dump("dyn_depth_zero",
        &dynamic_depth(&DynamicDepthSpec {
            half_spread: 0.5, step: 0.1, max_levels: 5,
            inv_taper_threshold: 1.0, size_per_level: 0.001,
        }, 100.0, 0.0));

    // Case 7 — geometric at non-trivial ratio + bigger N
    dump("geometric_1.2_n5",
        &geometric(&GeometricSpec {
            half_spread: 0.25, ratio: 1.2, n_levels: 5, size_per_level: 0.002,
        }, 1000.0, 0.0));
}
'''


def run_rust() -> Dict[str, str]:
    src = REPO_RUST / "examples" / "_parity_shapes.rs"
    src.write_text(RUST_DRIVER)
    build = subprocess.run(
        ["cargo", "build", "--jobs", "1", "--release",
         "--features", "quoter", "--example", "_parity_shapes"],
        cwd=REPO_RUST, capture_output=True, text=True, timeout=600,
    )
    if build.returncode != 0:
        sys.stderr.write(build.stderr[-2000:]); sys.exit(2)
    bin_path = REPO_RUST / "target" / "release" / "examples" / "_parity_shapes"
    proc = subprocess.run(
        [str(bin_path)], cwd=REPO_RUST,
        capture_output=True, text=True, timeout=60,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr[-2000:]); sys.exit(2)
    out: Dict[str, str] = {}
    for line in proc.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1); out[k] = v
    return out


def run_python() -> Dict[str, str]:
    driver = f'''
import sys
sys.path.insert(0, {str(REPO_PY)!r})
from mmsim.quoter.shapes import (
    single, paired, ladder, geometric, dynamic_depth,
    SingleSpec, PairedSpec, LadderSpec, GeometricSpec, DynamicDepthSpec,
)

def dump(label, quotes):
    print(f"{{label}}_n={{len(quotes)}}")
    for i, q in enumerate(quotes):
        print(f"{{label}}_{{i}}_side={{q.side}}")
        print(f"{{label}}_{{i}}_price={{q.price:.12f}}")
        print(f"{{label}}_{{i}}_size={{q.size:.12f}}")

dump("single@100",
    single(SingleSpec(size=0.001, half_spread=0.5), 100.0, 0.0))

dump("paired_2x2",
    paired(PairedSpec(
        levels_bid=((0.5, 0.001), (1.0, 0.002)),
        levels_ask=((0.5, 0.001), (1.0, 0.002)),
    ), 50.0, 0.0))

dump("ladder_3x",
    ladder(LadderSpec(
        half_spread=1.0, step=0.2, n_levels=3, size_per_level=0.005,
    ), 200.0, 0.0))

dump("geometric_1.5",
    geometric(GeometricSpec(
        half_spread=1.0, ratio=1.5, n_levels=3, size_per_level=0.005,
    ), 10.0, 0.0))

dump("dyn_depth_tapered",
    dynamic_depth(DynamicDepthSpec(
        half_spread=0.5, step=0.1, max_levels=5,
        inv_taper_threshold=1.0, size_per_level=0.001,
    ), 100.0, 3.0))

dump("dyn_depth_zero",
    dynamic_depth(DynamicDepthSpec(
        half_spread=0.5, step=0.1, max_levels=5,
        inv_taper_threshold=1.0, size_per_level=0.001,
    ), 100.0, 0.0))

dump("geometric_1.2_n5",
    geometric(GeometricSpec(
        half_spread=0.25, ratio=1.2, n_levels=5, size_per_level=0.002,
    ), 1000.0, 0.0))
'''
    py = os.environ.get("MMSIM_PY", sys.executable)
    proc = subprocess.run([py, "-c", driver],
                            capture_output=True, text=True, timeout=60)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr[-2000:]); sys.exit(2)
    out: Dict[str, str] = {}
    for line in proc.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1); out[k] = v
    return out


def main() -> int:
    rs = run_rust()
    py = run_python()
    keys = sorted(set(rs.keys()) | set(py.keys()))
    failures: List[str] = []
    for k in keys:
        if k not in rs or k not in py:
            print(f"  {k}: MISSING (py={k in py}, rs={k in rs})")
            failures.append(k); continue
        if py[k] == rs[k]:
            print(f"  {k}: py={py[k]} rs={rs[k]} EXACT"); continue
        print(f"  {k}: py={py[k]} rs={rs[k]} [FAIL]")
        failures.append(k)
    if failures:
        print(f"\nSHAPES PARITY FAILED: {len(failures)} mismatch(es)")
        return 1
    print("\nSHAPES PARITY OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
