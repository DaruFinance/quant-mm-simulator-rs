#!/usr/bin/env python3
"""Cross-language parity for inventory penalties.

All 6 primitives × 7 inv values per side = 42+ probes.  Pure
functions; every key EXACT.

Single-threaded.
"""
from __future__ import annotations

import os, subprocess, sys
from pathlib import Path

REPO_RUST = Path(__file__).resolve().parent.parent
REPO_PY = Path(os.environ.get("MMSIM_PY_DIR", REPO_RUST.parent / "quant-mm-simulator"))

RUST_DRIVER = r'''
#![cfg(feature = "quoter")]
use quant_mm_simulator_rs::quoter::inv_penalty::{
    asymmetric, exponential, hard_cap, linear, quadratic, soft_cap, Skew,
};

fn dump(label: &str, inv: f64, s: Skew) {
    println!("{}_{:.1}_offset={:.12}", label, inv, s.price_offset);
    println!("{}_{:.1}_bid={:.12}", label, inv, s.size_scale_bid);
    println!("{}_{:.1}_ask={:.12}", label, inv, s.size_scale_ask);
}

fn main() {
    let invs: Vec<f64> = vec![-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0];
    for &inv in &invs {
        dump("linear", inv, linear(inv, 0.5));
        dump("quadratic", inv, quadratic(inv, 0.5));
        dump("exponential", inv, exponential(inv, 0.5, 1.0));
        dump("asymmetric", inv, asymmetric(inv, 0.5, 2.0));
        dump("soft_cap", inv, soft_cap(inv, 0.5, 2.0));
        dump("hard_cap", inv, hard_cap(inv, 2.0));
    }
}
'''


def run_rust():
    src = REPO_RUST / "examples" / "_parity_inv_penalty.rs"
    src.write_text(RUST_DRIVER)
    b = subprocess.run(["cargo", "build", "--jobs", "1", "--release",
                          "--features", "quoter", "--example", "_parity_inv_penalty"],
                         cwd=REPO_RUST, capture_output=True, text=True, timeout=600)
    if b.returncode != 0:
        sys.stderr.write(b.stderr[-2000:]); sys.exit(2)
    bin_path = REPO_RUST / "target" / "release" / "examples" / "_parity_inv_penalty"
    p = subprocess.run([str(bin_path)], cwd=REPO_RUST,
                         capture_output=True, text=True, timeout=60)
    if p.returncode != 0:
        sys.stderr.write(p.stderr[-2000:]); sys.exit(2)
    return dict(line.split("=", 1) for line in p.stdout.splitlines() if "=" in line)


def run_python():
    driver = f'''
import sys
sys.path.insert(0, {str(REPO_PY)!r})
from mmsim.quoter.inv_penalty import (
    linear, quadratic, exponential, asymmetric, soft_cap, hard_cap,
)

def dump(label, inv, s):
    print(f"{{label}}_{{inv}}_offset={{s.price_offset:.12f}}")
    print(f"{{label}}_{{inv}}_bid={{s.size_scale_bid:.12f}}")
    print(f"{{label}}_{{inv}}_ask={{s.size_scale_ask:.12f}}")

for inv in [-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0]:
    dump("linear", inv, linear(inv, 0.5))
    dump("quadratic", inv, quadratic(inv, 0.5))
    dump("exponential", inv, exponential(inv, 0.5, 1.0))
    dump("asymmetric", inv, asymmetric(inv, 0.5, 2.0))
    dump("soft_cap", inv, soft_cap(inv, 0.5, 2.0))
    dump("hard_cap", inv, hard_cap(inv, 2.0))
'''
    py = os.environ.get("MMSIM_PY", sys.executable)
    p = subprocess.run([py, "-c", driver], capture_output=True, text=True, timeout=60)
    if p.returncode != 0:
        sys.stderr.write(p.stderr[-2000:]); sys.exit(2)
    return dict(line.split("=", 1) for line in p.stdout.splitlines() if "=" in line)


def main() -> int:
    rs = run_rust(); py = run_python()
    keys = sorted(set(rs) | set(py))
    fails = []
    for k in keys:
        if rs.get(k) == py.get(k):
            print(f"  {k}: EXACT"); continue
        print(f"  {k}: py={py.get(k)} rs={rs.get(k)} [FAIL]"); fails.append(k)
    if fails:
        print(f"\nINV_PENALTY PARITY FAILED ({len(fails)} mismatches)"); return 1
    print("\nINV_PENALTY PARITY OK"); return 0


if __name__ == "__main__":
    sys.exit(main())
