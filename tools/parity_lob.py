#!/usr/bin/env python3
"""Cross-language parity harness for LOB ingestion.

Converts the bundled parquet fixtures to CSV-flattened wide format
the Rust loader consumes, runs `mmsim.ingest.lob.{load_lob,
reconstruct_book_at}` in Python and the equivalent Rust binary on
the same inputs, then diffs the reconstructed books at five chosen
timestamps.

Tolerance: 0 / EXACT.  Both sides parse f64 from the same CSV text
and the loaders are deterministic; no numerical drift is expected.

Single-threaded by user request: cargo runs with --jobs 1, no
multiprocessing on the Python side.

Usage:
    python tools/parity_lob.py
    python tools/parity_lob.py --snapshots <path> --trades <path>

Exit 0 = parity OK, 1 = mismatch.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

REPO_RUST = Path(__file__).resolve().parent.parent
REPO_PY = Path(os.environ.get(
    "MMSIM_PY_DIR", REPO_RUST.parent / "quant-mm-simulator"))


RUST_DRIVER = r'''
//! Cross-language parity harness binary.
//! Generated at runtime by tools/parity_lob.py.

#![cfg(feature = "ingest")]

use quant_mm_simulator_rs::ingest::{load_lob, reconstruct_book_at};

fn main() {
    let snap = std::env::args().nth(1).expect("snapshots csv");
    let trade = std::env::args().nth(2).expect("trades csv");
    let probe_args: Vec<i64> = std::env::args()
        .skip(3)
        .map(|s| s.parse().expect("probe ts_ns"))
        .collect();

    let stream = load_lob(&snap, &trade, None, None).expect("load_lob");
    println!("n_events={}", stream.len());

    for (i, t) in probe_args.iter().enumerate() {
        match reconstruct_book_at(&stream, *t) {
            Some(b) => {
                println!("probe_{}_t={}", i, t);
                println!("probe_{}_book_ts={}", i, b.ts_ns);
                println!("probe_{}_best_bid={:.12}", i, b.best_bid().unwrap());
                println!("probe_{}_best_ask={:.12}", i, b.best_ask().unwrap());
                println!("probe_{}_mid={:.12}", i, b.mid().unwrap());
                // Top-3 bid/ask sizes.
                for k in 0..3.min(b.bids.len()) {
                    println!("probe_{}_bid_{}_px={:.12}", i, k, b.bids[k].0);
                    println!("probe_{}_bid_{}_sz={:.12}", i, k, b.bids[k].1);
                }
                for k in 0..3.min(b.asks.len()) {
                    println!("probe_{}_ask_{}_px={:.12}", i, k, b.asks[k].0);
                    println!("probe_{}_ask_{}_sz={:.12}", i, k, b.asks[k].1);
                }
            }
            None => {
                println!("probe_{}_t={}", i, t);
                println!("probe_{}_book_ts=-1", i);
            }
        }
    }
}
'''


def parquet_to_csv(snapshots_pq: Path, trades_pq: Path,
                     out_dir: Path) -> tuple[Path, Path]:
    """Convert the parquet snapshots+trades to wide CSV the Rust
    loader expects.  Returns (snapshots_csv, trades_csv)."""
    import pyarrow.parquet as pq
    snap_t = pq.read_table(snapshots_pq)
    trade_t = pq.read_table(trades_pq)
    snaps = snap_t.to_pylist()
    trades = trade_t.to_pylist()
    if not snaps:
        sys.stderr.write("no snapshots in fixture\n")
        sys.exit(2)

    depth = max(len(s["bids"]) for s in snaps)
    snap_csv = out_dir / "snapshots.csv"
    cols = ["ts_ns", "recv_ns", "symbol", "venue", "depth"]
    cols += [f"bid_px_{k}" for k in range(depth)]
    cols += [f"bid_sz_{k}" for k in range(depth)]
    cols += [f"ask_px_{k}" for k in range(depth)]
    cols += [f"ask_sz_{k}" for k in range(depth)]
    with snap_csv.open("w") as fh:
        fh.write(",".join(cols) + "\n")
        for s in snaps:
            row = [str(s["ts_ns"]), str(s["recv_ns"]),
                    s["symbol"], s["venue"], str(s["depth"])]
            bids = list(s["bids"]) + [{"px": 0.0, "sz": 0.0}] * (depth - len(s["bids"]))
            asks = list(s["asks"]) + [{"px": 0.0, "sz": 0.0}] * (depth - len(s["asks"]))
            row += [f"{b['px']:.12f}" for b in bids]
            row += [f"{b['sz']:.12f}" for b in bids]
            row += [f"{a['px']:.12f}" for a in asks]
            row += [f"{a['sz']:.12f}" for a in asks]
            fh.write(",".join(row) + "\n")

    trade_csv = out_dir / "trades.csv"
    cols2 = ["ts_ns", "recv_ns", "symbol", "venue", "price", "size", "side"]
    with trade_csv.open("w") as fh:
        fh.write(",".join(cols2) + "\n")
        for t in trades:
            fh.write(",".join([
                str(t["ts_ns"]), str(t["recv_ns"]),
                t["symbol"], t["venue"],
                f"{t['price']:.12f}", f"{t['size']:.12f}",
                str(t["side"]),
            ]) + "\n")
    return snap_csv, trade_csv


def pick_probes(snap_csv: Path, n: int = 5) -> List[int]:
    """Five timestamps strictly between consecutive snapshot rows
    so reconstruction is non-trivial (must pick the earlier snap)."""
    ts: List[int] = []
    with snap_csv.open() as fh:
        next(fh)  # header
        for line in fh:
            ts.append(int(line.split(",", 1)[0]))
    if len(ts) < n + 1:
        return [t + 1 for t in ts]
    step = len(ts) // (n + 1)
    return [(ts[step * (i + 1)] + ts[step * (i + 1) + 1]) // 2
            for i in range(n)]


def run_rust(snap_csv: Path, trade_csv: Path, probes: List[int]) -> Dict[str, str]:
    src = REPO_RUST / "examples" / "_parity_lob.rs"
    src.write_text(RUST_DRIVER)
    build = subprocess.run(
        ["cargo", "build", "--jobs", "1", "--release",
         "--features", "ingest", "--example", "_parity_lob"],
        cwd=REPO_RUST, capture_output=True, text=True, timeout=600,
    )
    if build.returncode != 0:
        sys.stderr.write(f"Rust build failed:\n{build.stderr[-2000:]}\n")
        sys.exit(2)
    bin_path = REPO_RUST / "target" / "release" / "examples" / "_parity_lob"
    proc = subprocess.run(
        [str(bin_path), str(snap_csv), str(trade_csv), *map(str, probes)],
        cwd=REPO_RUST, capture_output=True, text=True, timeout=120,
    )
    if proc.returncode != 0:
        sys.stderr.write(f"Rust run failed:\n{proc.stderr[-2000:]}\n")
        sys.exit(2)
    out: Dict[str, str] = {}
    for line in proc.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k] = v
    return out


def run_python(snap_csv: Path, trade_csv: Path,
                snapshots_pq: Path, trades_pq: Path,
                probes: List[int]) -> Dict[str, str]:
    """Drive Python load_lob on the parquet originals (the canonical
    Python input format) and emit the same key=value lines.  We use
    the parquets here, not the CSVs, to verify Python's parquet read
    matches Rust's CSV read byte-for-byte through the lossless
    round-trip we did in parquet_to_csv."""
    driver = f'''
import sys
sys.path.insert(0, {str(REPO_PY)!r})
from mmsim.ingest.lob import load_lob, reconstruct_book_at, SnapshotEvent

stream = load_lob({str(snapshots_pq)!r}, {str(trades_pq)!r})
print(f"n_events={{len(stream)}}")

probes = {probes!r}
for i, t in enumerate(probes):
    b = reconstruct_book_at(stream, t)
    if b is None:
        print(f"probe_{{i}}_t={{t}}")
        print(f"probe_{{i}}_book_ts=-1")
        continue
    print(f"probe_{{i}}_t={{t}}")
    print(f"probe_{{i}}_book_ts={{b.ts_ns}}")
    print(f"probe_{{i}}_best_bid={{b.best_bid:.12f}}")
    print(f"probe_{{i}}_best_ask={{b.best_ask:.12f}}")
    print(f"probe_{{i}}_mid={{b.mid:.12f}}")
    for k in range(min(3, len(b.bids))):
        print(f"probe_{{i}}_bid_{{k}}_px={{b.bids[k][0]:.12f}}")
        print(f"probe_{{i}}_bid_{{k}}_sz={{b.bids[k][1]:.12f}}")
    for k in range(min(3, len(b.asks))):
        print(f"probe_{{i}}_ask_{{k}}_px={{b.asks[k][0]:.12f}}")
        print(f"probe_{{i}}_ask_{{k}}_sz={{b.asks[k][1]:.12f}}")
'''
    py = os.environ.get("MMSIM_PY", sys.executable)
    proc = subprocess.run(
        [py, "-c", driver],
        capture_output=True, text=True, timeout=120,
    )
    if proc.returncode != 0:
        sys.stderr.write(f"Python run failed:\n{proc.stderr[-2000:]}\n")
        sys.exit(2)
    out: Dict[str, str] = {}
    for line in proc.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k] = v
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--snapshots", type=Path,
        default=REPO_PY / "tests" / "fixtures" / "lob_btcusdt_30sec_snapshots.parquet",
    )
    parser.add_argument(
        "--trades", type=Path,
        default=REPO_PY / "tests" / "fixtures" / "lob_btcusdt_30sec_trades.parquet",
    )
    args = parser.parse_args()

    if not args.snapshots.exists() or not args.trades.exists():
        sys.stderr.write(
            f"missing fixture(s): {args.snapshots} / {args.trades}\n")
        return 2

    with tempfile.TemporaryDirectory(prefix="mmsim_parity_lob_") as tmp:
        out_dir = Path(tmp)
        snap_csv, trade_csv = parquet_to_csv(args.snapshots, args.trades, out_dir)
        probes = pick_probes(snap_csv, n=5)
        rs = run_rust(snap_csv, trade_csv, probes)
        py = run_python(snap_csv, trade_csv, args.snapshots, args.trades, probes)

    keys = sorted(set(rs.keys()) | set(py.keys()))
    failures: List[str] = []
    for k in keys:
        if k not in rs or k not in py:
            print(f"  {k}: MISSING (py={k in py}, rs={k in rs})")
            failures.append(k)
            continue
        if py[k] == rs[k]:
            print(f"  {k}: py={py[k]} rs={rs[k]} EXACT")
            continue
        try:
            pv, rv = float(py[k]), float(rs[k])
            denom = max(abs(pv), abs(rv), 1e-15)
            rel = abs(pv - rv) / denom
            print(f"  {k}: py={py[k]} rs={rs[k]} rel={rel:.2e} [FAIL]")
        except ValueError:
            print(f"  {k}: py={py[k]} rs={rs[k]} string-mismatch [FAIL]")
        failures.append(k)

    if failures:
        print(f"\nLOB PARITY FAILED: {len(failures)} mismatch(es)")
        return 1
    print("\nLOB PARITY OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
