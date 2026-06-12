#!/usr/bin/env python3
"""Cross-language parity harness for queue tracker.

Places a TOB bid at DS-LOB-1H's first snapshot, runs
``track_queue_position`` over the rest of the hour in both Python
and Rust, and diffs:
  - sample count
  - final queue_pos
  - total_fills_ahead
  - total_cancels_ahead
  - frozen flag
  - per-sample (ts, queue_pos, cause) at five evenly-spaced indices

Tolerance: integers / strings / counts EXACT; floats EXACT for
per-sample queue_pos (no aggregation), 1e-9 for the cumulative
totals (sum-order f64 noise).

Single-threaded.

Usage:
    python tools/parity_queue.py
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
//! Generated at runtime by tools/parity_queue.py.

#![cfg(feature = "sim")]

use quant_mm_simulator_rs::ingest::{load_lob, Book, Event, SnapshotEvent};
use quant_mm_simulator_rs::sim::sim_loop::Order;
use quant_mm_simulator_rs::sim::queue::track_queue_position;

fn main() {
    let snap = std::env::args().nth(1).expect("snapshots csv");
    let trade = std::env::args().nth(2).expect("trades csv");

    let stream = load_lob(&snap, &trade, None, None).expect("load_lob");

    // Find first snapshot, place a TOB bid at its best_bid with size 0.001.
    let first_snap_idx = stream.iter().position(|e| matches!(e, Event::Snapshot(_)))
        .expect("no snapshot in stream");
    let first_snap: &SnapshotEvent = match &stream[first_snap_idx] {
        Event::Snapshot(s) => s,
        _ => unreachable!(),
    };
    let best_bid = first_snap.bids[0].0;
    let initial_book = Book {
        ts_ns: first_snap.ts_ns,
        bids: first_snap.bids.clone(),
        asks: first_snap.asks.clone(),
    };
    let order = Order {
        order_id: 0, side: 1, price: best_bid, size: 0.001,
        placed_at_ns: first_snap.ts_ns,
    };
    let post: Vec<Event> = stream.iter()
        .filter(|e| e.ts_ns() > first_snap.ts_ns)
        .cloned()
        .collect();

    let trace = track_queue_position(order, &post, &initial_book).expect("trace");

    println!("n_samples={}", trace.samples.len());
    println!("final_queue_pos={:.12}", trace.final_queue_pos);
    println!("total_fills_ahead={:.12}", trace.total_fills_ahead);
    println!("total_cancels_ahead={:.12}", trace.total_cancels_ahead);
    println!("frozen={}", trace.frozen);
    println!("initial_queue_pos={:.12}", trace.samples[0].queue_pos);

    // 5 evenly-spaced sample probes.
    let n = trace.samples.len();
    let probes: Vec<usize> = if n < 5 {
        (0..n).collect()
    } else {
        vec![0, n / 4, n / 2, 3 * n / 4, n - 1]
    };
    for (k, idx) in probes.iter().enumerate() {
        let s = &trace.samples[*idx];
        println!("sample_{}_idx={}", k, idx);
        println!("sample_{}_ts={}", k, s.ts_ns);
        println!("sample_{}_queue_pos={:.12}", k, s.queue_pos);
        println!("sample_{}_cause={}", k, s.cause);
    }
}
'''


def parquet_to_csv(snapshots_pq: Path, trades_pq: Path,
                     out_dir: Path) -> tuple[Path, Path]:
    """Same converter as parity_lob/parity_sim_loop; copy here so the
    script is self-contained."""
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


def run_rust(snap_csv: Path, trade_csv: Path) -> Dict[str, str]:
    src = REPO_RUST / "examples" / "_parity_queue.rs"
    src.write_text(RUST_DRIVER)
    build = subprocess.run(
        ["cargo", "build", "--jobs", "1", "--release",
         "--features", "sim", "--example", "_parity_queue"],
        cwd=REPO_RUST, capture_output=True, text=True, timeout=600,
    )
    if build.returncode != 0:
        sys.stderr.write(f"Rust build failed:\n{build.stderr[-2000:]}\n")
        sys.exit(2)
    bin_path = REPO_RUST / "target" / "release" / "examples" / "_parity_queue"
    proc = subprocess.run(
        [str(bin_path), str(snap_csv), str(trade_csv)],
        cwd=REPO_RUST, capture_output=True, text=True, timeout=300,
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


def run_python(snapshots_pq: Path, trades_pq: Path) -> Dict[str, str]:
    driver = f'''
import sys
sys.path.insert(0, {str(REPO_PY)!r})
from mmsim.ingest.lob import load_lob, SnapshotEvent, Book
from mmsim.sim.loop import Order
from mmsim.sim.queue import track_queue_position

stream = load_lob({str(snapshots_pq)!r}, {str(trades_pq)!r})
first_snap = next(e for e in stream if isinstance(e, SnapshotEvent))
initial_book = Book(ts_ns=first_snap.ts_ns, bids=first_snap.bids, asks=first_snap.asks)
order = Order(order_id=0, side=1, price=first_snap.bids[0][0],
              size=0.001, placed_at_ns=first_snap.ts_ns)
post = [e for e in stream if e.ts_ns > first_snap.ts_ns]
trace = track_queue_position(order, post, initial_book)

print(f"n_samples={{len(trace.samples)}}")
print(f"final_queue_pos={{trace.final_queue_pos:.12f}}")
print(f"total_fills_ahead={{trace.total_fills_ahead:.12f}}")
print(f"total_cancels_ahead={{trace.total_cancels_ahead:.12f}}")
print(f"frozen={{str(trace.frozen).lower()}}")
print(f"initial_queue_pos={{trace.samples[0].queue_pos:.12f}}")

n = len(trace.samples)
probes = [0, n//4, n//2, 3*n//4, n-1] if n >= 5 else list(range(n))
for k, idx in enumerate(probes):
    s = trace.samples[idx]
    print(f"sample_{{k}}_idx={{idx}}")
    print(f"sample_{{k}}_ts={{s.ts_ns}}")
    print(f"sample_{{k}}_queue_pos={{s.queue_pos:.12f}}")
    print(f"sample_{{k}}_cause={{s.cause}}")
'''
    py = os.environ.get("MMSIM_PY", sys.executable)
    proc = subprocess.run(
        [py, "-c", driver],
        capture_output=True, text=True, timeout=300,
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
        default=REPO_PY / "tests" / "fixtures" / "lob_btcusdt_60min_snapshots.parquet",
    )
    parser.add_argument(
        "--trades", type=Path,
        default=REPO_PY / "tests" / "fixtures" / "lob_btcusdt_60min_trades.parquet",
    )
    args = parser.parse_args()

    if not args.snapshots.exists() or not args.trades.exists():
        sys.stderr.write(
            f"missing fixture(s): {args.snapshots} / {args.trades}\n")
        return 2

    with tempfile.TemporaryDirectory(prefix="mmsim_parity_queue_") as tmp:
        out_dir = Path(tmp)
        snap_csv, trade_csv = parquet_to_csv(args.snapshots, args.trades, out_dir)
        rs = run_rust(snap_csv, trade_csv)
        py = run_python(args.snapshots, args.trades)

    # Aggregate float keys allowed to drift to 1e-9 (sum-order f64 noise);
    # per-sample queue_pos must be EXACT (no aggregation involved).
    AGGREGATE_FLOAT_KEYS = {
        "total_fills_ahead", "total_cancels_ahead", "final_queue_pos",
    }
    AGGREGATE_TOL = 1e-9

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
        except ValueError:
            print(f"  {k}: py={py[k]} rs={rs[k]} string-mismatch [FAIL]")
            failures.append(k)
            continue
        if k in AGGREGATE_FLOAT_KEYS and rel <= AGGREGATE_TOL:
            print(f"  {k}: py={py[k]} rs={rs[k]} rel={rel:.2e} [OK aggregate]")
            continue
        print(f"  {k}: py={py[k]} rs={rs[k]} rel={rel:.2e} [FAIL]")
        failures.append(k)

    if failures:
        print(f"\nQUEUE PARITY FAILED: {len(failures)} mismatch(es)")
        return 1
    print("\nQUEUE PARITY OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
