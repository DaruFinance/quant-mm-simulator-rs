#!/usr/bin/env python3
"""Cross-language parity for maker/taker fill model.

Drives the bracket quoter (TOB makers + every-Nth-snapshot taker
fire) on DS-LOB-1H in both Python and Rust, comparing fill counts,
maker/taker split, total filled qty, and per-fill records at five
evenly-spaced indices.

Single-threaded.

Tolerance:
  - integer / string keys: EXACT
  - per-fill record floats: EXACT (no aggregation)
  - aggregate floats (total_qty): rel ≤ 1e-9 (sum-order f64 noise)
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
//! Generated at runtime by tools/parity_fills.py.

#![cfg(feature = "sim")]

use quant_mm_simulator_rs::ingest::{load_lob, Book};
use quant_mm_simulator_rs::sim::{
    run_sim_with_model, FillModel, QueueAwareFillModel, TakerRequest,
};
use quant_mm_simulator_rs::sim::sim_loop::{QuoteRequest, QuoterItem};

const MAKER_SIZE: f64 = 0.001;
const TAKER_SIZE: f64 = 0.0001;
const TAKER_EVERY: usize = 500;

fn main() {
    let snap = std::env::args().nth(1).expect("snapshots csv");
    let trade = std::env::args().nth(2).expect("trades csv");
    let stream = load_lob(&snap, &trade, None, None).expect("load_lob");

    let mut snap_count: usize = 0;
    let bracket_quoter = move |book: Option<&Book>, _active: &[_], _t_ns: i64| {
        snap_count += 1;
        let mut out: Vec<QuoterItem> = Vec::new();
        if let Some(b) = book {
            if let (Some(bid), Some(ask)) = (b.best_bid(), b.best_ask()) {
                out.push(QuoterItem::Maker(QuoteRequest { side: 1, price: bid, size: MAKER_SIZE }));
                out.push(QuoterItem::Maker(QuoteRequest { side: -1, price: ask, size: MAKER_SIZE }));
                if snap_count % TAKER_EVERY == 0 {
                    out.push(QuoterItem::Taker(TakerRequest {
                        side: 1, size: TAKER_SIZE, limit_px: None,
                    }));
                }
            }
        }
        out
    };

    let mut model = FillModel::QueueAware(QueueAwareFillModel::new());
    let res = run_sim_with_model(&stream, bracket_quoter, &mut model);

    println!("n_events={}", res.n_events_processed);
    println!("n_snapshot_events={}", res.n_snapshot_events);
    println!("n_trade_events={}", res.n_trade_events);
    println!("n_quoter_calls={}", res.n_quoter_calls);
    println!("n_fills={}", res.fills.len());
    println!("n_maker_fills={}", res.n_maker_fills);
    println!("n_taker_fills={}", res.n_taker_fills);
    let total: f64 = res.fills.iter().map(|f| f.size).sum();
    let maker_qty: f64 = res.fills.iter().filter(|f| f.is_maker).map(|f| f.size).sum();
    let taker_qty: f64 = res.fills.iter().filter(|f| !f.is_maker).map(|f| f.size).sum();
    println!("total_filled_qty={:.12}", total);
    println!("maker_qty={:.12}", maker_qty);
    println!("taker_qty={:.12}", taker_qty);

    // 5 evenly-spaced fill probes.
    let n = res.fills.len();
    let probes: Vec<usize> = if n < 5 { (0..n).collect() }
        else { vec![0, n / 4, n / 2, 3 * n / 4, n - 1] };
    for (k, idx) in probes.iter().enumerate() {
        let f = &res.fills[*idx];
        println!("fill_{}_idx={}", k, idx);
        println!("fill_{}_ts={}", k, f.ts_ns);
        println!("fill_{}_price={:.12}", k, f.price);
        println!("fill_{}_size={:.12}", k, f.size);
        println!("fill_{}_side={}", k, f.side);
        println!("fill_{}_is_maker={}", k, f.is_maker);
    }
}
'''


def parquet_to_csv(snapshots_pq: Path, trades_pq: Path, out_dir: Path) -> tuple[Path, Path]:
    """Same wide-CSV converter the other parity scripts use."""
    import pyarrow.parquet as pq
    snap_t = pq.read_table(snapshots_pq)
    trade_t = pq.read_table(trades_pq)
    snaps = snap_t.to_pylist()
    trades = trade_t.to_pylist()
    if not snaps:
        sys.stderr.write("no snapshots\n"); sys.exit(2)
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
    src = REPO_RUST / "examples" / "_parity_fills.rs"
    src.write_text(RUST_DRIVER)
    build = subprocess.run(
        ["cargo", "build", "--jobs", "1", "--release",
         "--features", "sim", "--example", "_parity_fills"],
        cwd=REPO_RUST, capture_output=True, text=True, timeout=600,
    )
    if build.returncode != 0:
        sys.stderr.write(f"Rust build failed:\n{build.stderr[-2000:]}\n")
        sys.exit(2)
    bin_path = REPO_RUST / "target" / "release" / "examples" / "_parity_fills"
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
sys.path.insert(0, {str(REPO_PY / "tests")!r})
from mmsim.ingest.lob import load_lob
from mmsim.sim.loop import run_sim
from mmsim.sim.fills import QueueAwareFillModel
from test_sim_fills import BracketQuoter

stream = load_lob({str(snapshots_pq)!r}, {str(trades_pq)!r})
model = QueueAwareFillModel()
res = run_sim(stream, BracketQuoter(taker_every=500), model)

print(f"n_events={{res.n_events_processed}}")
print(f"n_snapshot_events={{res.n_snapshot_events}}")
print(f"n_trade_events={{res.n_trade_events}}")
print(f"n_quoter_calls={{res.n_quoter_calls}}")
print(f"n_fills={{len(res.fills)}}")
print(f"n_maker_fills={{res.n_maker_fills}}")
print(f"n_taker_fills={{res.n_taker_fills}}")
total = sum(f.size for f in res.fills)
maker = sum(f.size for f in res.fills if f.is_maker)
taker = sum(f.size for f in res.fills if not f.is_maker)
print(f"total_filled_qty={{total:.12f}}")
print(f"maker_qty={{maker:.12f}}")
print(f"taker_qty={{taker:.12f}}")

n = len(res.fills)
probes = [0, n//4, n//2, 3*n//4, n-1] if n >= 5 else list(range(n))
for k, idx in enumerate(probes):
    f = res.fills[idx]
    print(f"fill_{{k}}_idx={{idx}}")
    print(f"fill_{{k}}_ts={{f.ts_ns}}")
    print(f"fill_{{k}}_price={{f.price:.12f}}")
    print(f"fill_{{k}}_size={{f.size:.12f}}")
    print(f"fill_{{k}}_side={{f.side}}")
    print(f"fill_{{k}}_is_maker={{str(f.is_maker).lower()}}")
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
            f"missing fixture(s): {args.snapshots} / {args.trades}\n"); return 2

    with tempfile.TemporaryDirectory(prefix="mmsim_parity_fills_") as tmp:
        out_dir = Path(tmp)
        snap_csv, trade_csv = parquet_to_csv(args.snapshots, args.trades, out_dir)
        rs = run_rust(snap_csv, trade_csv)
        py = run_python(args.snapshots, args.trades)

    AGGREGATE_FLOAT_KEYS = {"total_filled_qty", "maker_qty", "taker_qty"}
    AGGREGATE_TOL = 1e-9

    keys = sorted(set(rs.keys()) | set(py.keys()))
    failures: List[str] = []
    for k in keys:
        if k not in rs or k not in py:
            print(f"  {k}: MISSING (py={k in py}, rs={k in rs})")
            failures.append(k); continue
        if py[k] == rs[k]:
            print(f"  {k}: py={py[k]} rs={rs[k]} EXACT"); continue
        try:
            pv, rv = float(py[k]), float(rs[k])
            denom = max(abs(pv), abs(rv), 1e-15)
            rel = abs(pv - rv) / denom
        except ValueError:
            print(f"  {k}: py={py[k]} rs={rs[k]} string-mismatch [FAIL]")
            failures.append(k); continue
        if k in AGGREGATE_FLOAT_KEYS and rel <= AGGREGATE_TOL:
            print(f"  {k}: py={py[k]} rs={rs[k]} rel={rel:.2e} [OK aggregate]")
            continue
        print(f"  {k}: py={py[k]} rs={rs[k]} rel={rel:.2e} [FAIL]")
        failures.append(k)

    if failures:
        print(f"\nFILLS PARITY FAILED: {len(failures)} mismatch(es)")
        return 1
    print("\nFILLS PARITY OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
