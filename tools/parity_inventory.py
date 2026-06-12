#!/usr/bin/env python3
"""Cross-language parity for inventory tracker.

Drives the bracket quoter on DS-LOB-1H in both languages, runs the
inventory tracker over the resulting fills, and diffs:
  - n_fills, final_inv, peak_long, peak_short
  - per-sample (ts, inv, fill_signed_size) at five evenly-spaced
    indices

Tolerance: per-record EXACT (the inventory tracker is a pure
running-sum applied to bit-identical fill ledgers from the fill-model parity);
aggregates allowed 1e-9 (sum-order f64 noise across ~1900 floats).

Single-threaded.
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
//! Generated at runtime by tools/parity_inventory.py.

#![cfg(feature = "sim")]

use quant_mm_simulator_rs::ingest::{load_lob, Book};
use quant_mm_simulator_rs::sim::{
    inventory_path, run_sim_with_model, FillModel, QueueAwareFillModel,
    TakerRequest,
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
    let trace = inventory_path(&res.fills);

    println!("n_fills={}", trace.n_fills);
    println!("final_inv={:.12}", trace.final_inv);
    println!("peak_long={:.12}", trace.peak_long);
    println!("peak_short={:.12}", trace.peak_short);

    let n = trace.samples.len();
    let probes: Vec<usize> = if n < 5 { (0..n).collect() }
        else { vec![0, n / 4, n / 2, 3 * n / 4, n - 1] };
    for (k, idx) in probes.iter().enumerate() {
        let s = &trace.samples[*idx];
        println!("sample_{}_idx={}", k, idx);
        println!("sample_{}_ts={}", k, s.ts_ns);
        println!("sample_{}_inv={:.12}", k, s.inv);
        println!("sample_{}_signed={:.12}", k, s.fill_signed_size);
    }
}
'''


def parquet_to_csv(snapshots_pq: Path, trades_pq: Path, out_dir: Path):
    """Same wide-CSV converter the other parity scripts use."""
    import pyarrow.parquet as pq
    snap_t = pq.read_table(snapshots_pq); trade_t = pq.read_table(trades_pq)
    snaps = snap_t.to_pylist(); trades = trade_t.to_pylist()
    if not snaps:
        sys.stderr.write("no snapshots\n"); sys.exit(2)
    depth = max(len(s["bids"]) for s in snaps)
    snap_csv = out_dir / "snapshots.csv"
    cols = ["ts_ns","recv_ns","symbol","venue","depth"] + \
           [f"bid_px_{k}" for k in range(depth)] + \
           [f"bid_sz_{k}" for k in range(depth)] + \
           [f"ask_px_{k}" for k in range(depth)] + \
           [f"ask_sz_{k}" for k in range(depth)]
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
    cols2 = ["ts_ns","recv_ns","symbol","venue","price","size","side"]
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
    src = REPO_RUST / "examples" / "_parity_inventory.rs"
    src.write_text(RUST_DRIVER)
    build = subprocess.run(
        ["cargo", "build", "--jobs", "1", "--release",
         "--features", "sim", "--example", "_parity_inventory"],
        cwd=REPO_RUST, capture_output=True, text=True, timeout=600,
    )
    if build.returncode != 0:
        sys.stderr.write(build.stderr[-2000:]); sys.exit(2)
    bin_path = REPO_RUST / "target" / "release" / "examples" / "_parity_inventory"
    proc = subprocess.run(
        [str(bin_path), str(snap_csv), str(trade_csv)],
        cwd=REPO_RUST, capture_output=True, text=True, timeout=300,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr[-2000:]); sys.exit(2)
    out: Dict[str, str] = {}
    for line in proc.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1); out[k] = v
    return out


def run_python(snapshots_pq: Path, trades_pq: Path) -> Dict[str, str]:
    driver = f'''
import sys
sys.path.insert(0, {str(REPO_PY)!r})
sys.path.insert(0, {str(REPO_PY / "tests")!r})
from mmsim.ingest.lob import load_lob
from mmsim.sim.loop import run_sim
from mmsim.sim.fills import QueueAwareFillModel
from mmsim.sim.inventory import inventory_path
from test_sim_fills import BracketQuoter

stream = load_lob({str(snapshots_pq)!r}, {str(trades_pq)!r})
res = run_sim(stream, BracketQuoter(taker_every=500), QueueAwareFillModel())
trace = inventory_path(res.fills)
print(f"n_fills={{trace.n_fills}}")
print(f"final_inv={{trace.final_inv:.12f}}")
print(f"peak_long={{trace.peak_long:.12f}}")
print(f"peak_short={{trace.peak_short:.12f}}")
n = len(trace.samples)
probes = [0, n//4, n//2, 3*n//4, n-1] if n >= 5 else list(range(n))
for k, idx in enumerate(probes):
    s = trace.samples[idx]
    print(f"sample_{{k}}_idx={{idx}}")
    print(f"sample_{{k}}_ts={{s.ts_ns}}")
    print(f"sample_{{k}}_inv={{s.inv:.12f}}")
    print(f"sample_{{k}}_signed={{s.fill_signed_size:.12f}}")
'''
    py = os.environ.get("MMSIM_PY", sys.executable)
    proc = subprocess.run([py, "-c", driver], capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr[-2000:]); sys.exit(2)
    out: Dict[str, str] = {}
    for line in proc.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1); out[k] = v
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
        sys.stderr.write(f"missing fixture(s)\n"); return 2

    with tempfile.TemporaryDirectory(prefix="mmsim_parity_inv_") as tmp:
        out_dir = Path(tmp)
        snap_csv, trade_csv = parquet_to_csv(args.snapshots, args.trades, out_dir)
        rs = run_rust(snap_csv, trade_csv)
        py = run_python(args.snapshots, args.trades)

    AGGREGATE_FLOAT_KEYS = {"final_inv", "peak_long", "peak_short"}
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
            print(f"  {k}: string-mismatch [FAIL]"); failures.append(k); continue
        if k in AGGREGATE_FLOAT_KEYS and rel <= AGGREGATE_TOL:
            print(f"  {k}: py={py[k]} rs={rs[k]} rel={rel:.2e} [OK aggregate]")
            continue
        print(f"  {k}: py={py[k]} rs={rs[k]} rel={rel:.2e} [FAIL]")
        failures.append(k)

    if failures:
        print(f"\nINVENTORY PARITY FAILED: {len(failures)} mismatch(es)")
        return 1
    print("\nINVENTORY PARITY OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
