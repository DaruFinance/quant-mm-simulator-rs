#!/usr/bin/env python3
"""Cross-language parity for aux log streams.

Drives the bracket quoter + QueueAware fill model on DS-LOB-1H in
both languages, dumps the three aux log CSVs (fill_rate.csv,
inventory.csv, queue_pos.csv) from each side, and diffs them
row-by-row.

Single-threaded.

Tolerance:
  - integer columns (bucket_s, t_ns_start, n_fills, ts_ns, order_id,
    side, n_fills_so_far, frozen): EXACT
  - float columns (total_qty, inv, price, queue_pos): rel <= 1e-9
    (cross-language f64 summation noise; queue_pos arithmetic in the
    tracker is per-tick exact already)
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

REPO_RUST = Path(__file__).resolve().parent.parent
REPO_PY = Path(os.environ.get(
    "MMSIM_PY_DIR", REPO_RUST.parent / "quant-mm-simulator"))


RUST_DRIVER = r'''
//! Cross-language parity harness binary.
//! Generated at runtime by tools/parity_aux_logs.py.

#![cfg(feature = "logs")]

use std::path::Path;

use quant_mm_simulator_rs::ingest::{load_lob, Book};
use quant_mm_simulator_rs::sim::{
    FillModel, QueueAwareFillModel, TakerRequest,
};
use quant_mm_simulator_rs::sim::sim_loop::{QuoteRequest, QuoterItem};
use quant_mm_simulator_rs::logs::run_sim_with_aux_logs;

const MAKER_SIZE: f64 = 0.001;
const TAKER_SIZE: f64 = 0.0001;
const TAKER_EVERY: usize = 500;

fn main() {
    let snap = std::env::args().nth(1).expect("snapshots csv");
    let trade = std::env::args().nth(2).expect("trades csv");
    let out_dir_s = std::env::args().nth(3).expect("out dir");
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
    let (_res, aux) = run_sim_with_aux_logs(&stream, bracket_quoter, &mut model);
    let out_dir = Path::new(&out_dir_s);
    std::fs::create_dir_all(out_dir).unwrap();
    let (fr, inv, qp) = aux.to_csvs(out_dir, None, None).unwrap();
    println!("fill_rate={}", fr.display());
    println!("inventory={}", inv.display());
    println!("queue_pos={}", qp.display());
}
'''


def parquet_to_csv(snapshots_pq: Path, trades_pq: Path, out_dir: Path):
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


def run_rust(snap_csv: Path, trade_csv: Path, out_dir: Path) -> Dict[str, Path]:
    src = REPO_RUST / "examples" / "_parity_aux_logs.rs"
    src.write_text(RUST_DRIVER)
    build = subprocess.run(
        ["cargo", "build", "--jobs", "1", "--release",
         "--features", "logs", "--example", "_parity_aux_logs"],
        cwd=REPO_RUST, capture_output=True, text=True, timeout=900,
    )
    if build.returncode != 0:
        sys.stderr.write(build.stderr[-2000:]); sys.exit(2)
    bin_path = REPO_RUST / "target" / "release" / "examples" / "_parity_aux_logs"
    proc = subprocess.run(
        [str(bin_path), str(snap_csv), str(trade_csv), str(out_dir)],
        cwd=REPO_RUST, capture_output=True, text=True, timeout=600,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr[-2000:]); sys.exit(2)
    out: Dict[str, Path] = {}
    for line in proc.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k] = Path(v)
    return out


def run_python(snapshots_pq: Path, trades_pq: Path, out_dir: Path) -> Dict[str, Path]:
    driver = f'''
import sys
sys.path.insert(0, {str(REPO_PY)!r})
sys.path.insert(0, {str(REPO_PY / "tests")!r})
from pathlib import Path
from mmsim.ingest.lob import load_lob
from mmsim.sim.fills import QueueAwareFillModel
from mmsim.logs.aux import run_sim_with_aux_logs
from test_sim_fills import BracketQuoter

stream = load_lob({str(snapshots_pq)!r}, {str(trades_pq)!r})
model = QueueAwareFillModel()
_res, aux = run_sim_with_aux_logs(stream, BracketQuoter(taker_every=500), model)
fr, inv, qp = aux.to_csvs(Path({str(out_dir)!r}))
print(f"fill_rate={{fr}}")
print(f"inventory={{inv}}")
print(f"queue_pos={{qp}}")
'''
    py = os.environ.get("MMSIM_PY", sys.executable)
    proc = subprocess.run([py, "-c", driver], capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr[-2000:]); sys.exit(2)
    out: Dict[str, Path] = {}
    for line in proc.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k] = Path(v)
    return out


INT_COLS = {
    "fill_rate": {"bucket_s", "t_ns_start", "n_fills"},
    "inventory": {"bucket_s", "t_ns_start", "n_fills_so_far"},
    "queue_pos": {"ts_ns", "order_id", "side", "frozen"},
}
FLOAT_COLS = {
    "fill_rate": {"total_qty"},
    "inventory": {"inv"},
    "queue_pos": {"price", "queue_pos"},
}

REL_TOL = 1e-9
ABS_TOL = 1e-12


def diff_csv(name: str, py_path: Path, rs_path: Path) -> List[str]:
    failures: List[str] = []
    with py_path.open() as fa, rs_path.open() as fb:
        py_rows = list(csv.DictReader(fa))
        rs_rows = list(csv.DictReader(fb))
    if len(py_rows) != len(rs_rows):
        failures.append(
            f"{name}: row count differs: py={len(py_rows)} rs={len(rs_rows)}")
        return failures
    int_cols = INT_COLS[name]
    flt_cols = FLOAT_COLS[name]
    n_mismatch = 0
    n_total = len(py_rows)
    for i, (pa, pb) in enumerate(zip(py_rows, rs_rows)):
        if set(pa.keys()) != set(pb.keys()):
            failures.append(
                f"{name} row {i}: column set differs py={list(pa.keys())} rs={list(pb.keys())}")
            return failures
        for k in pa:
            if k in int_cols:
                if pa[k] != pb[k]:
                    if n_mismatch < 5:
                        failures.append(
                            f"{name} row {i} col {k}: py={pa[k]!r} rs={pb[k]!r} [INT MISMATCH]")
                    n_mismatch += 1
            elif k in flt_cols:
                pv = float(pa[k]); rv = float(pb[k])
                denom = max(abs(pv), abs(rv), 1e-15)
                rel = abs(pv - rv) / denom
                if rel > REL_TOL and abs(pv - rv) > ABS_TOL:
                    if n_mismatch < 5:
                        failures.append(
                            f"{name} row {i} col {k}: py={pv:.15g} rs={rv:.15g} rel={rel:.2e}")
                    n_mismatch += 1
            else:
                if pa[k] != pb[k]:
                    if n_mismatch < 5:
                        failures.append(
                            f"{name} row {i} col {k}: py={pa[k]!r} rs={pb[k]!r}")
                    n_mismatch += 1
    if n_mismatch > 0:
        failures.append(f"{name}: {n_mismatch}/{n_total * len(int_cols | flt_cols)} cell mismatches")
    else:
        print(f"  {name}: {n_total} rows EXACT / within tol [OK]")
    return failures


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

    with tempfile.TemporaryDirectory(prefix="mmsim_parity_aux_") as tmp:
        out_dir = Path(tmp)
        snap_csv, trade_csv = parquet_to_csv(args.snapshots, args.trades, out_dir)
        py_out = out_dir / "py"
        rs_out = out_dir / "rs"
        py_out.mkdir(); rs_out.mkdir()
        py_paths = run_python(args.snapshots, args.trades, py_out)
        rs_paths = run_rust(snap_csv, trade_csv, rs_out)

        failures: List[str] = []
        for name in ("fill_rate", "inventory", "queue_pos"):
            failures += diff_csv(name, py_paths[name], rs_paths[name])

    if failures:
        print(f"\nAUX_LOGS PARITY FAILED:")
        for f in failures:
            print(f"  {f}")
        return 1
    print("\nAUX_LOGS PARITY OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
