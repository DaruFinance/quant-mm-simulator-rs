#!/usr/bin/env python3
"""Cross-language parity for reference-price primitives.

Runs each of the 6 ref-price primitives on identical hand-picked
inputs in Python and Rust, then computes per-primitive mean+std
over full DS-LOB-1H.

Tolerance: pure book-only primitives EXACT; stateful trackers
EXACT (the math is closed-form floats); aggregates over full
DS-LOB-1H allowed rel 1e-9 (sum-order f64 noise across ~36k floats).

Single-threaded.
"""
from __future__ import annotations

import argparse, os, subprocess, sys, tempfile
from pathlib import Path
from typing import Dict, List

REPO_RUST = Path(__file__).resolve().parent.parent
REPO_PY = Path(os.environ.get(
    "MMSIM_PY_DIR", REPO_RUST.parent / "quant-mm-simulator"))


RUST_DRIVER = r'''
//! Cross-language parity harness binary.
//! Generated at runtime by tools/parity_refprice.py.

#![cfg(feature = "quoter")]

use quant_mm_simulator_rs::ingest::{load_lob, Book, Event, SnapshotEvent, TradeEvent};
use quant_mm_simulator_rs::quoter::refprice::{
    microprice, top_mid, weighted_mid,
    EWMAFairTracker, ModelPredictedTracker, ModelState, VWAPTracker,
    linear_drift_predictor,
};

fn book(bids: Vec<(f64, f64)>, asks: Vec<(f64, f64)>) -> Book {
    Book { ts_ns: 0, bids, asks }
}

fn trade(ts: i64, price: f64, size: f64) -> TradeEvent {
    TradeEvent {
        ts_ns: ts, recv_ns: ts,
        symbol: "X".into(), venue: "v".into(),
        price, size, side: 1,
    }
}

fn opt_f(v: Option<f64>) -> String {
    match v { Some(x) => format!("{:.12}", x), None => "None".into() }
}

fn main() {
    // Case 1 — pure primitives at varied book states
    let b1 = book(vec![(100.0, 5.0)], vec![(101.0, 5.0)]);
    let b2 = book(vec![(100.0, 9.0)], vec![(101.0, 1.0)]);
    let b3 = book(vec![(100.0, 1.0)], vec![(101.0, 9.0)]);
    for (label, b) in [("equal", &b1), ("bid_heavy", &b2), ("ask_heavy", &b3)] {
        println!("top_mid_{}={}", label, opt_f(top_mid(Some(b))));
        println!("weighted_mid_{}={}", label, opt_f(weighted_mid(Some(b))));
        println!("microprice_{}={}", label, opt_f(microprice(Some(b))));
    }
    // None cases
    println!("top_mid_none={}", opt_f(top_mid(None)));
    println!("weighted_mid_none={}", opt_f(weighted_mid(None)));
    println!("microprice_none={}", opt_f(microprice(None)));

    // Case 2 — VWAPTracker sequence
    {
        let mut t = VWAPTracker::new(10_000);
        t.observe(&trade(0, 100.0, 1.0));
        t.observe(&trade(100, 102.0, 3.0));
        t.observe(&trade(5000, 105.0, 2.0));
        println!("vwap_at_200={}", opt_f(t.value(200)));
        println!("vwap_at_6000={}", opt_f(t.value(6000)));
    }

    // Case 3 — EWMAFairTracker
    {
        let mut t = EWMAFairTracker::new(1000);
        t.observe(0, 100.0);
        t.observe(1000, 102.0);
        t.observe(2000, 104.0);
        println!("ewma_at_2000={}", opt_f(t.value(2000)));
    }

    // Case 4 — ModelPredictedTracker (linear_drift_predictor)
    {
        let mut t = ModelPredictedTracker::new(Box::new(linear_drift_predictor));
        t.observe(Some(100.0));
        t.set_slope_per_obs(0.5);
        let v1 = t.value(0);
        t.observe(Some(100.5));
        let v2 = t.value(0);
        println!("model_v1={}", opt_f(v1));
        println!("model_v2={}", opt_f(v2));
    }

    // DS-LOB-1H aggregates: mean of top_mid / weighted_mid / microprice
    let snap = std::env::args().nth(1).expect("snapshots csv");
    let trade_csv = std::env::args().nth(2).expect("trades csv");
    let stream = load_lob(&snap, &trade_csv, None, None).expect("load_lob");
    let mut sum_tm = 0.0; let mut sum_wm = 0.0; let mut sum_mp = 0.0;
    let mut n = 0usize;
    for ev in &stream {
        if let Event::Snapshot(s) = ev {
            let b = Book { ts_ns: s.ts_ns, bids: s.bids.clone(), asks: s.asks.clone() };
            sum_tm += top_mid(Some(&b)).unwrap();
            sum_wm += weighted_mid(Some(&b)).unwrap();
            sum_mp += microprice(Some(&b)).unwrap();
            n += 1;
        }
    }
    println!("ds_lob_n={}", n);
    println!("ds_lob_top_mid_mean={:.12}", sum_tm / n as f64);
    println!("ds_lob_weighted_mid_mean={:.12}", sum_wm / n as f64);
    println!("ds_lob_microprice_mean={:.12}", sum_mp / n as f64);
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
                f"{t['price']:.12f}", f"{t['size']:.12f}", str(t["side"]),
            ]) + "\n")
    return snap_csv, trade_csv


def run_rust(snap_csv, trade_csv):
    src = REPO_RUST / "examples" / "_parity_refprice.rs"
    src.write_text(RUST_DRIVER)
    build = subprocess.run(
        ["cargo", "build", "--jobs", "1", "--release",
         "--features", "quoter", "--example", "_parity_refprice"],
        cwd=REPO_RUST, capture_output=True, text=True, timeout=600,
    )
    if build.returncode != 0:
        sys.stderr.write(build.stderr[-2000:]); sys.exit(2)
    bin_path = REPO_RUST / "target" / "release" / "examples" / "_parity_refprice"
    proc = subprocess.run([str(bin_path), str(snap_csv), str(trade_csv)],
                            cwd=REPO_RUST, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr[-2000:]); sys.exit(2)
    out = {}
    for line in proc.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1); out[k] = v
    return out


def run_python(snapshots_pq, trades_pq):
    driver = f'''
import sys
sys.path.insert(0, {str(REPO_PY)!r})
from mmsim.ingest.lob import load_lob, SnapshotEvent, Book, TradeEvent
from mmsim.quoter.refprice import (
    top_mid, weighted_mid, microprice, VWAPTracker, EWMAFairTracker,
    ModelPredictedTracker, linear_drift_predictor,
)

def _book(bids, asks):
    return Book(ts_ns=0, bids=tuple(bids), asks=tuple(asks))

def _trade(ts, price, size):
    return TradeEvent(ts_ns=ts, recv_ns=ts, symbol="X", venue="v",
                       price=price, size=size, side=1)

def opt(v):
    return "None" if v is None else f"{{v:.12f}}"

b1 = _book([(100.0, 5.0)], [(101.0, 5.0)])
b2 = _book([(100.0, 9.0)], [(101.0, 1.0)])
b3 = _book([(100.0, 1.0)], [(101.0, 9.0)])
for label, b in [("equal", b1), ("bid_heavy", b2), ("ask_heavy", b3)]:
    print(f"top_mid_{{label}}={{opt(top_mid(b))}}")
    print(f"weighted_mid_{{label}}={{opt(weighted_mid(b))}}")
    print(f"microprice_{{label}}={{opt(microprice(b))}}")
print(f"top_mid_none={{opt(top_mid(None))}}")
print(f"weighted_mid_none={{opt(weighted_mid(None))}}")
print(f"microprice_none={{opt(microprice(None))}}")

t = VWAPTracker(window_ns=10_000)
t.observe(_trade(0, 100.0, 1.0))
t.observe(_trade(100, 102.0, 3.0))
t.observe(_trade(5000, 105.0, 2.0))
print(f"vwap_at_200={{opt(t.value(200))}}")
print(f"vwap_at_6000={{opt(t.value(6000))}}")

t = EWMAFairTracker(half_life_ns=1000)
t.observe(0, 100.0)
t.observe(1000, 102.0)
t.observe(2000, 104.0)
print(f"ewma_at_2000={{opt(t.value(2000))}}")

t = ModelPredictedTracker(predict=linear_drift_predictor)
t.observe(mid=100.0, slope_per_obs=0.5)
v1 = t.value(0)
t.observe(mid=100.5, slope_per_obs=0.5)
v2 = t.value(0)
print(f"model_v1={{opt(v1)}}")
print(f"model_v2={{opt(v2)}}")

stream = load_lob({str(snapshots_pq)!r}, {str(trades_pq)!r})
sum_tm = 0.0; sum_wm = 0.0; sum_mp = 0.0; n = 0
for ev in stream:
    if isinstance(ev, SnapshotEvent):
        b = Book(ts_ns=ev.ts_ns, bids=ev.bids, asks=ev.asks)
        sum_tm += top_mid(b)
        sum_wm += weighted_mid(b)
        sum_mp += microprice(b)
        n += 1
print(f"ds_lob_n={{n}}")
print(f"ds_lob_top_mid_mean={{sum_tm / n:.12f}}")
print(f"ds_lob_weighted_mid_mean={{sum_wm / n:.12f}}")
print(f"ds_lob_microprice_mean={{sum_mp / n:.12f}}")
'''
    py = os.environ.get("MMSIM_PY", sys.executable)
    proc = subprocess.run([py, "-c", driver], capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr[-2000:]); sys.exit(2)
    out = {}
    for line in proc.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1); out[k] = v
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshots", type=Path,
        default=REPO_PY / "tests" / "fixtures" / "lob_btcusdt_60min_snapshots.parquet")
    parser.add_argument("--trades", type=Path,
        default=REPO_PY / "tests" / "fixtures" / "lob_btcusdt_60min_trades.parquet")
    args = parser.parse_args()
    if not args.snapshots.exists() or not args.trades.exists():
        sys.stderr.write("missing fixture(s)\n"); return 2

    with tempfile.TemporaryDirectory(prefix="mmsim_parity_rp_") as tmp:
        snap_csv, trade_csv = parquet_to_csv(args.snapshots, args.trades, Path(tmp))
        rs = run_rust(snap_csv, trade_csv)
        py = run_python(args.snapshots, args.trades)

    AGG_FLOAT = {"ds_lob_top_mid_mean", "ds_lob_weighted_mid_mean", "ds_lob_microprice_mean"}
    TOL = 1e-9
    keys = sorted(set(rs) | set(py))
    failures = []
    for k in keys:
        if k not in rs or k not in py:
            print(f"  {k}: MISSING"); failures.append(k); continue
        if py[k] == rs[k]:
            print(f"  {k}: py={py[k]} rs={rs[k]} EXACT"); continue
        try:
            pv, rv = float(py[k]), float(rs[k])
            denom = max(abs(pv), abs(rv), 1e-15)
            rel = abs(pv - rv) / denom
        except ValueError:
            print(f"  {k}: string-mismatch [FAIL]"); failures.append(k); continue
        if k in AGG_FLOAT and rel <= TOL:
            print(f"  {k}: py={py[k]} rs={rs[k]} rel={rel:.2e} [OK aggregate]")
            continue
        print(f"  {k}: py={py[k]} rs={rs[k]} rel={rel:.2e} [FAIL]")
        failures.append(k)

    if failures:
        print(f"\nREFPRICE PARITY FAILED: {len(failures)} mismatch(es)")
        return 1
    print("\nREFPRICE PARITY OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
