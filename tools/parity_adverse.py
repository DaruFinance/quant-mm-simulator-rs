#!/usr/bin/env python3
"""Cross-language parity for adverse-selection filters.

Drives all 6 filter families through identical sequences in Python
and Rust, then runs OFI on full DS-LOB-1H comparing activation counts.

Single-threaded.
"""
from __future__ import annotations

import argparse, os, subprocess, sys, tempfile
from pathlib import Path
from typing import Dict, List

REPO_RUST = Path(__file__).resolve().parent.parent
REPO_PY = Path(os.environ.get("MMSIM_PY_DIR", REPO_RUST.parent / "quant-mm-simulator"))


RUST_DRIVER = r'''
//! Cross-language parity harness binary.

#![cfg(feature = "quoter")]

use quant_mm_simulator_rs::ingest::{load_lob, Book, Event, TradeEvent};
use quant_mm_simulator_rs::quoter::adverse::{
    AdverseFilter, HybridAdverseFilter, HybridAdverseMode,
    MicropriceDevFilter, OFIFilter, QueueImbalanceFilter,
    TradeToxicityFilter, VolSurgeFilter,
};

fn book(bids: Vec<(f64, f64)>, asks: Vec<(f64, f64)>) -> Book {
    Book { ts_ns: 0, bids, asks }
}
fn trade(ts: i64, px: f64, sz: f64, side: i32) -> TradeEvent {
    TradeEvent { ts_ns: ts, recv_ns: ts, symbol: "X".into(), venue: "v".into(),
                 price: px, size: sz, side }
}

fn main() {
    // Case 1: OFI
    let mut f = OFIFilter::new(1000, 0.5);
    f.observe_trade(&trade(0, 100.0, 5.0, 1));
    f.observe_trade(&trade(100, 100.0, 1.0, -1));
    println!("ofi_200={}", f.is_adverse(200) as u8);
    let mut f = OFIFilter::new(100, 0.5);
    f.observe_trade(&trade(0, 100.0, 5.0, 1));
    println!("ofi_evicted={}", f.is_adverse(200) as u8);

    // Case 2: TradeToxicity
    let mut f = TradeToxicityFilter::new(1000, 0.8);
    for i in 0..10 { f.observe_trade(&trade(i, 100.0, 1.0, 1)); }
    f.observe_trade(&trade(20, 100.0, 0.5, -1));
    println!("toxicity_30={}", f.is_adverse(30) as u8);

    // Case 3: VolSurge
    let mut f = VolSurgeFilter::new(10_000, 1.0);
    for (i, px) in [100.0, 110.0, 95.0, 105.0, 90.0].iter().enumerate() {
        f.observe_trade(&trade((i * 100) as i64, *px, 1.0, 1));
    }
    println!("volsurge_loud={}", f.is_adverse(1000) as u8);

    // Case 4: MicropriceDev
    let mut f = MicropriceDevFilter::new(10.0);
    f.observe_book(&book(vec![(100.0, 9.0)], vec![(101.0, 1.0)]));
    println!("microprice_dev_imb={}", f.is_adverse(0) as u8);

    // Case 5: QueueImbalance
    let mut f = QueueImbalanceFilter::new(0.5);
    f.observe_book(&book(vec![(100.0, 9.0)], vec![(101.0, 1.0)]));
    println!("queue_imb_extreme={}", f.is_adverse(0) as u8);

    // Case 6: Hybrid any
    let mut f = HybridAdverseFilter::new(
        vec![
            Box::new(QueueImbalanceFilter::new(0.5)) as Box<dyn AdverseFilter>,
            Box::new(MicropriceDevFilter::new(10.0)) as Box<dyn AdverseFilter>,
        ],
        HybridAdverseMode::Any,
    );
    f.observe_book(&book(vec![(100.0, 9.0)], vec![(101.0, 1.0)]));
    println!("hybrid_any_imb={}", f.is_adverse(0) as u8);

    // DS-LOB-1H OFI activations
    let snap = std::env::args().nth(1).expect("snapshots csv");
    let trade_csv = std::env::args().nth(2).expect("trades csv");
    let stream = load_lob(&snap, &trade_csv, None, None).expect("load_lob");
    let mut f = OFIFilter::new(1_000_000_000, 0.95);
    let mut acts = 0usize;
    let mut last = false;
    for ev in &stream {
        if let Event::Trade(t) = ev {
            f.observe_trade(t);
            let cur = f.is_adverse(t.ts_ns);
            if cur && !last { acts += 1; }
            last = cur;
        }
    }
    println!("ds_lob_ofi_activations={}", acts);
}
'''


def parquet_to_csv(snapshots_pq, trades_pq, out_dir):
    import pyarrow.parquet as pq
    snaps = pq.read_table(snapshots_pq).to_pylist()
    trades = pq.read_table(trades_pq).to_pylist()
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
            row = [str(s["ts_ns"]), str(s["recv_ns"]), s["symbol"], s["venue"], str(s["depth"])]
            bids = list(s["bids"]) + [{"px":0.0, "sz":0.0}] * (depth - len(s["bids"]))
            asks = list(s["asks"]) + [{"px":0.0, "sz":0.0}] * (depth - len(s["asks"]))
            row += [f"{b['px']:.12f}" for b in bids]
            row += [f"{b['sz']:.12f}" for b in bids]
            row += [f"{a['px']:.12f}" for a in asks]
            row += [f"{a['sz']:.12f}" for a in asks]
            fh.write(",".join(row) + "\n")
    trade_csv = out_dir / "trades.csv"
    with trade_csv.open("w") as fh:
        fh.write("ts_ns,recv_ns,symbol,venue,price,size,side\n")
        for t in trades:
            fh.write(",".join([str(t["ts_ns"]), str(t["recv_ns"]),
                                t["symbol"], t["venue"],
                                f"{t['price']:.12f}", f"{t['size']:.12f}",
                                str(t["side"])]) + "\n")
    return snap_csv, trade_csv


def run_rust(snap_csv, trade_csv):
    src = REPO_RUST / "examples" / "_parity_adverse.rs"
    src.write_text(RUST_DRIVER)
    build = subprocess.run(
        ["cargo", "build", "--jobs", "1", "--release",
         "--features", "quoter", "--example", "_parity_adverse"],
        cwd=REPO_RUST, capture_output=True, text=True, timeout=600)
    if build.returncode != 0:
        sys.stderr.write(build.stderr[-2000:]); sys.exit(2)
    bin_path = REPO_RUST / "target" / "release" / "examples" / "_parity_adverse"
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
from mmsim.ingest.lob import load_lob, TradeEvent, Book
from mmsim.quoter.adverse import (
    OFIFilter, TradeToxicityFilter, VolSurgeFilter,
    MicropriceDevFilter, QueueImbalanceFilter, HybridAdverseFilter,
)

def _book(bids, asks):
    return Book(ts_ns=0, bids=tuple(bids), asks=tuple(asks))
def _trade(ts, px, sz, side=1):
    return TradeEvent(ts_ns=ts, recv_ns=ts, symbol="X", venue="v",
                       price=px, size=sz, side=side)

f = OFIFilter(1000, 0.5)
f.observe_trade(_trade(0, 100.0, 5.0, 1))
f.observe_trade(_trade(100, 100.0, 1.0, -1))
print(f"ofi_200={{int(f.is_adverse(200))}}")
f = OFIFilter(100, 0.5)
f.observe_trade(_trade(0, 100.0, 5.0, 1))
print(f"ofi_evicted={{int(f.is_adverse(200))}}")

f = TradeToxicityFilter(1000, 0.8)
for i in range(10): f.observe_trade(_trade(i, 100.0, 1.0, 1))
f.observe_trade(_trade(20, 100.0, 0.5, -1))
print(f"toxicity_30={{int(f.is_adverse(30))}}")

f = VolSurgeFilter(10_000, 1.0)
for i, px in enumerate([100.0, 110.0, 95.0, 105.0, 90.0]):
    f.observe_trade(_trade(i * 100, px, 1.0))
print(f"volsurge_loud={{int(f.is_adverse(1000))}}")

f = MicropriceDevFilter(10.0)
f.observe_book(_book([(100.0, 9.0)], [(101.0, 1.0)]))
print(f"microprice_dev_imb={{int(f.is_adverse(0))}}")

f = QueueImbalanceFilter(0.5)
f.observe_book(_book([(100.0, 9.0)], [(101.0, 1.0)]))
print(f"queue_imb_extreme={{int(f.is_adverse(0))}}")

f = HybridAdverseFilter([QueueImbalanceFilter(0.5), MicropriceDevFilter(10.0)], mode="any")
f.observe_book(_book([(100.0, 9.0)], [(101.0, 1.0)]))
print(f"hybrid_any_imb={{int(f.is_adverse(0))}}")

stream = load_lob({str(snapshots_pq)!r}, {str(trades_pq)!r})
f = OFIFilter(1_000_000_000, 0.95)
acts = 0
last = False
for ev in stream:
    if isinstance(ev, TradeEvent):
        f.observe_trade(ev)
        cur = f.is_adverse(ev.ts_ns)
        if cur and not last:
            acts += 1
        last = cur
print(f"ds_lob_ofi_activations={{acts}}")
'''
    py = os.environ.get("MMSIM_PY", sys.executable)
    proc = subprocess.run([py, "-c", driver], capture_output=True, text=True, timeout=600)
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

    with tempfile.TemporaryDirectory(prefix="mmsim_parity_adv_") as tmp:
        snap_csv, trade_csv = parquet_to_csv(args.snapshots, args.trades, Path(tmp))
        rs = run_rust(snap_csv, trade_csv)
        py = run_python(args.snapshots, args.trades)

    keys = sorted(set(rs) | set(py))
    failures = []
    for k in keys:
        if k not in rs or k not in py:
            print(f"  {k}: MISSING"); failures.append(k); continue
        if py[k] == rs[k]:
            print(f"  {k}: py={py[k]} rs={rs[k]} EXACT"); continue
        print(f"  {k}: py={py[k]} rs={rs[k]} [FAIL]")
        failures.append(k)
    if failures:
        print(f"\nADVERSE PARITY FAILED: {len(failures)} mismatch(es)")
        return 1
    print("\nADVERSE PARITY OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
