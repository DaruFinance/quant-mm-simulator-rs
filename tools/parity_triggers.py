#!/usr/bin/env python3
"""Cross-language parity for refresh-trigger primitives.

Drives all five triggers through identical hand-picked input
sequences in Python and Rust, then runs the 1bp mid-move trigger
over full DS-LOB-1H in both languages.  Diffs the per-step fire
booleans + the DS-LOB-1H fire count.

Tolerance: integer/boolean keys EXACT (no floating-point arithmetic
at the trigger level — they compare against fixed thresholds).

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
//! Generated at runtime by tools/parity_triggers.py.

#![cfg(feature = "quoter")]

use quant_mm_simulator_rs::ingest::{load_lob, Book, Event, SnapshotEvent};
use quant_mm_simulator_rs::quoter::triggers::{
    BookEventTrigger, HybridMode, HybridTrigger, InvChangeTrigger,
    MidMoveTrigger, RefreshTrigger, TimeTrigger,
};

fn book(ts: i64, bid: f64, ask: f64) -> Book {
    Book { ts_ns: ts, bids: vec![(bid, 1.0)], asks: vec![(ask, 1.0)] }
}

fn dump_seq(label: &str, fires: &[bool]) {
    println!("{}_n={}", label, fires.len());
    for (i, f) in fires.iter().enumerate() {
        println!("{}_{}={}", label, i, *f as u8);
    }
}

fn main() {
    // Case 1 — TimeTrigger
    {
        let mut t = TimeTrigger::new(1000);
        let ts_seq: Vec<i64> = vec![0, 500, 999, 1000, 1500, 2000, 3001];
        let fires: Vec<bool> = ts_seq.iter()
            .map(|&ts| t.step(None, 0.0, ts)).collect();
        dump_seq("time_1000", &fires);
    }

    // Case 2 — MidMoveTrigger (5 bp threshold)
    {
        let mut t = MidMoveTrigger::new(5.0);
        let seq = vec![
            (None, 0.0_f64, 0_i64),
            (Some(book(1, 100.0, 100.1)), 0.0, 1),
            (Some(book(2, 100.0, 100.1)), 0.0, 2),
            (Some(book(3, 100.05, 100.15)), 0.0, 3),
            (Some(book(4, 100.10, 100.20)), 0.0, 4),
        ];
        let fires: Vec<bool> = seq.iter()
            .map(|(b, inv, ts)| t.step(b.as_ref(), *inv, *ts)).collect();
        dump_seq("mid_move_5bp", &fires);
    }

    // Case 3 — InvChangeTrigger
    {
        let mut t = InvChangeTrigger::new(0.5);
        let seq = vec![
            (0.0_f64, 0_i64),
            (0.2, 1),
            (0.6, 2),
            (0.7, 3),
            (1.1, 4),
            (1.1, 5),
        ];
        let fires: Vec<bool> = seq.iter()
            .map(|(inv, ts)| t.step(None, *inv, *ts)).collect();
        dump_seq("inv_change_0.5", &fires);
    }

    // Case 4 — BookEventTrigger
    {
        let mut t = BookEventTrigger::new();
        let fires: Vec<bool> = (0..5).map(|i| t.step(None, 0.0, i)).collect();
        dump_seq("book_event", &fires);
    }

    // Case 5 — HybridTrigger any
    {
        let mut t = HybridTrigger::new(
            vec![
                Box::new(TimeTrigger::new(1000)) as Box<dyn RefreshTrigger>,
                Box::new(BookEventTrigger::new()) as Box<dyn RefreshTrigger>,
            ],
            HybridMode::Any,
        );
        let fires: Vec<bool> = (0..3).map(|i| t.step(None, 0.0, i as i64)).collect();
        dump_seq("hybrid_any", &fires);
    }

    // DS-LOB-1H 1bp mid-move fire count.
    let snap_csv = std::env::args().nth(1).expect("snapshots csv");
    let trade_csv = std::env::args().nth(2).expect("trades csv");
    let stream = load_lob(&snap_csv, &trade_csv, None, None).expect("load_lob");
    let mut trig = MidMoveTrigger::new(1.0);
    let mut fires_n = 0usize;
    for ev in &stream {
        if let Event::Snapshot(s) = ev {
            let b = Book {
                ts_ns: s.ts_ns, bids: s.bids.clone(), asks: s.asks.clone(),
            };
            if trig.step(Some(&b), 0.0, s.ts_ns) {
                fires_n += 1;
            }
        }
    }
    println!("ds_lob_1h_1bp_fires={}", fires_n);
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


def run_rust(snap_csv: Path, trade_csv: Path) -> Dict[str, str]:
    src = REPO_RUST / "examples" / "_parity_triggers.rs"
    src.write_text(RUST_DRIVER)
    build = subprocess.run(
        ["cargo", "build", "--jobs", "1", "--release",
         "--features", "quoter", "--example", "_parity_triggers"],
        cwd=REPO_RUST, capture_output=True, text=True, timeout=600,
    )
    if build.returncode != 0:
        sys.stderr.write(build.stderr[-2000:]); sys.exit(2)
    bin_path = REPO_RUST / "target" / "release" / "examples" / "_parity_triggers"
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
from mmsim.ingest.lob import load_lob, SnapshotEvent, Book
from mmsim.quoter.triggers import (
    TimeTrigger, MidMoveTrigger, InvChangeTrigger, BookEventTrigger, HybridTrigger,
)

def _book(ts, bid, ask):
    return Book(ts_ns=ts, bids=((bid, 1.0),), asks=((ask, 1.0),))

def dump(label, fires):
    print(f"{{label}}_n={{len(fires)}}")
    for i, f in enumerate(fires):
        print(f"{{label}}_{{i}}={{1 if f else 0}}")

# Case 1
t = TimeTrigger(1000)
dump("time_1000", [t.step(None, 0.0, ts) for ts in [0, 500, 999, 1000, 1500, 2000, 3001]])

# Case 2
t = MidMoveTrigger(5.0)
seq = [
    (None, 0.0, 0),
    (_book(1, 100.0, 100.1), 0.0, 1),
    (_book(2, 100.0, 100.1), 0.0, 2),
    (_book(3, 100.05, 100.15), 0.0, 3),
    (_book(4, 100.10, 100.20), 0.0, 4),
]
dump("mid_move_5bp", [t.step(b, inv, ts) for b, inv, ts in seq])

# Case 3
t = InvChangeTrigger(0.5)
seq = [(0.0, 0), (0.2, 1), (0.6, 2), (0.7, 3), (1.1, 4), (1.1, 5)]
dump("inv_change_0.5", [t.step(None, inv, ts) for inv, ts in seq])

# Case 4
t = BookEventTrigger()
dump("book_event", [t.step(None, 0.0, i) for i in range(5)])

# Case 5 — HybridTrigger any
t = HybridTrigger([TimeTrigger(1000), BookEventTrigger()], mode="any")
dump("hybrid_any", [t.step(None, 0.0, i) for i in range(3)])

# DS-LOB-1H 1bp mid-move fire count
stream = load_lob({str(snapshots_pq)!r}, {str(trades_pq)!r})
trig = MidMoveTrigger(threshold_bp=1.0)
fires_n = 0
for ev in stream:
    if isinstance(ev, SnapshotEvent):
        b = Book(ts_ns=ev.ts_ns, bids=ev.bids, asks=ev.asks)
        if trig.step(b, 0.0, ev.ts_ns):
            fires_n += 1
print(f"ds_lob_1h_1bp_fires={{fires_n}}")
'''
    py = os.environ.get("MMSIM_PY", sys.executable)
    proc = subprocess.run([py, "-c", driver],
                            capture_output=True, text=True, timeout=300)
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
        sys.stderr.write("missing fixture(s)\n"); return 2

    with tempfile.TemporaryDirectory(prefix="mmsim_parity_trig_") as tmp:
        out_dir = Path(tmp)
        snap_csv, trade_csv = parquet_to_csv(args.snapshots, args.trades, out_dir)
        rs = run_rust(snap_csv, trade_csv)
        py = run_python(args.snapshots, args.trades)

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
        print(f"\nTRIGGERS PARITY FAILED: {len(failures)} mismatch(es)")
        return 1
    print("\nTRIGGERS PARITY OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
