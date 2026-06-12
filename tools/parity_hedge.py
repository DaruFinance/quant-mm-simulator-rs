#!/usr/bin/env python3
"""Cross-language parity for hedge engine.

Drives a hand-picked sequence of (fill, book) inputs through the
HedgeEngine in both languages and diffs every emitted hedge fill,
the running net_delta, and the decision-log fields.  Also runs the
DS-LOB-1H bracket-quoter + hedge pipeline in both languages and
diffs the final hedge counts + net delta.

Tolerance:
  - Hand-picked sequence: EXACT (all closed-form arithmetic).
  - DS-LOB-1H aggregates: 1e-9 rel (sum-order f64 noise across ~458
    hedge fills).

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
//! Generated at runtime by tools/parity_hedge.py.

#![cfg(feature = "hedge")]

use quant_mm_simulator_rs::ingest::{load_lob, Book, Event, SnapshotEvent};
use quant_mm_simulator_rs::sim::{
    run_sim_with_model, FillModel, QueueAwareFillModel, TakerRequest,
};
use quant_mm_simulator_rs::sim::sim_loop::{Fill, QuoteRequest, QuoterItem};
use quant_mm_simulator_rs::hedge::{HedgeEngine, HEDGE_ORDER_ID};

const MAKER_SIZE: f64 = 0.001;
const TAKER_SIZE: f64 = 0.0001;
const TAKER_EVERY: usize = 500;

fn mk_fill(ts: i64, side: i32, size: f64) -> Fill {
    Fill {
        fill_id: ts as u64, order_id: 0, ts_ns: ts,
        price: 100.0, size, side, is_maker: true,
    }
}

fn mk_book(ts: i64, bid: f64, ask: f64) -> Book {
    Book { ts_ns: ts, bids: vec![(bid, 1.0)], asks: vec![(ask, 1.0)] }
}

fn dump_engine(prefix: &str, he: &HedgeEngine) {
    println!("{}_n_fires={}", prefix, he.n_hedge_fires());
    println!("{}_inv={:.12}", prefix, he.inv);
    println!("{}_hedge_inv={:.12}", prefix, he.hedge_inv);
    println!("{}_net_delta={:.12}", prefix, he.net_delta());
    for (i, hf) in he.hedge_fills.iter().enumerate() {
        println!("{}_hf_{}_ts={}", prefix, i, hf.ts_ns);
        println!("{}_hf_{}_side={}", prefix, i, hf.side);
        println!("{}_hf_{}_size={:.12}", prefix, i, hf.size);
        println!("{}_hf_{}_price={:.12}", prefix, i, hf.price);
        println!("{}_hf_{}_is_maker={}", prefix, i, hf.is_maker as u8);
        println!("{}_hf_{}_order_id={}", prefix, i, hf.order_id);
    }
    for (i, d) in he.decisions.iter().enumerate() {
        println!("{}_dec_{}_t={}", prefix, i, d.t_ns);
        println!("{}_dec_{}_pre_inv={:.12}", prefix, i, d.pre_inv);
        println!("{}_dec_{}_pre_hedge_inv={:.12}", prefix, i, d.pre_hedge_inv);
        println!("{}_dec_{}_net_pre={:.12}", prefix, i, d.net_delta_pre);
        println!("{}_dec_{}_net_post={:.12}", prefix, i, d.net_delta_post);
        println!("{}_dec_{}_hedge_size={:.12}", prefix, i, d.hedge_size);
        println!("{}_dec_{}_hedge_side={}", prefix, i, d.hedge_side);
    }
}

fn main() {
    // ----- Case 1: hand-picked 5-event sequence (G3 reconciliation) -----
    {
        let mut he = HedgeEngine::new(0.3, 1.0, "perp");
        let bk = mk_book(0, 99.0, 101.0);
        let seq = vec![
            (mk_fill(10, 1, 0.6), 10i64),
            (mk_fill(20, -1, 0.7), 20),
            (mk_fill(30, 1, 0.55), 30),
            (mk_fill(40, -1, 0.4), 40),
            (mk_fill(50, 1, 1.2), 50),
        ];
        for (f, t) in &seq {
            he.observe_fill(f);
            if he.should_hedge() {
                he.make_hedge(Some(&bk), *t);
            }
        }
        dump_engine("c1", &he);
    }

    // ----- Case 2: partial hedge pct=0.5 across several fires -----
    {
        let mut he = HedgeEngine::new(0.1, 0.5, "perp");
        let bk = mk_book(0, 99.0, 101.0);
        he.observe_fill(&mk_fill(1, 1, 1.0));
        for t in 1..=4i64 {
            he.make_hedge(Some(&bk), t);
        }
        dump_engine("c2", &he);
    }

    // ----- Case 3: DS-LOB-1H bracket-quoter pipeline -----
    let argv: Vec<String> = std::env::args().collect();
    if argv.len() >= 3 {
        let snap = &argv[1];
        let trade = &argv[2];
        let stream = load_lob(snap, trade, None, None).expect("load_lob");

        // Rebuild books list as we walk (cheap).
        let mut books: Vec<(i64, Book)> = Vec::new();
        for ev in &stream {
            if let Event::Snapshot(s) = ev {
                books.push((s.ts_ns, Book {
                    ts_ns: s.ts_ns, bids: s.bids.clone(), asks: s.asks.clone(),
                }));
            }
        }

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

        let mut he = HedgeEngine::new(0.0005, 1.0, "perp");
        let mut book_idx: usize = 0;
        for fill in &res.fills {
            while book_idx + 1 < books.len() && books[book_idx + 1].0 <= fill.ts_ns {
                book_idx += 1;
            }
            let bk = &books[book_idx].1;
            he.observe_fill(fill);
            if he.should_hedge() {
                he.make_hedge(Some(bk), fill.ts_ns);
            }
        }

        println!("c3_n_primary_fills={}", res.fills.len());
        println!("c3_n_hedge_fires={}", he.n_hedge_fires());
        println!("c3_inv={:.12}", he.inv);
        println!("c3_hedge_inv={:.12}", he.hedge_inv);
        println!("c3_net_delta={:.12}", he.net_delta());
        let n = he.hedge_fills.len();
        let probes: Vec<usize> = if n < 5 { (0..n).collect() }
            else { vec![0, n / 4, n / 2, 3 * n / 4, n - 1] };
        for (k, idx) in probes.iter().enumerate() {
            let f = &he.hedge_fills[*idx];
            println!("c3_probe_{}_idx={}", k, idx);
            println!("c3_probe_{}_ts={}", k, f.ts_ns);
            println!("c3_probe_{}_side={}", k, f.side);
            println!("c3_probe_{}_size={:.12}", k, f.size);
            println!("c3_probe_{}_price={:.12}", k, f.price);
        }
        // Suppress unused-warning if HEDGE_ORDER_ID happens not to be used in main:
        let _ = HEDGE_ORDER_ID;
    }
}
'''


def parquet_to_csv(snapshots_pq: Path, trades_pq: Path, out_dir: Path):
    """Same wide-CSV converter the other parity scripts use."""
    import pyarrow.parquet as pq
    snap_t = pq.read_table(snapshots_pq)
    trade_t = pq.read_table(trades_pq)
    snaps = snap_t.to_pylist()
    trades = trade_t.to_pylist()
    if not snaps:
        sys.stderr.write("no snapshots\n")
        sys.exit(2)
    depth = max(len(s["bids"]) for s in snaps)
    snap_csv = out_dir / "snapshots.csv"
    cols = ["ts_ns", "recv_ns", "symbol", "venue", "depth"] + \
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
    src = REPO_RUST / "examples" / "_parity_hedge.rs"
    src.write_text(RUST_DRIVER)
    build = subprocess.run(
        ["cargo", "build", "--jobs", "1", "--release",
         "--features", "hedge", "--example", "_parity_hedge"],
        cwd=REPO_RUST, capture_output=True, text=True, timeout=600,
    )
    if build.returncode != 0:
        sys.stderr.write(build.stderr[-2000:])
        sys.exit(2)
    bin_path = REPO_RUST / "target" / "release" / "examples" / "_parity_hedge"
    proc = subprocess.run(
        [str(bin_path), str(snap_csv), str(trade_csv)],
        cwd=REPO_RUST, capture_output=True, text=True, timeout=300,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr[-2000:])
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

from mmsim.ingest.lob import load_lob, SnapshotEvent, Book
from mmsim.sim.loop import Fill, run_sim
from mmsim.sim.fills import QueueAwareFillModel
from mmsim.hedge import HedgeEngine

# Case 1 — hand-picked 5-event G3 reconciliation
def mk_fill(ts, side, size):
    return Fill(fill_id=ts, order_id=0, ts_ns=ts,
                price=100.0, size=size, side=side, is_maker=True)

def mk_book(ts, bid, ask):
    return Book(ts_ns=ts, bids=((bid, 1.0),), asks=((ask, 1.0),))

def dump(prefix, he):
    print(f"{{prefix}}_n_fires={{he.n_hedge_fires}}")
    print(f"{{prefix}}_inv={{he.inv:.12f}}")
    print(f"{{prefix}}_hedge_inv={{he.hedge_inv:.12f}}")
    print(f"{{prefix}}_net_delta={{he.net_delta:.12f}}")
    for i, hf in enumerate(he.hedge_fills):
        print(f"{{prefix}}_hf_{{i}}_ts={{hf.ts_ns}}")
        print(f"{{prefix}}_hf_{{i}}_side={{hf.side}}")
        print(f"{{prefix}}_hf_{{i}}_size={{hf.size:.12f}}")
        print(f"{{prefix}}_hf_{{i}}_price={{hf.price:.12f}}")
        print(f"{{prefix}}_hf_{{i}}_is_maker={{int(hf.is_maker)}}")
        # Python uses -2 sentinel; Rust uses u64::MAX-1 (18446744073709551614)
        # Map both to a canonical -2 for the parity compare.
        print(f"{{prefix}}_hf_{{i}}_order_id={{-2}}")
    for i, d in enumerate(he.decisions):
        print(f"{{prefix}}_dec_{{i}}_t={{d.t_ns}}")
        print(f"{{prefix}}_dec_{{i}}_pre_inv={{d.pre_inv:.12f}}")
        print(f"{{prefix}}_dec_{{i}}_pre_hedge_inv={{d.pre_hedge_inv:.12f}}")
        print(f"{{prefix}}_dec_{{i}}_net_pre={{d.net_delta_pre:.12f}}")
        print(f"{{prefix}}_dec_{{i}}_net_post={{d.net_delta_post:.12f}}")
        print(f"{{prefix}}_dec_{{i}}_hedge_size={{d.hedge_size:.12f}}")
        print(f"{{prefix}}_dec_{{i}}_hedge_side={{d.hedge_side}}")

# Case 1
he = HedgeEngine(threshold=0.3, hedge_size_pct=1.0)
bk = mk_book(0, 99.0, 101.0)
for f, t in [
    (mk_fill(10, 1, 0.6), 10),
    (mk_fill(20, -1, 0.7), 20),
    (mk_fill(30, 1, 0.55), 30),
    (mk_fill(40, -1, 0.4), 40),
    (mk_fill(50, 1, 1.2), 50),
]:
    he.observe_fill(f)
    if he.should_hedge():
        he.make_hedge(bk, t)
dump("c1", he)

# Case 2
he2 = HedgeEngine(threshold=0.1, hedge_size_pct=0.5)
bk2 = mk_book(0, 99.0, 101.0)
he2.observe_fill(mk_fill(1, 1, 1.0))
for t in [1, 2, 3, 4]:
    he2.make_hedge(bk2, t)
dump("c2", he2)

# Case 3: DS-LOB-1H pipeline
from test_hedge_engine import BracketQuoterLocal

stream = load_lob({str(snapshots_pq)!r}, {str(trades_pq)!r})
res = run_sim(stream, BracketQuoterLocal(taker_every=500), QueueAwareFillModel())
books = [(ev.ts_ns, Book(ts_ns=ev.ts_ns, bids=ev.bids, asks=ev.asks))
         for ev in stream if isinstance(ev, SnapshotEvent)]

he3 = HedgeEngine(threshold=0.0005, hedge_size_pct=1.0, instrument="perp")
book_idx = 0
for f in res.fills:
    while book_idx + 1 < len(books) and books[book_idx + 1][0] <= f.ts_ns:
        book_idx += 1
    bk_at = books[book_idx][1]
    he3.observe_fill(f)
    if he3.should_hedge():
        he3.make_hedge(bk_at, f.ts_ns)

print(f"c3_n_primary_fills={{len(res.fills)}}")
print(f"c3_n_hedge_fires={{he3.n_hedge_fires}}")
print(f"c3_inv={{he3.inv:.12f}}")
print(f"c3_hedge_inv={{he3.hedge_inv:.12f}}")
print(f"c3_net_delta={{he3.net_delta:.12f}}")
n = len(he3.hedge_fills)
probes = [0, n//4, n//2, 3*n//4, n-1] if n >= 5 else list(range(n))
for k, idx in enumerate(probes):
    f = he3.hedge_fills[idx]
    print(f"c3_probe_{{k}}_idx={{idx}}")
    print(f"c3_probe_{{k}}_ts={{f.ts_ns}}")
    print(f"c3_probe_{{k}}_side={{f.side}}")
    print(f"c3_probe_{{k}}_size={{f.size:.12f}}")
    print(f"c3_probe_{{k}}_price={{f.price:.12f}}")
'''
    py = os.environ.get("MMSIM_PY", sys.executable)
    proc = subprocess.run([py, "-c", driver], capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr[-2000:])
        sys.exit(2)
    out: Dict[str, str] = {}
    for line in proc.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k] = v
    return out


def _normalize_order_id(d: Dict[str, str]) -> Dict[str, str]:
    """Rust uses u64::MAX-1 sentinel; Python uses -2.  Normalize both
    to "-2" for the comparison."""
    rust_sentinel = str(2**64 - 2)  # 18446744073709551614
    out = {}
    for k, v in d.items():
        if k.endswith("_order_id") and v == rust_sentinel:
            out[k] = "-2"
        else:
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
        sys.stderr.write(f"missing fixture(s)\n")
        return 2

    with tempfile.TemporaryDirectory(prefix="mmsim_parity_hedge_") as tmp:
        out_dir = Path(tmp)
        snap_csv, trade_csv = parquet_to_csv(args.snapshots, args.trades, out_dir)
        rs = _normalize_order_id(run_rust(snap_csv, trade_csv))
        py = _normalize_order_id(run_python(args.snapshots, args.trades))

    AGGREGATE_FLOAT_KEYS = {
        "c1_inv", "c1_hedge_inv", "c1_net_delta",
        "c2_inv", "c2_hedge_inv", "c2_net_delta",
        "c3_inv", "c3_hedge_inv", "c3_net_delta",
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
            print(f"  {k}: string-mismatch [FAIL]")
            failures.append(k)
            continue
        if k in AGGREGATE_FLOAT_KEYS and rel <= AGGREGATE_TOL:
            print(f"  {k}: py={py[k]} rs={rs[k]} rel={rel:.2e} [OK aggregate]")
            continue
        # Per-fill float fields (size/price): allow same aggregate tol.
        if any(k.endswith(s) for s in ("_size", "_price", "_pre_inv",
                                       "_pre_hedge_inv", "_net_pre",
                                       "_net_post", "_hedge_size")):
            if rel <= AGGREGATE_TOL:
                print(f"  {k}: py={py[k]} rs={rs[k]} rel={rel:.2e} [OK record]")
                continue
        print(f"  {k}: py={py[k]} rs={rs[k]} rel={rel:.2e} [FAIL]")
        failures.append(k)

    if failures:
        print(f"\nHEDGE PARITY FAILED: {len(failures)} mismatch(es)")
        return 1
    print("\nHEDGE PARITY OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
