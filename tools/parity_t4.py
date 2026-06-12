#!/usr/bin/env python3
"""Cross-language parity for the T4 corpus layer.

Per-primitive comparisons (mirroring the T5 statarb pattern):

| Surface                  | Tolerance / shape |
|--------------------------|-------------------|
| GRID_SIZE                | EXACT             |
| sample_combos count      | EXACT (shape)     |
| iter_all_combos count    | EXACT             |
| inventory-penalty math   | rel 1e-12 fp      |
| quote-shape per-level    | rel 1e-12 fp      |
| refresh-trigger booleans | EXACT bit         |
| reference-price scalars  | rel 1e-12 fp      |
| adverse filter booleans  | EXACT bit         |
| hedge mode net_delta     | rel 1e-12 fp      |
| search_space sample      | shape only (per   |
|                          | T5 PCG-divergence)|
| runner per-fill metrics  | rel 1e-9 (end-to- |
|                          |  end fp drift)    |

Single-threaded. Cargo with --jobs 1.
"""
from __future__ import annotations

import argparse
import json
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
//! Cross-language parity driver for T4 corpus layer.
//! Generated at runtime by tools/parity_t4.py.

#![cfg(feature = "t4-corpus")]

use std::collections::HashMap;

use quant_mm_simulator_rs::ingest::{load_lob, Book, Event};
use quant_mm_simulator_rs::quoter::{
    inv_penalty as ip, shapes,
    triggers::{
        BookEventTrigger, InvChangeTrigger, MidMoveTrigger, RefreshTrigger, TimeTrigger,
    },
    refprice::{microprice as mp_fn, top_mid as tm_fn, weighted_mid as wm_fn},
    adverse::{
        AdverseFilter, MicropriceDevFilter, OFIFilter, QueueImbalanceFilter,
    },
};
use quant_mm_simulator_rs::hedge::engine::HedgeEngine;
use quant_mm_simulator_rs::ingest::TradeEvent;
use quant_mm_simulator_rs::sim::sim_loop::Fill;
use quant_mm_simulator_rs::t4_corpus::{
    combos, search_space::search_space_t4, runner::run_t4_combo,
};
use quant_mm_simulator_rs::t4_corpus::combos::Combo;

fn book(ts: i64, bb: f64, bsz: f64, ba: f64, asz: f64) -> Book {
    Book { ts_ns: ts, bids: vec![(bb, bsz)], asks: vec![(ba, asz)] }
}

fn main() {
    let snap_csv = std::env::args().nth(1).expect("snapshots csv");
    let trade_csv = std::env::args().nth(2).expect("trades csv");

    // ----------------- combos / search_space shape -----------------
    println!("grid_size={}", combos::GRID_SIZE);
    let s10 = combos::sample_combos(10, 2026);
    println!("sample_n10_count={}", s10.len());
    let iter_n = combos::iter_all_combos().count();
    println!("iter_count={}", iter_n);

    let sp = search_space_t4();
    println!("search_space_n_axes={}", sp.axes.len());
    let names: Vec<&str> = sp.axes.iter().map(|a| a.name.as_str()).collect();
    println!("search_space_names={}", names.join(","));
    let samp = sp.sample(16, 2026);
    println!("search_space_sample_count={}", samp.len());
    let first_keys: Vec<&str> = samp[0].keys().map(|k| k.as_str()).collect();
    let mut sk: Vec<&str> = first_keys.clone();
    sk.sort();
    println!("search_space_sample_keys={}", sk.join(","));

    // ----------------- inv_penalty math (1e-12) -----------------
    let s = ip::linear(1.0, 0.5);
    println!("ip_linear_pos={:.15}", s.price_offset);
    let s = ip::linear(-1.0, 0.5);
    println!("ip_linear_neg={:.15}", s.price_offset);
    let s = ip::quadratic(2.0, 0.5);
    println!("ip_quad_2={:.15}", s.price_offset);
    let s = ip::quadratic(-3.0, 0.5);
    println!("ip_quad_n3={:.15}", s.price_offset);
    let s = ip::exponential(0.5, 1.0, 0.2);
    println!("ip_exp={:.15}", s.price_offset);
    let s = ip::asymmetric(1.0, 0.5, 0.25);
    println!("ip_asym_long={:.15}", s.price_offset);
    let s = ip::asymmetric(-1.0, 0.5, 0.25);
    println!("ip_asym_short={:.15}", s.price_offset);
    let s = ip::soft_cap(1.5, 0.5, 1.0);
    println!("ip_softcap={:.15}", s.price_offset);
    let s = ip::hard_cap(1.0, 1.0);
    println!("ip_hardcap_bid_scale={}", s.size_scale_bid);
    println!("ip_hardcap_ask_scale={}", s.size_scale_ask);

    // ----------------- shapes math (1e-12) -----------------
    let single_out = shapes::single(
        &shapes::SingleSpec { size: 0.001, half_spread: 1.0 },
        100.0, 0.0);
    println!("shape_single_n={}", single_out.len());
    println!("shape_single_bid={:.15}", single_out[0].price);
    println!("shape_single_ask={:.15}", single_out[1].price);
    let ladder_out = shapes::ladder(
        &shapes::LadderSpec { half_spread: 1.0, step: 2.0, n_levels: 3,
                               size_per_level: 0.1 },
        100.0, 0.0);
    println!("shape_ladder_n={}", ladder_out.len());
    for (i, q) in ladder_out.iter().enumerate() {
        println!("shape_ladder_{}_px={:.15}", i, q.price);
        println!("shape_ladder_{}_sz={:.15}", i, q.size);
    }
    let geo_out = shapes::geometric(
        &shapes::GeometricSpec { half_spread: 1.0, ratio: 2.0, n_levels: 3,
                                  size_per_level: 0.1 },
        100.0, 0.0);
    println!("shape_geo_n={}", geo_out.len());
    for (i, q) in geo_out.iter().enumerate() {
        println!("shape_geo_{}_px={:.15}", i, q.price);
    }

    // ----------------- refresh triggers (boolean) -----------------
    let mut t = TimeTrigger::new(1_000_000_000);
    println!("trig_time_0={}", t.step(None, 0.0, 100));
    println!("trig_time_1={}", t.step(None, 0.0, 100));
    println!("trig_time_2={}", t.step(None, 0.0, 1_100_000_100));
    let mut tm = MidMoveTrigger::new(10.0);
    let b1 = book(100, 100.0, 1.0, 101.0, 1.0);
    println!("trig_mm_0={}", tm.step(Some(&b1), 0.0, 100));
    let b2 = book(200, 100.05, 1.0, 101.05, 1.0);
    println!("trig_mm_1={}", tm.step(Some(&b2), 0.0, 200));
    let b3 = book(300, 110.0, 1.0, 110.5, 1.0);
    println!("trig_mm_2={}", tm.step(Some(&b3), 0.0, 300));
    let mut iv = InvChangeTrigger::new(0.05);
    println!("trig_inv_0={}", iv.step(None, 0.0, 100));
    println!("trig_inv_1={}", iv.step(None, 0.02, 200));
    println!("trig_inv_2={}", iv.step(None, 0.10, 300));
    let mut be = BookEventTrigger::new();
    println!("trig_book_0={}", be.step(None, 0.0, 100));
    println!("trig_book_1={}", be.step(None, 0.0, 200));

    // ----------------- reference prices (1e-12) -----------------
    let b = book(100, 100.0, 1.0, 101.0, 1.0);
    println!("ref_mid={:.15}", tm_fn(Some(&b)).unwrap());
    let bb = book(100, 100.0, 10.0, 101.0, 1.0);
    println!("ref_micro_imb={:.15}", mp_fn(Some(&bb)).unwrap());
    println!("ref_wmid_imb={:.15}", wm_fn(Some(&bb)).unwrap());

    // ----------------- adverse filters (boolean / 1e-12) -----------------
    let mut q = QueueImbalanceFilter::new(0.5);
    let bb2 = book(100, 100.0, 9.0, 101.0, 1.0);
    q.observe_book(&bb2);
    println!("adv_qimb_heavy={}", q.is_adverse(100));
    let bb3 = book(200, 100.0, 5.0, 101.0, 5.0);
    q.observe_book(&bb3);
    println!("adv_qimb_even={}", q.is_adverse(200));
    let mut mpd = MicropriceDevFilter::new(0.0);
    mpd.observe_book(&bb2);
    println!("adv_mpdev={}", mpd.is_adverse(100));

    let mut ofi = OFIFilter::new(5_000_000_000, 0.5);
    ofi.observe_trade(&TradeEvent {
        ts_ns: 1_000_000, recv_ns: 1_000_000,
        symbol: "X".into(), venue: "v".into(),
        price: 100.0, size: 10.0, side: 1,
    });
    println!("adv_ofi={}", ofi.is_adverse(1_000_000));

    // ----------------- hedge engine (1e-12) -----------------
    let mut he = HedgeEngine::new(0.01, 1.0, "perp");
    he.observe_fill(&Fill {
        fill_id: 0, order_id: 0, ts_ns: 100, price: 100.0,
        size: 1.0, side: 1, is_maker: true,
    });
    let hb = Book { ts_ns: 100,
                     bids: vec![(99.9, 10.0)], asks: vec![(100.1, 10.0)] };
    let _ = he.make_hedge(Some(&hb), 100);
    println!("hedge_net_delta={:.15}", he.net_delta());

    // ----------------- runner end-to-end (1e-9) -----------------
    let stream = load_lob(&snap_csv, &trade_csv, None, None).expect("load_lob");
    let n_events = stream.len();
    let n_snap = stream.iter().filter(|e| matches!(e, Event::Snapshot(_))).count();
    let n_trade = stream.iter().filter(|e| matches!(e, Event::Trade(_))).count();
    println!("stream_n={}", n_events);
    println!("stream_snap={}", n_snap);
    println!("stream_trade={}", n_trade);

    // 4 representative combos covering different axis values.
    let cases: Vec<(&str, Combo)> = vec![
        ("sym_lin_none_none_mid_sgl_book", Combo {
            quoting_model: "symmetric", inventory_penalty: "linear",
            adverse_filter: "none", hedge_mode: "none",
            reference_price: "mid", quote_shape: "single",
            refresh_trigger: "book_event",
        }),
        ("AS_quad_none_perp_mid_sgl_time", Combo {
            quoting_model: "avellaneda_stoikov", inventory_penalty: "quadratic",
            adverse_filter: "none", hedge_mode: "perp",
            reference_price: "mid", quote_shape: "single",
            refresh_trigger: "time",
        }),
        ("ladder_lin_qimb_none_micro_lad_book", Combo {
            quoting_model: "ladder", inventory_penalty: "linear",
            adverse_filter: "queue_imb", hedge_mode: "none",
            reference_price: "microprice", quote_shape: "ladder",
            refresh_trigger: "book_event",
        }),
        ("HS_softcap_none_basket_mid_sgl_book", Combo {
            quoting_model: "ho_stoll", inventory_penalty: "soft_cap",
            adverse_filter: "none", hedge_mode: "basket",
            reference_price: "mid", quote_shape: "single",
            refresh_trigger: "book_event",
        }),
    ];

    for (name, combo) in &cases {
        let params: HashMap<String, f64> = HashMap::new();
        let rr = run_t4_combo(combo.clone(), params, &stream, "BTCUSDT");
        println!("run_{}_n_fills={}", name, rr.n_fills);
        println!("run_{}_n_maker={}", name, rr.n_maker_fills);
        println!("run_{}_n_taker={}", name, rr.n_taker_fills);
        println!("run_{}_total_fees={:.12e}", name, rr.metrics.total_fees);
        println!("run_{}_total_slip={:.12e}", name, rr.metrics.total_slippage);
        println!("run_{}_total_cost={:.12e}", name, rr.metrics.total_cost);
        println!("run_{}_total_notional={:.12e}", name, rr.metrics.total_notional);
    }
}
'''


def parquet_to_csv(snapshots_pq: Path, trades_pq: Path,
                    out_dir: Path) -> tuple[Path, Path]:
    """Re-export the parquet fixtures as wide-format CSV for the Rust
    loader (mirrors parity_lob.py / parity_models.py)."""
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
    src = REPO_RUST / "examples" / "_parity_t4.rs"
    src.write_text(RUST_DRIVER)
    build = subprocess.run(
        ["cargo", "build", "--jobs", "1", "--release",
         "--features", "t4-corpus", "--example", "_parity_t4"],
        cwd=REPO_RUST, capture_output=True, text=True, timeout=900,
    )
    if build.returncode != 0:
        sys.stderr.write(f"Rust build failed:\n{build.stderr[-2000:]}\n")
        sys.exit(2)
    bin_path = REPO_RUST / "target" / "release" / "examples" / "_parity_t4"
    proc = subprocess.run(
        [str(bin_path), str(snap_csv), str(trade_csv)],
        cwd=REPO_RUST, capture_output=True, text=True, timeout=600,
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
    """Python driver: same per-primitive probes + 4 end-to-end combos."""
    driver = f'''
import sys
sys.path.insert(0, {str(REPO_PY)!r})
from mmsim.ingest.lob import load_lob, Book, TradeEvent, SnapshotEvent
from mmsim.sim.loop import Fill
from mmsim.hedge.engine import HedgeEngine
from mmsim.quoter import inv_penalty as ip
from mmsim.quoter import shapes
from mmsim.quoter.triggers import (
    TimeTrigger, MidMoveTrigger, InvChangeTrigger, BookEventTrigger,
)
from mmsim.quoter.refprice import top_mid, weighted_mid, microprice
from mmsim.quoter.adverse import OFIFilter, QueueImbalanceFilter, MicropriceDevFilter

from mmsim.t4_corpus import (
    combos as t4_combos, search_space as t4_ss,
    run_t4_combo,
)
from mmsim.t4_corpus.combos import Combo

# ---------- combos / search_space shape ----------
print(f"grid_size={{t4_combos.GRID_SIZE}}")
s10 = t4_combos.sample_combos(10, 2026)
print(f"sample_n10_count={{len(s10)}}")
iter_n = sum(1 for _ in t4_combos.iter_all_combos())
print(f"iter_count={{iter_n}}")

sp = t4_ss.SEARCH_SPACE_T4
print(f"search_space_n_axes={{len(sp.axes)}}")
print(f"search_space_names={{','.join(sp.names)}}")
samp = sp.sample(16, 2026)
print(f"search_space_sample_count={{len(samp)}}")
keys = sorted(samp[0].keys())
print(f"search_space_sample_keys={{','.join(keys)}}")

# ---------- inv_penalty math ----------
s = ip.linear(1.0, 0.5)
print(f"ip_linear_pos={{s.price_offset:.15g}}")
s = ip.linear(-1.0, 0.5)
print(f"ip_linear_neg={{s.price_offset:.15g}}")
s = ip.quadratic(2.0, 0.5)
print(f"ip_quad_2={{s.price_offset:.15g}}")
s = ip.quadratic(-3.0, 0.5)
print(f"ip_quad_n3={{s.price_offset:.15g}}")
s = ip.exponential(0.5, 1.0, 0.2)
print(f"ip_exp={{s.price_offset:.15g}}")
s = ip.asymmetric(1.0, 0.5, 0.25)
print(f"ip_asym_long={{s.price_offset:.15g}}")
s = ip.asymmetric(-1.0, 0.5, 0.25)
print(f"ip_asym_short={{s.price_offset:.15g}}")
s = ip.soft_cap(1.5, 0.5, 1.0)
print(f"ip_softcap={{s.price_offset:.15g}}")
s = ip.hard_cap(1.0, 1.0)
print(f"ip_hardcap_bid_scale={{s.size_scale_bid}}")
print(f"ip_hardcap_ask_scale={{s.size_scale_ask}}")

# ---------- shapes math ----------
single_out = shapes.single(shapes.SingleSpec(size=0.001, half_spread=1.0), 100.0)
print(f"shape_single_n={{len(single_out)}}")
print(f"shape_single_bid={{single_out[0].price:.15g}}")
print(f"shape_single_ask={{single_out[1].price:.15g}}")
ladder_out = shapes.ladder(shapes.LadderSpec(half_spread=1.0, step=2.0, n_levels=3, size_per_level=0.1), 100.0)
print(f"shape_ladder_n={{len(ladder_out)}}")
for i, q in enumerate(ladder_out):
    print(f"shape_ladder_{{i}}_px={{q.price:.15g}}")
    print(f"shape_ladder_{{i}}_sz={{q.size:.15g}}")
geo_out = shapes.geometric(shapes.GeometricSpec(half_spread=1.0, ratio=2.0, n_levels=3, size_per_level=0.1), 100.0)
print(f"shape_geo_n={{len(geo_out)}}")
for i, q in enumerate(geo_out):
    print(f"shape_geo_{{i}}_px={{q.price:.15g}}")

# ---------- refresh triggers (boolean) ----------
def bs(b):
    return "true" if b else "false"
t = TimeTrigger(interval_ns=1_000_000_000)
print(f"trig_time_0={{bs(t.step(None, 0.0, 100))}}")
print(f"trig_time_1={{bs(t.step(None, 0.0, 100))}}")
print(f"trig_time_2={{bs(t.step(None, 0.0, 1_100_000_100))}}")
def book(ts, bb, bsz, ba, asz):
    return Book(ts_ns=ts, bids=((bb, bsz),), asks=((ba, asz),))
tm = MidMoveTrigger(threshold_bp=10.0)
b1 = book(100, 100.0, 1.0, 101.0, 1.0)
print(f"trig_mm_0={{bs(tm.step(b1, 0.0, 100))}}")
b2 = book(200, 100.05, 1.0, 101.05, 1.0)
print(f"trig_mm_1={{bs(tm.step(b2, 0.0, 200))}}")
b3 = book(300, 110.0, 1.0, 110.5, 1.0)
print(f"trig_mm_2={{bs(tm.step(b3, 0.0, 300))}}")
iv = InvChangeTrigger(threshold=0.05)
print(f"trig_inv_0={{bs(iv.step(None, 0.0, 100))}}")
print(f"trig_inv_1={{bs(iv.step(None, 0.02, 200))}}")
print(f"trig_inv_2={{bs(iv.step(None, 0.10, 300))}}")
be = BookEventTrigger()
print(f"trig_book_0={{bs(be.step(None, 0.0, 100))}}")
print(f"trig_book_1={{bs(be.step(None, 0.0, 200))}}")

# ---------- reference prices ----------
b = book(100, 100.0, 1.0, 101.0, 1.0)
print(f"ref_mid={{top_mid(b):.15g}}")
bb = book(100, 100.0, 10.0, 101.0, 1.0)
print(f"ref_micro_imb={{microprice(bb):.15g}}")
print(f"ref_wmid_imb={{weighted_mid(bb):.15g}}")

# ---------- adverse filters ----------
q = QueueImbalanceFilter(threshold=0.5)
bb2 = book(100, 100.0, 9.0, 101.0, 1.0)
q.observe_book(bb2)
print(f"adv_qimb_heavy={{bs(q.is_adverse(100))}}")
bb3 = book(200, 100.0, 5.0, 101.0, 5.0)
q.observe_book(bb3)
print(f"adv_qimb_even={{bs(q.is_adverse(200))}}")
mpd = MicropriceDevFilter(threshold_bp=0.0)
mpd.observe_book(bb2)
print(f"adv_mpdev={{bs(mpd.is_adverse(100))}}")
ofi = OFIFilter(window_ns=5_000_000_000, threshold=0.5)
ofi.observe_trade(TradeEvent(ts_ns=1_000_000, recv_ns=1_000_000,
                              symbol="X", venue="v",
                              price=100.0, size=10.0, side=1))
print(f"adv_ofi={{bs(ofi.is_adverse(1_000_000))}}")

# ---------- hedge ----------
he = HedgeEngine(threshold=0.01, hedge_size_pct=1.0, instrument="perp")
he.observe_fill(Fill(fill_id=0, order_id=0, ts_ns=100, price=100.0,
                       size=1.0, side=1, is_maker=True))
hb = Book(ts_ns=100, bids=((99.9, 10.0),), asks=((100.1, 10.0),))
_ = he.make_hedge(hb, t_ns=100)
print(f"hedge_net_delta={{he.net_delta:.15g}}")

# ---------- runner end-to-end ----------
stream = load_lob({str(snapshots_pq)!r}, {str(trades_pq)!r})
print(f"stream_n={{len(stream)}}")
print(f"stream_snap={{sum(1 for e in stream if isinstance(e, SnapshotEvent))}}")
print(f"stream_trade={{sum(1 for e in stream if isinstance(e, TradeEvent))}}")

cases = [
    ("sym_lin_none_none_mid_sgl_book", Combo(
        quoting_model="symmetric", inventory_penalty="linear",
        adverse_filter="none", hedge_mode="none",
        reference_price="mid", quote_shape="single",
        refresh_trigger="book_event",
    )),
    ("AS_quad_none_perp_mid_sgl_time", Combo(
        quoting_model="avellaneda_stoikov", inventory_penalty="quadratic",
        adverse_filter="none", hedge_mode="perp",
        reference_price="mid", quote_shape="single",
        refresh_trigger="time",
    )),
    ("ladder_lin_qimb_none_micro_lad_book", Combo(
        quoting_model="ladder", inventory_penalty="linear",
        adverse_filter="queue_imb", hedge_mode="none",
        reference_price="microprice", quote_shape="ladder",
        refresh_trigger="book_event",
    )),
    ("HS_softcap_none_basket_mid_sgl_book", Combo(
        quoting_model="ho_stoll", inventory_penalty="soft_cap",
        adverse_filter="none", hedge_mode="basket",
        reference_price="mid", quote_shape="single",
        refresh_trigger="book_event",
    )),
]
for name, combo in cases:
    rr = run_t4_combo(combo, {{}}, stream, asset="BTCUSDT")
    print(f"run_{{name}}_n_fills={{rr.n_fills}}")
    print(f"run_{{name}}_n_maker={{rr.n_maker_fills}}")
    print(f"run_{{name}}_n_taker={{rr.n_taker_fills}}")
    print(f"run_{{name}}_total_fees={{rr.metrics['total_fees']:.12e}}")
    print(f"run_{{name}}_total_slip={{rr.metrics['total_slippage']:.12e}}")
    print(f"run_{{name}}_total_cost={{rr.metrics['total_cost']:.12e}}")
    print(f"run_{{name}}_total_notional={{rr.metrics['total_notional']:.12e}}")
'''
    py = os.environ.get("MMSIM_PY", sys.executable)
    proc = subprocess.run(
        [py, "-c", driver],
        capture_output=True, text=True, timeout=900,
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


# ---- comparison rules ----

INTEGER_KEYS = {
    "grid_size", "sample_n10_count", "iter_count",
    "search_space_n_axes", "search_space_sample_count",
    "stream_n", "stream_snap", "stream_trade",
    "shape_single_n", "shape_ladder_n", "shape_geo_n",
}
STRING_KEYS = {
    "search_space_names", "search_space_sample_keys",
}
# Booleans (must be exact "true"/"false")
BOOL_KEYS_PREFIXES = ("trig_", "adv_")

# Per-fill end-to-end metrics get a looser tolerance (1e-9) to absorb
# f64 sum-order noise.
LOOSE_KEYS_PREFIXES = ("run_",)


def is_loose_key(k: str) -> bool:
    return any(k.startswith(p) for p in LOOSE_KEYS_PREFIXES)


def is_bool_key(k: str) -> bool:
    return any(k.startswith(p) for p in BOOL_KEYS_PREFIXES)


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

    with tempfile.TemporaryDirectory(prefix="mmsim_parity_t4_") as tmp:
        out_dir = Path(tmp)
        snap_csv, trade_csv = parquet_to_csv(args.snapshots, args.trades, out_dir)
        rs = run_rust(snap_csv, trade_csv)
        py = run_python(args.snapshots, args.trades)

    # Skip integer-counts that have known shape-only parity (search_space sample
    # numeric values diverge by PCG between languages — but counts/keys match).
    SHAPE_ONLY_NUMERIC = set()  # all numeric here are math-exact or 1e-9 loose

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
        if k in STRING_KEYS:
            # Strings must match exactly (e.g., axis name set).
            # Allow order-independent comma-joined sets (search_space_names
            # may diverge by HashMap iteration order in Rust).
            if "," in py[k]:
                py_set = set(py[k].split(","))
                rs_set = set(rs[k].split(","))
                if py_set == rs_set:
                    print(f"  {k}: py={py[k]} rs={rs[k]} EXACT(set)")
                    continue
            print(f"  {k}: py={py[k]} rs={rs[k]} string-mismatch [FAIL]")
            failures.append(k)
            continue
        if k in INTEGER_KEYS:
            print(f"  {k}: py={py[k]} rs={rs[k]} int-mismatch [FAIL]")
            failures.append(k)
            continue
        if is_bool_key(k):
            print(f"  {k}: py={py[k]} rs={rs[k]} bool-mismatch [FAIL]")
            failures.append(k)
            continue
        # Numeric: compare with appropriate tolerance.
        try:
            pv, rv = float(py[k]), float(rs[k])
            denom = max(abs(pv), abs(rv), 1e-15)
            rel = abs(pv - rv) / denom
        except ValueError:
            print(f"  {k}: py={py[k]} rs={rs[k]} unparseable [FAIL]")
            failures.append(k)
            continue
        tol = 1e-9 if is_loose_key(k) else 1e-12
        if rel <= tol:
            print(f"  {k}: py={py[k]} rs={rs[k]} rel={rel:.2e} [OK]")
            continue
        print(f"  {k}: py={py[k]} rs={rs[k]} rel={rel:.2e} [FAIL]")
        failures.append(k)

    if failures:
        print(f"\nT4 PARITY FAILED: {len(failures)} mismatch(es)")
        return 1
    print("\nALL PARITY GREEN")
    return 0


if __name__ == "__main__":
    sys.exit(main())
