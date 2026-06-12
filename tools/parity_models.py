#!/usr/bin/env python3
"""Cross-language parity for quoting-model library.

Drives all 8 models in Python and Rust on the DS-LOB-1H fixture
and compares the per-model quote stream — both spot-probes at
selected snapshot indices AND aggregate stats over the full stream.

Each model is run with FIXED, parity-friendly params (same on both
sides; documented inline).  Inv is held flat at 0.0 throughout the
drive so the comparison isolates the math from inventory feedback
loops; deterministic.

Tolerance:
  - integer counts: EXACT
  - per-snapshot quote prices: rel 1e-12 (closed-form math)
  - aggregate mean prices over ~35k snapshots: rel 1e-9 (f64 sum
    order noise across the stream)

Single-threaded.  Cargo with --jobs 1.
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


# Models to test, with shared params.
MODELS = [
    "symmetric",
    "ladder",
    "microprice_skew",
    "fair_anchored",
    "avellaneda_stoikov",
    "cartea_jaimungal",
    "glft",
    "ho_stoll",
]


RUST_DRIVER = r'''
//! Cross-language parity harness binary.
//! Generated at runtime by tools/parity_models.py.

#![cfg(feature = "models")]

use quant_mm_simulator_rs::ingest::{load_lob, Book, Event, SnapshotEvent};
use quant_mm_simulator_rs::quoter::{Decision, Quoter};
use quant_mm_simulator_rs::models::{
    AvellanedaStoikovQuoter, CarteaJaimungalQuoter, FairAnchoredQuoter, GLFTQuoter,
    HoStollQuoter, LadderQuoter, MicropriceSkewQuoter, SymmetricQuoter,
};

fn build(name: &str) -> Box<dyn Quoter> {
    match name {
        "symmetric" => Box::new(SymmetricQuoter::new(5.0, 0.001)),
        "ladder" => Box::new(LadderQuoter::new(5.0, 2.5, 3, 0.001)),
        "microprice_skew" => Box::new(MicropriceSkewQuoter::new(5.0, 0.001)),
        "fair_anchored" => Box::new(FairAnchoredQuoter::new(5.0, 0.001, 5_000_000_000)),
        "avellaneda_stoikov" => Box::new(AvellanedaStoikovQuoter::new(
            0.5, 1.5, 3_600_000_000_000, 0.001, 10_000_000_000)),
        "cartea_jaimungal" => Box::new(CarteaJaimungalQuoter::new(
            0.5, 1.5, 0.0, 3_600_000_000_000, 0.001, 10_000_000_000)),
        "glft" => Box::new(GLFTQuoter::new(
            0.5, 1.5, 140.0, 3_600_000_000_000, 0.001, 10_000_000_000)),
        "ho_stoll" => Box::new(HoStollQuoter::new(10.0, 0.05, 0.001, 10_000_000_000)),
        other => panic!("unknown model: {}", other),
    }
}

fn run_model(name: &str, snaps: &[SnapshotEvent]) -> Vec<(i64, Option<(f64, f64)>)> {
    let mut q = build(name);
    let mut out = Vec::with_capacity(snaps.len());
    for s in snaps {
        let book = Book {
            ts_ns: s.ts_ns, bids: s.bids.clone(), asks: s.asks.clone(),
        };
        let decisions = q.quote(Some(&book), 0.0, s.ts_ns);
        // Capture first bid + first ask (or None if model emits empty).
        let mut bid: Option<f64> = None;
        let mut ask: Option<f64> = None;
        for d in &decisions {
            if let Decision::Maker(qr) = d {
                if qr.side == 1 && bid.is_none() {
                    bid = Some(qr.price);
                }
                if qr.side == -1 && ask.is_none() {
                    ask = Some(qr.price);
                }
            }
        }
        match (bid, ask) {
            (Some(b), Some(a)) => out.push((s.ts_ns, Some((b, a)))),
            _ => out.push((s.ts_ns, None)),
        }
    }
    out
}

fn main() {
    let snap_csv = std::env::args().nth(1).expect("snapshots csv");
    let trade_csv = std::env::args().nth(2).expect("trades csv");
    let model_names: Vec<&str> = vec![
        "symmetric", "ladder", "microprice_skew", "fair_anchored",
        "avellaneda_stoikov", "cartea_jaimungal", "glft", "ho_stoll",
    ];

    let stream = load_lob(&snap_csv, &trade_csv, None, None).expect("load_lob");
    // Filter to snapshots only (quoter is called only on snapshots).
    let snaps: Vec<SnapshotEvent> = stream.iter().filter_map(|ev| {
        if let Event::Snapshot(s) = ev { Some(s.clone()) } else { None }
    }).collect();
    println!("n_snaps={}", snaps.len());

    for name in &model_names {
        let series = run_model(name, &snaps);
        let n_emit = series.iter().filter(|(_, q)| q.is_some()).count();
        let n_skip = series.len() - n_emit;
        println!("{}_n_emit={}", name, n_emit);
        println!("{}_n_skip={}", name, n_skip);

        // Aggregate: mean bid + ask px across emitting snapshots.
        let (mut sum_bid, mut sum_ask) = (0.0_f64, 0.0_f64);
        for (_, q) in &series {
            if let Some((b, a)) = q {
                sum_bid += b;
                sum_ask += a;
            }
        }
        if n_emit > 0 {
            println!("{}_mean_bid={:.9}", name, sum_bid / n_emit as f64);
            println!("{}_mean_ask={:.9}", name, sum_ask / n_emit as f64);
        } else {
            println!("{}_mean_bid=None", name);
            println!("{}_mean_ask=None", name);
        }

        // Per-snapshot probes at 5 indices.
        let n = series.len();
        let probes: Vec<usize> = if n < 5 {
            (0..n).collect()
        } else {
            vec![n / 8, n / 4, n / 2, 3 * n / 4, 7 * n / 8]
        };
        for (k, idx) in probes.iter().enumerate() {
            let (ts, qopt) = &series[*idx];
            match qopt {
                Some((b, a)) => {
                    println!("{}_probe_{}_idx={}", name, k, idx);
                    println!("{}_probe_{}_ts={}", name, k, ts);
                    println!("{}_probe_{}_bid={:.12}", name, k, b);
                    println!("{}_probe_{}_ask={:.12}", name, k, a);
                }
                None => {
                    println!("{}_probe_{}_idx={}", name, k, idx);
                    println!("{}_probe_{}_ts={}", name, k, ts);
                    println!("{}_probe_{}_bid=None", name, k);
                    println!("{}_probe_{}_ask=None", name, k);
                }
            }
        }
    }
}
'''


def parquet_to_csv(snapshots_pq: Path, trades_pq: Path,
                     out_dir: Path) -> tuple:
    """Re-export the parquet fixtures as wide-format CSV for the Rust
    loader.  Identical to parity_lob.py's converter."""
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
    src = REPO_RUST / "examples" / "_parity_models.rs"
    src.write_text(RUST_DRIVER)
    build = subprocess.run(
        ["cargo", "build", "--jobs", "1", "--release",
         "--features", "models", "--example", "_parity_models"],
        cwd=REPO_RUST, capture_output=True, text=True, timeout=900,
    )
    if build.returncode != 0:
        sys.stderr.write(f"Rust build failed:\n{build.stderr[-2000:]}\n")
        sys.exit(2)
    bin_path = REPO_RUST / "target" / "release" / "examples" / "_parity_models"
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
    """Python driver: build each model with the same params, walk
    snapshots, dump the same probe + aggregate stats."""
    driver = f'''
import sys
sys.path.insert(0, {str(REPO_PY)!r})
from mmsim.ingest.lob import load_lob, SnapshotEvent
from mmsim.models import (
    AvellanedaStoikovQuoter, CarteaJaimungalQuoter, FairAnchoredQuoter, GLFTQuoter,
    HoStollQuoter, LadderQuoter, MicropriceSkewQuoter, SymmetricQuoter,
)
from mmsim.sim.loop import QuoteRequest

def build(name):
    if name == "symmetric":
        return SymmetricQuoter(half_spread=5.0, size=0.001)
    if name == "ladder":
        return LadderQuoter(half_spread=5.0, step=2.5, n_levels=3, size_per_level=0.001)
    if name == "microprice_skew":
        return MicropriceSkewQuoter(half_spread=5.0, size=0.001)
    if name == "fair_anchored":
        return FairAnchoredQuoter(half_spread=5.0, size=0.001, half_life_ns=5_000_000_000)
    if name == "avellaneda_stoikov":
        return AvellanedaStoikovQuoter(gamma=0.5, k=1.5, horizon_ns=3_600_000_000_000,
                                          size=0.001, vol_window_ns=10_000_000_000)
    if name == "cartea_jaimungal":
        return CarteaJaimungalQuoter(gamma=0.5, k=1.5, kappa=0.0, horizon_ns=3_600_000_000_000,
                                        size=0.001, vol_window_ns=10_000_000_000)
    if name == "glft":
        return GLFTQuoter(gamma=0.5, k=1.5, A=140.0, horizon_ns=3_600_000_000_000,
                            size=0.001, vol_window_ns=10_000_000_000)
    if name == "ho_stoll":
        return HoStollQuoter(alpha=10.0, beta=0.05, size=0.001, vol_window_ns=10_000_000_000)
    raise SystemExit(f"unknown model: {{name}}")


def run_model(name, snaps):
    q = build(name)
    out = []
    for s in snaps:
        from mmsim.ingest.lob import Book
        b = Book(ts_ns=s.ts_ns, bids=s.bids, asks=s.asks)
        decisions = q.quote(b, 0.0, s.ts_ns)
        bid = ask = None
        for d in decisions:
            if isinstance(d, QuoteRequest):
                if d.side == 1 and bid is None:
                    bid = d.price
                if d.side == -1 and ask is None:
                    ask = d.price
        if bid is not None and ask is not None:
            out.append((s.ts_ns, (bid, ask)))
        else:
            out.append((s.ts_ns, None))
    return out

stream = load_lob({str(snapshots_pq)!r}, {str(trades_pq)!r})
snaps = [e for e in stream if isinstance(e, SnapshotEvent)]
print(f"n_snaps={{len(snaps)}}")

for name in {MODELS!r}:
    series = run_model(name, snaps)
    n_emit = sum(1 for _, q in series if q is not None)
    n_skip = len(series) - n_emit
    print(f"{{name}}_n_emit={{n_emit}}")
    print(f"{{name}}_n_skip={{n_skip}}")
    sum_bid = sum(q[0] for _, q in series if q is not None)
    sum_ask = sum(q[1] for _, q in series if q is not None)
    if n_emit > 0:
        print(f"{{name}}_mean_bid={{sum_bid / n_emit:.9f}}")
        print(f"{{name}}_mean_ask={{sum_ask / n_emit:.9f}}")
    else:
        print(f"{{name}}_mean_bid=None")
        print(f"{{name}}_mean_ask=None")
    n = len(series)
    probes = [n // 8, n // 4, n // 2, 3 * n // 4, 7 * n // 8] if n >= 5 else list(range(n))
    for k, idx in enumerate(probes):
        ts, qopt = series[idx]
        print(f"{{name}}_probe_{{k}}_idx={{idx}}")
        print(f"{{name}}_probe_{{k}}_ts={{ts}}")
        if qopt is None:
            print(f"{{name}}_probe_{{k}}_bid=None")
            print(f"{{name}}_probe_{{k}}_ask=None")
        else:
            b, a = qopt
            print(f"{{name}}_probe_{{k}}_bid={{b:.12f}}")
            print(f"{{name}}_probe_{{k}}_ask={{a:.12f}}")
'''
    py = os.environ.get("MMSIM_PY", sys.executable)
    proc = subprocess.run(
        [py, "-c", driver],
        capture_output=True, text=True, timeout=600,
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
        sys.stderr.write(f"missing fixture(s): {args.snapshots} / {args.trades}\n")
        return 2

    with tempfile.TemporaryDirectory(prefix="mmsim_parity_models_") as tmp:
        out_dir = Path(tmp)
        snap_csv, trade_csv = parquet_to_csv(args.snapshots, args.trades, out_dir)
        rs = run_rust(snap_csv, trade_csv)
        py = run_python(args.snapshots, args.trades)

    # Comparison rule:
    #   - Integer keys (n_emit / n_skip / n_snaps / probe_*_idx / probe_*_ts): EXACT
    #   - Per-probe price strings: EXACT to 12 decimals
    #   - Aggregate mean prices over ~35k snapshots: rel 1e-9 (sum-order noise)
    AGGREGATE_FLOAT_KEYS = {k for k in py if k.endswith("_mean_bid") or k.endswith("_mean_ask")}
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
        # Handle None values.
        if py[k] == "None" or rs[k] == "None":
            print(f"  {k}: py={py[k]} rs={rs[k]} [FAIL — None mismatch]")
            failures.append(k)
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
        # Per-snapshot probe prices: tight tolerance (1e-12).
        if rel <= 1e-12:
            print(f"  {k}: py={py[k]} rs={rs[k]} rel={rel:.2e} [OK ~exact]")
            continue
        print(f"  {k}: py={py[k]} rs={rs[k]} rel={rel:.2e} [FAIL]")
        failures.append(k)

    if failures:
        print(f"\nMODELS PARITY FAILED: {len(failures)} mismatch(es)")
        return 1
    print("\nMODELS PARITY OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
