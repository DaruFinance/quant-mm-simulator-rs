//! T4 corpus runner.
//!
//! Mirror of Python's `mmsim.t4_corpus.t4_runner`. Drives the existing
//! `run_sim_with_model` engine for one (combo, asset) configuration on
//! a real LOB EventStream, emitting per-fill ledger rows with the
//! cost-stack applied:
//!
//!   - maker fee: 0.02% of notional
//!   - taker fee: 0.05% of notional
//!   - slippage:  0.02% of notional (one-sided per fill)
//!
//! These match the project-wide crypto defaults (no costless backtests).

#![cfg(feature = "t4-corpus")]

use std::collections::HashMap;

use crate::ingest::{Book, Event, EventStream};
use crate::sim::fills::{FillModel, QueueAwareFillModel};
use crate::sim::sim_loop::{Fill, run_sim_with_model, QuoterItem, QuoteRequest, SimResult};
use crate::quoter::Decision;

use super::combos::Combo;
use super::composer::build_mm_strategy;

pub const MAKER_FEE_PCT: f64 = 2e-4;
pub const TAKER_FEE_PCT: f64 = 5e-4;
pub const SLIP_PCT: f64 = 2e-4;

/// Per-fill leg-row schema. Mirrors Python's `LEG_COLS`.
pub const LEG_COLS: [&str; 13] = [
    "fill_id", "ts_ns", "side", "price", "size", "is_maker",
    "notional", "fee", "slippage", "gross_pnl", "net_pnl",
    "order_id", "trade_group_id",
];

#[derive(Debug, Clone)]
pub struct T4LegRow {
    pub fill_id: u64,
    pub ts_ns: i64,
    pub side: i32,
    pub price: f64,
    pub size: f64,
    pub is_maker: bool,
    pub notional: f64,
    pub fee: f64,
    pub slippage: f64,
    pub gross_pnl: f64,
    pub net_pnl: f64,
    pub order_id: u64,
    pub trade_group_id: u64,
}

#[derive(Debug, Clone)]
pub struct T4Metrics {
    pub n_fills: usize,
    pub n_maker_fills: usize,
    pub n_taker_fills: usize,
    pub n_snapshots: usize,
    pub n_trades: usize,
    pub n_quoter_calls: usize,
    pub total_notional: f64,
    pub total_fees: f64,
    pub total_slippage: f64,
    pub total_cost: f64,
    pub gross_pnl: f64,
    pub net_pnl: f64,
}

#[derive(Debug, Clone)]
pub struct T4RunResult {
    pub combo: Combo,
    pub asset: String,
    pub leg_rows: Vec<T4LegRow>,
    pub metrics: T4Metrics,
    pub n_fills: usize,
    pub n_maker_fills: usize,
    pub n_taker_fills: usize,
}

fn fee_for_fill(fill: &Fill, notional: f64) -> f64 {
    let rate = if fill.is_maker { MAKER_FEE_PCT } else { TAKER_FEE_PCT };
    notional.abs() * rate
}

fn slip_for_fill(_fill: &Fill, notional: f64) -> f64 {
    notional.abs() * SLIP_PCT
}

/// Drive `run_sim_with_model` for one (combo, params) pair on `events`.
///
/// Returns a [`T4RunResult`] holding per-fill leg rows + cost-aware
/// summary metrics. Cost identity: every realized fill contributes a
/// strictly-positive fee + slip.
pub fn run_t4_combo(
    combo: Combo,
    params: HashMap<String, f64>,
    events: &EventStream,
    asset: &str,
) -> T4RunResult {
    let cfg = build_mm_strategy(combo.clone(), params);

    // Adapter state: cached last-emitted decision list (for non-firing
    // refresh triggers).
    let mut last_quotes: Vec<QuoterItem> = Vec::new();

    // We must capture cfg by &mut into the closure; wrap in RefCell.
    let cfg_ref = std::cell::RefCell::new(cfg);

    let quoter_closure = |book: Option<&Book>,
                          _active: &[crate::sim::sim_loop::Order],
                          t_ns: i64| -> Vec<QuoterItem> {
        let mut c = cfg_ref.borrow_mut();

        // 1) Adverse filter — observe + gate.
        if let Some(af) = c.adverse_filter.as_mut() {
            if let Some(b) = book {
                af.observe_book(b);
            }
            if af.is_adverse(t_ns) {
                last_quotes.clear();
                return Vec::new();
            }
        }
        // 2) Refresh trigger — observe; only refresh on fire.
        let inv_now = 0.0; // The Rust run_sim_with_quoter threads the
                            // inventory but run_sim_with_model doesn't —
                            // we pass 0 here (closures in run_sim_with_model
                            // don't get inv). The quoter call below
                            // operates on book-derived state only.
        let fired = c.refresh_trigger.step(book, inv_now, t_ns);
        if !fired {
            return last_quotes.clone();
        }
        // 3) Call the underlying quoter.
        let raw = c.quoter.quote(book, inv_now, t_ns);
        // 4) Apply inventory-penalty Skew + snap-to-best.
        let skew = (c.inv_penalty_fn)(inv_now);
        let mut out: Vec<QuoterItem> = Vec::with_capacity(raw.len());
        for d in raw {
            match d {
                Decision::Maker(q) => {
                    let mut new_price = q.price + skew.price_offset;
                    let scale = if q.side == 1 {
                        skew.size_scale_bid
                    } else {
                        skew.size_scale_ask
                    };
                    let new_size = q.size * scale;
                    if new_size <= 0.0 {
                        continue;
                    }
                    // Snap-to-best: align to nearest visible level so the
                    // QueueAwareFillModel can construct a queue tracker.
                    if let Some(b) = book {
                        if q.side == 1 {
                            if let Some(bb) = b.best_bid() {
                                new_price = bb;
                            }
                        } else if let Some(ba) = b.best_ask() {
                            new_price = ba;
                        }
                    }
                    out.push(QuoterItem::Maker(QuoteRequest {
                        side: q.side, price: new_price, size: new_size,
                    }));
                }
                Decision::Taker(t) => {
                    out.push(QuoterItem::Taker(t));
                }
            }
        }
        last_quotes = out.clone();
        out
    };

    let mut fill_model = FillModel::QueueAware(QueueAwareFillModel::new());
    let sim_res: SimResult = run_sim_with_model(events, quoter_closure, &mut fill_model);

    // Build leg rows. Use the most-recent book at fill time for
    // mark-to-fair PnL.
    let mut book_at_ts: HashMap<i64, Book> = HashMap::new();
    let fill_ts_set: std::collections::BTreeSet<i64> =
        sim_res.fills.iter().map(|f| f.ts_ns).collect();
    if !fill_ts_set.is_empty() {
        let mut last: Option<Book> = None;
        let mut targets = fill_ts_set.iter().peekable();
        for ev in events {
            if let Event::Snapshot(s) = ev {
                last = Some(Book {
                    ts_ns: s.ts_ns,
                    bids: s.bids.clone(),
                    asks: s.asks.clone(),
                });
            }
            while let Some(&&t) = targets.peek() {
                if t <= ev.ts_ns() {
                    if let Some(ref b) = last {
                        book_at_ts.insert(t, b.clone());
                    }
                    targets.next();
                } else {
                    break;
                }
            }
            if targets.peek().is_none() {
                break;
            }
        }
    }

    // Rebuild a fresh ref_fn for marking — independent of the live one
    // mutated during the run.
    let mut ref_marker = super::composer::RefPriceFn::new(
        cfg_ref.borrow().combo.reference_price,
        &cfg_ref.borrow().params,
    );

    let mut leg_rows: Vec<T4LegRow> = Vec::with_capacity(sim_res.fills.len());
    let mut total_fees = 0.0;
    let mut total_slip = 0.0;
    let mut total_gross = 0.0;
    let mut total_net = 0.0;
    let mut n_maker = 0usize;
    let mut n_taker = 0usize;
    for f in &sim_res.fills {
        let notional = (f.price * f.size).abs();
        let fee = fee_for_fill(f, notional);
        let slip = slip_for_fill(f, notional);
        let book_opt = book_at_ts.get(&f.ts_ns);
        let fair = book_opt.and_then(|b| ref_marker.eval(Some(b)));
        let gross = match fair {
            Some(p) => (p - f.price) * (f.side as f64) * f.size,
            None => 0.0,
        };
        let net = gross - fee - slip;
        total_fees += fee;
        total_slip += slip;
        total_gross += gross;
        total_net += net;
        if f.is_maker {
            n_maker += 1;
        } else {
            n_taker += 1;
        }
        leg_rows.push(T4LegRow {
            fill_id: f.fill_id,
            ts_ns: f.ts_ns,
            side: f.side,
            price: f.price,
            size: f.size,
            is_maker: f.is_maker,
            notional,
            fee,
            slippage: slip,
            gross_pnl: gross,
            net_pnl: net,
            order_id: f.order_id,
            trade_group_id: f.fill_id,
        });
    }

    let metrics = T4Metrics {
        n_fills: sim_res.fills.len(),
        n_maker_fills: n_maker,
        n_taker_fills: n_taker,
        n_snapshots: sim_res.n_snapshot_events,
        n_trades: sim_res.n_trade_events,
        n_quoter_calls: sim_res.n_quoter_calls,
        total_notional: sim_res.fills.iter()
            .map(|f| (f.price * f.size).abs())
            .sum::<f64>(),
        total_fees,
        total_slippage: total_slip,
        total_cost: total_fees + total_slip,
        gross_pnl: total_gross,
        net_pnl: total_net,
    };

    T4RunResult {
        combo,
        asset: asset.to_string(),
        leg_rows,
        metrics,
        n_fills: sim_res.fills.len(),
        n_maker_fills: n_maker,
        n_taker_fills: n_taker,
    }
}

/// Small extension trait — give every Event a uniform `ts_ns()` method.
///
/// Although `Event` already carries ts in both variants, exposing a
/// uniform method lets the runner's leg-row builder iterate generic
/// events without matching the sum-type at every call site.
#[allow(dead_code)]
trait EventTs {
    fn ts_ns(&self) -> i64;
}
impl EventTs for Event {
    fn ts_ns(&self) -> i64 {
        match self {
            Event::Snapshot(s) => s.ts_ns,
            Event::Trade(t) => t.ts_ns,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::t4_corpus::combos::Combo;

    fn empty_stream() -> EventStream {
        Vec::new()
    }

    #[test]
    fn cost_constants_match_project_defaults() {
        assert_eq!(MAKER_FEE_PCT, 2e-4);
        assert_eq!(TAKER_FEE_PCT, 5e-4);
        assert_eq!(SLIP_PCT, 2e-4);
    }

    #[test]
    fn empty_stream_produces_zero_fills() {
        let combo = Combo {
            quoting_model: "symmetric",
            inventory_penalty: "linear",
            adverse_filter: "none",
            hedge_mode: "none",
            reference_price: "mid",
            quote_shape: "single",
            refresh_trigger: "book_event",
        };
        let rr = run_t4_combo(combo, HashMap::new(), &empty_stream(), "BTCUSDT");
        assert_eq!(rr.n_fills, 0);
        assert_eq!(rr.metrics.total_cost, 0.0);
    }
}
