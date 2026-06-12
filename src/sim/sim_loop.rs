//! Event-driven sim loop.
//!
//! Mirror of Python's `mmsim.sim.loop`.  Public surface:
//!   - `Order` (mutable; loop shrinks `size` on partial fills)
//!   - `QuoteRequest` (immutable; what the quoter asks to post)
//!   - `Fill` (immutable; one realised fill record)
//!   - `SimResult` (immutable; loop output)
//!   - `run_sim(events, &mut quoter, &mut fills) -> SimResult`
//!
//! Quoter / fills are passed as `FnMut` closures so callers can
//! carry mutable internal state if a stateful quoter wants it.
//! The mirror's reference stubs (used by the parity binary) live
//! alongside `tests/` per the same convention as the Python sibling.
//!
//! Causality: at each event with `ts_ns == t`, callbacks see only
//! events with index `<= t` (those already consumed).  Verified by
//! the leak test at the bottom of this file and by the parity
//! script's pollute battery.

#![cfg(feature = "sim")]

use crate::ingest::{Book, Event, EventStream, TradeEvent};

#[derive(Debug, Clone)]
pub struct Order {
    pub order_id: u64,
    /// `+1` for a bid (our buy), `-1` for an ask (our sell).
    pub side: i32,
    pub price: f64,
    pub size: f64,
    pub placed_at_ns: i64,
}

#[derive(Debug, Clone, Copy)]
pub struct QuoteRequest {
    pub side: i32,
    pub price: f64,
    pub size: f64,
}

#[derive(Debug, Clone)]
pub struct Fill {
    pub fill_id: u64,
    pub order_id: u64,
    pub ts_ns: i64,
    pub price: f64,
    pub size: f64,
    pub side: i32,
    pub is_maker: bool,
}

#[derive(Debug, Clone)]
pub struct SimResult {
    pub fills: Vec<Fill>,
    pub n_events_processed: usize,
    pub n_snapshot_events: usize,
    pub n_trade_events: usize,
    pub n_quoter_calls: usize,
    ///: maker fills emitted by the FillModel on TradeEvents.
    /// Always 0 in the closure-based `run_sim` path (no taker
    /// support there).
    pub n_maker_fills: usize,
    ///: taker fills emitted by the FillModel on TakerRequests.
    /// Always 0 in the closure-based `run_sim` path.
    pub n_taker_fills: usize,
    pub final_orders: Vec<Order>,
}

/// Drive an event stream through the loop.
///
/// `quoter` is `FnMut(Option<&Book>, &[Order], i64) -> Vec<QuoteRequest>`.
/// `fills` is `FnMut(&[Order], &TradeEvent) -> Vec<(u64, f64)>` — pairs
/// of `(order_id, fill_size)` to fill against this trade event.
pub fn run_sim<Q, F>(
    events: &EventStream,
    mut quoter: Q,
    mut fills: F,
) -> SimResult
where
    Q: FnMut(Option<&Book>, &[Order], i64) -> Vec<QuoteRequest>,
    F: FnMut(&[Order], &TradeEvent) -> Vec<(u64, f64)>,
{
    let mut active: Vec<Order> = Vec::new();
    let mut out_fills: Vec<Fill> = Vec::new();
    let mut next_order_id: u64 = 0;
    let mut next_fill_id: u64 = 0;
    let mut n_snap = 0usize;
    let mut n_trade = 0usize;
    let mut n_quoter_calls = 0usize;

    for ev in events {
        match ev {
            Event::Snapshot(s) => {
                n_snap += 1;
                let book = Book {
                    ts_ns: s.ts_ns,
                    bids: s.bids.clone(),
                    asks: s.asks.clone(),
                };
                let quotes = quoter(Some(&book), &active, s.ts_ns);
                n_quoter_calls += 1;
                // Replace active orders with newly-IDed orders.
                let mut new_active: Vec<Order> = Vec::with_capacity(quotes.len());
                for q in quotes {
                    new_active.push(Order {
                        order_id: next_order_id,
                        side: q.side,
                        price: q.price,
                        size: q.size,
                        placed_at_ns: s.ts_ns,
                    });
                    next_order_id += 1;
                }
                active = new_active;
            }
            Event::Trade(t) => {
                n_trade += 1;
                if active.is_empty() {
                    continue;
                }
                let hits = fills(&active, t);
                for (oid, fsize) in hits {
                    let pos = active.iter().position(|o| o.order_id == oid)
                        .unwrap_or_else(|| panic!(
                            "run_sim: order_id={} not in active set at trade ts={}",
                            oid, t.ts_ns));
                    if fsize <= 0.0 {
                        panic!("run_sim: non-positive fill size {}", fsize);
                    }
                    let actual = fsize.min(active[pos].size);
                    out_fills.push(Fill {
                        fill_id: next_fill_id,
                        order_id: active[pos].order_id,
                        ts_ns: t.ts_ns,
                        price: active[pos].price,
                        size: actual,
                        side: active[pos].side,
                        is_maker: true,
                    });
                    next_fill_id += 1;
                    active[pos].size -= actual;
                }
                active.retain(|o| o.size > 0.0);
            }
        }
    }

    SimResult {
        fills: out_fills,
        n_events_processed: events.len(),
        n_snapshot_events: n_snap,
        n_trade_events: n_trade,
        n_quoter_calls,
        n_maker_fills: 0,
        n_taker_fills: 0,
        final_orders: active,
    }
}

// --------------------------------------------------------------------- //
//: model-based event loop.  Layered on top of the same
// per-event walk; differs from `run_sim` in that the fills hook is a
// stateful `FillModel` with full lifecycle hooks (on_order_placed,
// on_orders_removed, on_snapshot, on_trade, fill_taker), and the
// quoter may return TakerRequests alongside QuoteRequests.
// --------------------------------------------------------------------- //

/// What a quoter returns at each snapshot.  Mirrors Python's union
/// type; in Rust we discriminate via this enum.
#[derive(Debug, Clone)]
pub enum QuoterItem {
    Maker(QuoteRequest),
    Taker(crate::sim::fills::TakerRequest),
}

/// Model-based loop.  `quoter` returns a list of QuoterItem;
/// `model` is the stateful FillModel that gets every lifecycle hook
/// the Python sibling fires.
pub fn run_sim_with_model<Q>(
    events: &EventStream,
    mut quoter: Q,
    model: &mut crate::sim::fills::FillModel,
) -> SimResult
where
    Q: FnMut(Option<&Book>, &[Order], i64) -> Vec<QuoterItem>,
{
    let mut active: Vec<Order> = Vec::new();
    let mut out_fills: Vec<Fill> = Vec::new();
    let mut next_order_id: u64 = 0;
    let mut next_fill_id: u64 = 0;
    let mut n_snap = 0usize;
    let mut n_trade = 0usize;
    let mut n_quoter_calls = 0usize;
    let mut n_maker_fills = 0usize;
    let mut n_taker_fills = 0usize;

    for ev in events {
        match ev {
            Event::Snapshot(s) => {
                n_snap += 1;
                let book = Book {
                    ts_ns: s.ts_ns, bids: s.bids.clone(), asks: s.asks.clone(),
                };
                // 1) Push snapshot to the model so trackers can do
                //    cancel attribution before any new orders post.
                model.on_snapshot(s);
                // 2) Quoter decides.
                let items = quoter(Some(&book), &active, s.ts_ns);
                n_quoter_calls += 1;
                let mut makers: Vec<QuoteRequest> = Vec::new();
                let mut takers: Vec<crate::sim::fills::TakerRequest> = Vec::new();
                for it in items {
                    match it {
                        QuoterItem::Maker(q) => makers.push(q),
                        QuoterItem::Taker(t) => takers.push(t),
                    }
                }
                // 3) Replace active makers; notify the model.
                let removed: Vec<u64> = active.iter().map(|o| o.order_id).collect();
                if !removed.is_empty() {
                    model.on_orders_removed(&removed);
                }
                let mut new_active: Vec<Order> = Vec::with_capacity(makers.len());
                for q in makers {
                    let o = Order {
                        order_id: next_order_id,
                        side: q.side, price: q.price, size: q.size,
                        placed_at_ns: s.ts_ns,
                    };
                    model.on_order_placed(&o, &book);
                    new_active.push(o);
                    next_order_id += 1;
                }
                active = new_active;
                // 4) Fire takers immediately.
                for req in &takers {
                    let rows = model.fill_taker(req, &book, s.ts_ns);
                    for (px, sz) in rows {
                        if sz <= 0.0 {
                            continue;
                        }
                        out_fills.push(Fill {
                            fill_id: next_fill_id,
                            order_id: u64::MAX,  // taker not tied to a resting order
                            ts_ns: s.ts_ns,
                            price: px,
                            size: sz,
                            side: req.side,
                            is_maker: false,
                        });
                        next_fill_id += 1;
                        n_taker_fills += 1;
                    }
                }
            }
            Event::Trade(t) => {
                n_trade += 1;
                if active.is_empty() {
                    continue;
                }
                let hits = model.on_trade(t, &active);
                for (oid, fsize) in hits {
                    let pos = active.iter().position(|o| o.order_id == oid)
                        .unwrap_or_else(|| panic!(
                            "run_sim_with_model: order_id={} not in active set at ts={}",
                            oid, t.ts_ns));
                    if fsize <= 0.0 {
                        panic!("run_sim_with_model: non-positive fill size {}", fsize);
                    }
                    let actual = fsize.min(active[pos].size);
                    out_fills.push(Fill {
                        fill_id: next_fill_id,
                        order_id: active[pos].order_id,
                        ts_ns: t.ts_ns,
                        price: active[pos].price,
                        size: actual,
                        side: active[pos].side,
                        is_maker: true,
                    });
                    next_fill_id += 1;
                    n_maker_fills += 1;
                    active[pos].size -= actual;
                }
                active.retain(|o| o.size > 0.0);
            }
        }
    }

    SimResult {
        fills: out_fills,
        n_events_processed: events.len(),
        n_snapshot_events: n_snap,
        n_trade_events: n_trade,
        n_quoter_calls,
        n_maker_fills,
        n_taker_fills,
        final_orders: active,
    }
}

// --------------------------------------------------------------------- //
// Reference stubs.  Used by the parity binary; they are NOT production.
// They will be replaced by the formal quoter trait and maker/taker
// fill model.  Mirrors Python's tests/test_sim_loop.py
// stubs 1:1 so the parity binary produces identical fills.
// --------------------------------------------------------------------- //

pub const STUB_QUOTE_SIZE: f64 = 0.001;

pub fn stub_quoter_top_of_book(
    book: Option<&Book>,
    _active: &[Order],
    _t_ns: i64,
) -> Vec<QuoteRequest> {
    match book {
        Some(b) => match (b.best_bid(), b.best_ask()) {
            (Some(bid), Some(ask)) => vec![
                QuoteRequest { side: 1, price: bid, size: STUB_QUOTE_SIZE },
                QuoteRequest { side: -1, price: ask, size: STUB_QUOTE_SIZE },
            ],
            _ => Vec::new(),
        },
        None => Vec::new(),
    }
}

pub fn stub_fills_naive(
    active: &[Order],
    trade: &TradeEvent,
) -> Vec<(u64, f64)> {
    if trade.size <= 0.0 || trade.side == 0 {
        return Vec::new();
    }
    // Iterate by ascending order_id for deterministic priority.
    let mut sorted: Vec<&Order> = active.iter().collect();
    sorted.sort_by_key(|o| o.order_id);
    if trade.side == -1 {
        for o in &sorted {
            if o.side == 1 && trade.price <= o.price {
                return vec![(o.order_id, trade.size.min(o.size))];
            }
        }
    } else if trade.side == 1 {
        for o in &sorted {
            if o.side == -1 && trade.price >= o.price {
                return vec![(o.order_id, trade.size.min(o.size))];
            }
        }
    }
    Vec::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ingest::SnapshotEvent;

    fn snap(ts: i64, bids: Vec<(f64, f64)>, asks: Vec<(f64, f64)>) -> Event {
        Event::Snapshot(SnapshotEvent {
            ts_ns: ts, recv_ns: ts,
            symbol: "BTC-USDT".into(), venue: "binance".into(),
            depth: bids.len() as u32, bids, asks,
        })
    }

    fn trade(ts: i64, price: f64, size: f64, side: i32) -> Event {
        Event::Trade(TradeEvent {
            ts_ns: ts, recv_ns: ts,
            symbol: "BTC-USDT".into(), venue: "binance".into(),
            price, size, side,
        })
    }

    #[test]
    fn loop_emits_a_fill_when_trade_crosses_our_quote() {
        let stream = vec![
            snap(100, vec![(100.0, 1.0)], vec![(101.0, 1.0)]),
            trade(150, 101.0, STUB_QUOTE_SIZE, 1),  // buy aggressor at our ask
        ];
        let res = run_sim(&stream, stub_quoter_top_of_book, stub_fills_naive);
        assert_eq!(res.fills.len(), 1);
        let f = &res.fills[0];
        assert_eq!(f.side, -1);  // ask fill (we sold)
        assert_eq!(f.price, 101.0);
        assert!((f.size - STUB_QUOTE_SIZE).abs() < 1e-12);
    }

    #[test]
    fn loop_partial_fill_decrements_order_size() {
        let stream = vec![
            snap(100, vec![(100.0, 1.0)], vec![(101.0, 1.0)]),
            trade(200, 100.0, STUB_QUOTE_SIZE / 2.0, -1),  // sell aggressor: partial bid hit
            trade(300, 100.0, STUB_QUOTE_SIZE * 2.0, -1),  // sell aggressor: remainder
        ];
        let res = run_sim(&stream, stub_quoter_top_of_book, stub_fills_naive);
        let bid_fills: Vec<&Fill> = res.fills.iter().filter(|f| f.side == 1).collect();
        assert_eq!(bid_fills.len(), 2);
        let total: f64 = bid_fills.iter().map(|f| f.size).sum();
        assert!((total - STUB_QUOTE_SIZE).abs() < 1e-12);
    }

    #[test]
    fn loop_reproducible_across_reruns() {
        let stream = vec![
            snap(100, vec![(100.0, 1.0)], vec![(101.0, 1.0)]),
            trade(200, 101.0, STUB_QUOTE_SIZE, 1),
            snap(300, vec![(99.0, 1.0)], vec![(102.0, 1.0)]),
            trade(350, 99.0, STUB_QUOTE_SIZE, -1),
        ];
        let a = run_sim(&stream, stub_quoter_top_of_book, stub_fills_naive);
        let b = run_sim(&stream, stub_quoter_top_of_book, stub_fills_naive);
        assert_eq!(a.fills.len(), b.fills.len());
        for (fa, fb) in a.fills.iter().zip(b.fills.iter()) {
            assert_eq!(fa.ts_ns, fb.ts_ns);
            assert_eq!(fa.price, fb.price);
            assert_eq!(fa.size, fb.size);
            assert_eq!(fa.side, fb.side);
        }
    }
}
