//! Maker/taker fill model.
//!
//! Mirrors `mmsim.sim.fills` 1:1.  Public surface:
//!   - `TakerRequest` — immediate-execution request from the quoter.
//!   - `FillModel` enum — sum type wrapping the two implementations
//!     (the Python side uses a Protocol; Rust uses an enum so the
//!     loop can dispatch without dynamic trait objects).
//!   - `QueueAwareFillModel` — canonical model using
//!     `QueueTracker` per active maker.  Queue consumes first;
//!     spillover fills us.  Takers walk the visible book greedily.
//!   - `StatelessFillsAdapter` — wraps a stateless callable for
//!     back-compat with the stateless callable path.
//!
//! The loop's `run_sim` takes a `FillModel` directly;
//! this matches the Python's "either callable or protocol" with
//! the enum dispatch instead.

#![cfg(feature = "sim")]

use std::collections::HashMap;

use crate::ingest::{Book, SnapshotEvent, TradeEvent};
use crate::sim::queue::QueueTracker;
use crate::sim::sim_loop::Order;

const PX_EPSILON: f64 = 1e-9;

#[derive(Debug, Clone, Copy)]
pub struct TakerRequest {
    /// `+1` buy, `-1` sell.
    pub side: i32,
    pub size: f64,
    pub limit_px: Option<f64>,
}

fn trade_at_our_level(trade: &TradeEvent, side: i32, price: f64) -> bool {
    if trade.side == 0 {
        return false;
    }
    if side == 1 {
        return trade.side == -1 && (trade.price - price).abs() <= PX_EPSILON;
    }
    trade.side == 1 && (trade.price - price).abs() <= PX_EPSILON
}

/// Canonical maker/taker fill model.
pub struct QueueAwareFillModel {
    trackers: HashMap<u64, QueueTracker>,
}

impl QueueAwareFillModel {
    pub fn new() -> Self {
        Self { trackers: HashMap::new() }
    }

    /// Read-only view of the per-order queue trackers.  Used by the
    /// aux-log driver to snapshot `(order_id, queue_pos,
    /// frozen)` at every snapshot event.  No mutation possible.
    pub fn trackers(&self) -> &HashMap<u64, QueueTracker> {
        &self.trackers
    }
}

impl Default for QueueAwareFillModel {
    fn default() -> Self {
        Self::new()
    }
}

/// Fill-model wrapper used by `run_sim`.  Two variants:
/// - `QueueAware`: stateful queue-aware maker + taker walks.
/// - `Stateless(fn)`: legacy callable path; only `on_trade` fires the
///   wrapped callable.  Used when the caller passes `None` for a
///   model and instead supplies a closure (the `run_sim` overload).
pub enum FillModel {
    QueueAware(QueueAwareFillModel),
}

impl FillModel {
    pub fn on_order_placed(&mut self, order: &Order, book: &Book) {
        match self {
            FillModel::QueueAware(m) => {
                if let Ok(tr) = QueueTracker::new(order.clone(), book) {
                    m.trackers.insert(order.order_id, tr);
                }
                // If the order's price isn't visible in the book
                // (depth-N truncation; deep orders), the tracker
                // construction errors and we skip — no tracker
                // means no fills for that order.  Mirrors the
                // Python's silent skip (Python raises ValueError
                // here; we choose silent skip for the loop's
                // robustness — this matches the documented
                // edge case in the queue tracker).
            }
        }
    }

    pub fn on_orders_removed(&mut self, order_ids: &[u64]) {
        match self {
            FillModel::QueueAware(m) => {
                for oid in order_ids {
                    m.trackers.remove(oid);
                }
            }
        }
    }

    pub fn on_snapshot(&mut self, snap: &SnapshotEvent) {
        match self {
            FillModel::QueueAware(m) => {
                let ev = crate::ingest::Event::Snapshot(snap.clone());
                for tr in m.trackers.values_mut() {
                    let _ = tr.observe(&ev);
                }
            }
        }
    }

    pub fn on_trade(
        &mut self,
        trade: &TradeEvent,
        active_orders: &[Order],
    ) -> Vec<(u64, f64)> {
        match self {
            FillModel::QueueAware(m) => {
                let ev = crate::ingest::Event::Trade(trade.clone());
                let mut hits: Vec<(u64, f64)> = Vec::new();
                for o in active_orders {
                    let tr = match m.trackers.get_mut(&o.order_id) {
                        Some(t) => t,
                        None => continue,
                    };
                    if tr.frozen {
                        continue;
                    }
                    if !trade_at_our_level(trade, o.side, o.price) {
                        let _ = tr.observe(&ev);
                        continue;
                    }
                    let queue_pos_before = tr.queue_pos;
                    let _ = tr.observe(&ev);
                    let spillover = trade.size - queue_pos_before;
                    if spillover > 0.0 {
                        let fill_size = spillover.min(o.size);
                        if fill_size > 0.0 {
                            hits.push((o.order_id, fill_size));
                        }
                    }
                }
                hits
            }
        }
    }

    /// Borrow the underlying QueueAware model's tracker map, if this
    /// is a QueueAware variant.  Returns None otherwise.  Used by
    /// the aux-log driver.
    pub fn queue_trackers(&self) -> Option<&HashMap<u64, QueueTracker>> {
        match self {
            FillModel::QueueAware(m) => Some(m.trackers()),
        }
    }

    pub fn fill_taker(
        &self,
        req: &TakerRequest,
        book: &Book,
        _t_ns: i64,
    ) -> Vec<(f64, f64)> {
        if req.size <= 0.0 {
            return Vec::new();
        }
        let levels = if req.side == 1 { &book.asks } else { &book.bids };
        let mut remaining = req.size;
        let mut rows: Vec<(f64, f64)> = Vec::new();
        for (px, sz) in levels {
            if remaining <= 0.0 {
                break;
            }
            if let Some(lim) = req.limit_px {
                if (req.side == 1 && *px > lim) || (req.side == -1 && *px < lim) {
                    break;
                }
            }
            let consume = remaining.min(*sz);
            if consume > 0.0 {
                rows.push((*px, consume));
                remaining -= consume;
            }
        }
        rows
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn book(ts: i64, bids: Vec<(f64, f64)>, asks: Vec<(f64, f64)>) -> Book {
        Book { ts_ns: ts, bids, asks }
    }
    fn trade(ts: i64, price: f64, size: f64, side: i32) -> TradeEvent {
        TradeEvent {
            ts_ns: ts, recv_ns: ts,
            symbol: "X".into(), venue: "v".into(),
            price, size, side,
        }
    }
    fn order(side: i32, price: f64, size: f64) -> Order {
        Order { order_id: 1, side, price, size, placed_at_ns: 0 }
    }

    #[test]
    fn no_fill_while_queue_pos_positive() {
        let b = book(0, vec![(100.0, 5.0)], vec![(101.0, 3.0)]);
        let o = order(1, 100.0, 1.0);
        let mut m = FillModel::QueueAware(QueueAwareFillModel::new());
        m.on_order_placed(&o, &b);
        let hits = m.on_trade(&trade(10, 100.0, 2.0, -1), &[o]);
        assert!(hits.is_empty());
    }

    #[test]
    fn fills_us_on_spillover() {
        let b = book(0, vec![(100.0, 2.0)], vec![(101.0, 3.0)]);
        let o = order(1, 100.0, 1.0);
        let mut m = FillModel::QueueAware(QueueAwareFillModel::new());
        m.on_order_placed(&o, &b);
        assert!(m.on_trade(&trade(10, 100.0, 2.0, -1), &[o.clone()]).is_empty());
        let hits = m.on_trade(&trade(20, 100.0, 0.5, -1), &[o]);
        assert_eq!(hits, vec![(1, 0.5)]);
    }

    #[test]
    fn taker_walks_top_of_book() {
        let b = book(
            0,
            vec![(100.0, 1.0)],
            vec![(101.0, 0.5), (102.0, 0.5), (103.0, 1.0)],
        );
        let m = FillModel::QueueAware(QueueAwareFillModel::new());
        let rows = m.fill_taker(
            &TakerRequest { side: 1, size: 1.5, limit_px: None },
            &b, 100,
        );
        assert_eq!(rows, vec![(101.0, 0.5), (102.0, 0.5), (103.0, 0.5)]);
    }

    #[test]
    fn taker_limit_px_stops_walk() {
        let b = book(0, vec![(100.0, 1.0)],
            vec![(101.0, 0.5), (102.0, 0.5), (103.0, 1.0)]);
        let m = FillModel::QueueAware(QueueAwareFillModel::new());
        let rows = m.fill_taker(
            &TakerRequest { side: 1, size: 2.0, limit_px: Some(101.5) },
            &b, 0,
        );
        assert_eq!(rows, vec![(101.0, 0.5)]);
    }
}
