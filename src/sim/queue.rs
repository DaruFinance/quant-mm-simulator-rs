//! Queue-position tracker for resting orders.
//!
//! Mirror of Python's `mmsim.sim.queue`.  Pro-rata cancel-attribution
//! model.  `QueueTracker::observe(event)` is the load-bearing call;
//! every state mutation reads only the current event and prior state,
//! so polluting events with `ts_ns > T` cannot change tracker state
//! at any event with `ts_ns <= T`.  The cross-language parity script
//! verifies bit-for-bit agreement on the trajectory and aggregates.

#![cfg(feature = "sim")]

use crate::ingest::{Book, Event, SnapshotEvent, TradeEvent};
use crate::sim::sim_loop::Order;

const PX_EPSILON: f64 = 1e-9;

fn size_at_level(book: &Book, side: i32, price: f64) -> Option<f64> {
    let levels = if side == 1 { &book.bids } else { &book.asks };
    for (px, sz) in levels {
        if (px - price).abs() <= PX_EPSILON {
            return Some(*sz);
        }
    }
    None
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

#[derive(Debug, Clone)]
pub struct QueueSample {
    pub ts_ns: i64,
    pub queue_pos: f64,
    /// "placed" | "fill_ahead" | "cancel_ahead" | "level_lost"
    pub cause: &'static str,
}

#[derive(Debug, Clone)]
pub struct QueueTrace {
    pub samples: Vec<QueueSample>,
    pub total_fills_ahead: f64,
    pub total_cancels_ahead: f64,
    pub final_queue_pos: f64,
    pub frozen: bool,
}

pub struct QueueTracker {
    order: Order,
    pub queue_pos: f64,
    last_level_size: f64,
    pending_trade_volume_at_level: f64,
    pub frozen: bool,
    last_ts_ns: i64,
    pub total_fills_ahead: f64,
    pub total_cancels_ahead: f64,
    pub samples: Vec<QueueSample>,
}

impl QueueTracker {
    pub fn new(order: Order, initial_book: &Book) -> Result<Self, String> {
        if order.size <= 0.0 {
            return Err("order.size must be positive".into());
        }
        if initial_book.ts_ns > order.placed_at_ns {
            return Err(
                "initial_book is from after order placement".into(),
            );
        }
        let initial_size = size_at_level(initial_book, order.side, order.price)
            .ok_or_else(|| {
                format!(
                    "order.price={} not visible at side={} in initial_book",
                    order.price, order.side
                )
            })?;
        let placed_at = order.placed_at_ns;
        let samples = vec![QueueSample {
            ts_ns: placed_at,
            queue_pos: initial_size,
            cause: "placed",
        }];
        Ok(Self {
            order,
            queue_pos: initial_size,
            last_level_size: initial_size,
            pending_trade_volume_at_level: 0.0,
            frozen: false,
            last_ts_ns: placed_at,
            total_fills_ahead: 0.0,
            total_cancels_ahead: 0.0,
            samples,
        })
    }

    pub fn observe(&mut self, event: &Event) -> Result<(), String> {
        let ts = event.ts_ns();
        if ts < self.last_ts_ns {
            return Err(format!(
                "out-of-order event: ev.ts_ns={} < last_ts_ns={}",
                ts, self.last_ts_ns
            ));
        }
        if ts < self.order.placed_at_ns {
            return Ok(());
        }
        if self.frozen {
            self.last_ts_ns = ts;
            return Ok(());
        }
        match event {
            Event::Trade(t) => self.on_trade(t),
            Event::Snapshot(s) => self.on_snapshot(s),
        }
        self.last_ts_ns = ts;
        Ok(())
    }

    fn on_trade(&mut self, trade: &TradeEvent) {
        if !trade_at_our_level(trade, self.order.side, self.order.price) {
            return;
        }
        let consumed_ahead = trade.size.min(self.queue_pos);
        if consumed_ahead > 0.0 {
            self.queue_pos -= consumed_ahead;
            self.total_fills_ahead += consumed_ahead;
            self.samples.push(QueueSample {
                ts_ns: trade.ts_ns,
                queue_pos: self.queue_pos,
                cause: "fill_ahead",
            });
        }
        self.pending_trade_volume_at_level += trade.size;
    }

    fn on_snapshot(&mut self, snap: &SnapshotEvent) {
        let book = Book {
            ts_ns: snap.ts_ns,
            bids: snap.bids.clone(),
            asks: snap.asks.clone(),
        };
        let new_size = size_at_level(&book, self.order.side, self.order.price);
        let new_size = match new_size {
            Some(s) => s,
            None => {
                self.frozen = true;
                self.samples.push(QueueSample {
                    ts_ns: snap.ts_ns,
                    queue_pos: self.queue_pos,
                    cause: "level_lost",
                });
                return;
            }
        };
        let size_after_trades =
            self.last_level_size - self.pending_trade_volume_at_level;
        let net_cancels = size_after_trades - new_size;
        if net_cancels > 0.0 && size_after_trades > 0.0 {
            let fraction = self.queue_pos / size_after_trades;
            let cancels_ahead = (net_cancels * fraction).min(self.queue_pos);
            if cancels_ahead > 0.0 {
                self.queue_pos -= cancels_ahead;
                self.total_cancels_ahead += cancels_ahead;
                self.samples.push(QueueSample {
                    ts_ns: snap.ts_ns,
                    queue_pos: self.queue_pos,
                    cause: "cancel_ahead",
                });
            }
        }
        self.last_level_size = new_size;
        self.pending_trade_volume_at_level = 0.0;
    }

    pub fn trace(&self) -> QueueTrace {
        QueueTrace {
            samples: self.samples.clone(),
            total_fills_ahead: self.total_fills_ahead,
            total_cancels_ahead: self.total_cancels_ahead,
            final_queue_pos: self.queue_pos,
            frozen: self.frozen,
        }
    }
}

/// Convenience: build a fresh tracker, run it through `events`, return the trace.
pub fn track_queue_position(
    order: Order,
    events: &[Event],
    initial_book: &Book,
) -> Result<QueueTrace, String> {
    let mut tr = QueueTracker::new(order, initial_book)?;
    for ev in events {
        tr.observe(ev)?;
    }
    Ok(tr.trace())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ingest::SnapshotEvent;

    fn book(ts: i64, bids: Vec<(f64, f64)>, asks: Vec<(f64, f64)>) -> Book {
        Book { ts_ns: ts, bids, asks }
    }
    fn snap(ts: i64, bids: Vec<(f64, f64)>, asks: Vec<(f64, f64)>) -> Event {
        Event::Snapshot(SnapshotEvent {
            ts_ns: ts, recv_ns: ts,
            symbol: "X".into(), venue: "v".into(),
            depth: bids.len() as u32, bids, asks,
        })
    }
    fn trade(ts: i64, price: f64, size: f64, side: i32) -> Event {
        Event::Trade(TradeEvent {
            ts_ns: ts, recv_ns: ts,
            symbol: "X".into(), venue: "v".into(),
            price, size, side,
        })
    }
    fn order(side: i32, price: f64, size: f64, placed: i64) -> Order {
        Order { order_id: 0, side, price, size, placed_at_ns: placed }
    }

    #[test]
    fn initial_queue_pos_equals_visible_size() {
        let b = book(0, vec![(100.0, 5.0), (99.0, 1.0)], vec![(101.0, 3.0)]);
        let tr = QueueTracker::new(order(1, 100.0, 1.0, 0), &b).unwrap();
        assert_eq!(tr.queue_pos, 5.0);
    }

    #[test]
    fn trade_at_level_consumes_from_front() {
        let b = book(0, vec![(100.0, 5.0)], vec![(101.0, 3.0)]);
        let mut tr = QueueTracker::new(order(1, 100.0, 1.0, 0), &b).unwrap();
        tr.observe(&trade(10, 100.0, 2.0, -1)).unwrap();
        assert_eq!(tr.queue_pos, 3.0);
        tr.observe(&trade(20, 100.0, 2.0, -1)).unwrap();
        assert_eq!(tr.queue_pos, 1.0);
        tr.observe(&trade(30, 100.0, 3.0, -1)).unwrap();
        assert_eq!(tr.queue_pos, 0.0);
        assert_eq!(tr.total_fills_ahead, 5.0);
    }

    #[test]
    fn pro_rata_cancel_attribution() {
        let b = book(0, vec![(100.0, 5.0)], vec![(101.0, 3.0)]);
        let mut tr = QueueTracker::new(order(1, 100.0, 1.0, 0), &b).unwrap();
        tr.observe(&snap(10, vec![(100.0, 2.0)], vec![(101.0, 3.0)])).unwrap();
        assert!((tr.queue_pos - 2.0).abs() < 1e-12);
        assert!((tr.total_cancels_ahead - 3.0).abs() < 1e-12);
    }

    #[test]
    fn pro_rata_partial_with_trade_then_cancel() {
        let b = book(0, vec![(100.0, 4.0)], vec![(101.0, 3.0)]);
        let mut tr = QueueTracker::new(order(1, 100.0, 1.0, 0), &b).unwrap();
        tr.observe(&trade(5, 100.0, 2.0, -1)).unwrap();
        assert_eq!(tr.queue_pos, 2.0);
        tr.observe(&snap(10, vec![(100.0, 1.0)], vec![(101.0, 3.0)])).unwrap();
        assert!((tr.queue_pos - 1.0).abs() < 1e-12);
        assert_eq!(tr.total_fills_ahead, 2.0);
        assert!((tr.total_cancels_ahead - 1.0).abs() < 1e-12);
    }

    #[test]
    fn level_drop_freezes_tracker() {
        let b = book(0, vec![(100.0, 5.0), (99.0, 2.0)], vec![(101.0, 3.0)]);
        let mut tr = QueueTracker::new(order(1, 99.0, 1.0, 0), &b).unwrap();
        tr.observe(&snap(10, vec![(100.0, 5.0)], vec![(101.0, 3.0)])).unwrap();
        assert!(tr.frozen);
    }

    #[test]
    fn out_of_order_observation_errors() {
        let b = book(0, vec![(100.0, 5.0)], vec![(101.0, 3.0)]);
        let mut tr = QueueTracker::new(order(1, 100.0, 1.0, 0), &b).unwrap();
        tr.observe(&trade(10, 100.0, 1.0, -1)).unwrap();
        let r = tr.observe(&trade(5, 100.0, 1.0, -1));
        assert!(r.is_err());
    }
}
