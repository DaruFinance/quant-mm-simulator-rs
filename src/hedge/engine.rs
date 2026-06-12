//! Hedge engine.
//!
//! Mirror of Python's `mmsim.hedge.engine`.  Hedges MM inventory
//! in a separate instrument; emits `Fill` records on the hedge
//! venue's book with a sentinel `order_id = HEDGE_ORDER_ID`.
//!
//! Causality: every state mutation reads only the current `Fill`
//! (for inventory accumulation) or the current `Book` (for hedge
//! decision + price).  No future fills, no future books.
//!
//! See Python sibling for full semantic docs.

#![cfg(feature = "hedge")]

use crate::ingest::Book;
use crate::sim::sim_loop::Fill;

/// Sentinel order_id for hedge fills.  Distinct from the primary
/// taker sentinel (`u64::MAX`) used in `run_sim_with_model` and
/// the regular maker orders (sequential u64 from 0).
pub const HEDGE_ORDER_ID: u64 = u64::MAX - 1;

#[derive(Debug, Clone, Copy)]
pub struct HedgeDecision {
    pub t_ns: i64,
    pub pre_inv: f64,
    pub post_inv: f64,
    pub hedge_size: f64,
    pub hedge_side: i32,
    pub hedge_px: f64,
    pub pre_hedge_inv: f64,
    pub post_hedge_inv: f64,
    pub net_delta_pre: f64,
    pub net_delta_post: f64,
}

/// Stateful hedge engine.  See module docs (and the Python mirror)
/// for full semantic definitions.
#[derive(Debug, Clone)]
pub struct HedgeEngine {
    pub threshold: f64,
    pub hedge_size_pct: f64,
    pub instrument: String,
    pub inv: f64,
    pub hedge_inv: f64,
    pub hedge_fills: Vec<Fill>,
    pub decisions: Vec<HedgeDecision>,
    next_fill_id: u64,
}

impl HedgeEngine {
    pub fn new(threshold: f64, hedge_size_pct: f64, instrument: &str) -> Self {
        if threshold < 0.0 {
            panic!("threshold must be >= 0, got {}", threshold);
        }
        if !(hedge_size_pct > 0.0 && hedge_size_pct <= 1.0) {
            panic!("hedge_size_pct must be in (0, 1], got {}", hedge_size_pct);
        }
        Self {
            threshold,
            hedge_size_pct,
            instrument: instrument.to_string(),
            inv: 0.0,
            hedge_inv: 0.0,
            hedge_fills: Vec::new(),
            decisions: Vec::new(),
            next_fill_id: 0,
        }
    }

    pub fn net_delta(&self) -> f64 {
        self.inv + self.hedge_inv
    }

    pub fn n_hedge_fires(&self) -> usize {
        self.hedge_fills.len()
    }

    /// Accumulate a primary-side fill into `self.inv`.  Defensive:
    /// silently ignores any fill carrying `HEDGE_ORDER_ID` so the
    /// caller cannot double-count by passing hedge fills back in.
    pub fn observe_fill(&mut self, fill: &Fill) {
        if fill.order_id == HEDGE_ORDER_ID {
            return;
        }
        self.inv += fill.size * (fill.side as f64);
    }

    /// True if `|net_delta| >= threshold`.
    pub fn should_hedge(&self) -> bool {
        self.net_delta().abs() >= self.threshold
    }

    /// Emit a hedge fill that drives `net_delta` toward zero.
    ///
    /// Returns the emitted Fill (also appended to `hedge_fills` +
    /// `decisions`) — or `None` when `should_hedge()` is false, the
    /// supplied book is missing the relevant best price, or the
    /// computed size is non-positive.
    pub fn make_hedge(&mut self, book: Option<&Book>, t_ns: i64) -> Option<Fill> {
        if !self.should_hedge() {
            return None;
        }
        let b = book?;
        let nd = self.net_delta();
        let size = nd.abs() * self.hedge_size_pct;
        if size <= 0.0 {
            return None;
        }
        let (hedge_side, px_opt) = if nd > 0.0 {
            (-1i32, b.best_bid())
        } else {
            (1i32, b.best_ask())
        };
        let px = px_opt?;

        let fill = Fill {
            fill_id: self.next_fill_id,
            order_id: HEDGE_ORDER_ID,
            ts_ns: t_ns,
            price: px,
            size,
            side: hedge_side,
            is_maker: false,
        };
        let pre_hedge_inv = self.hedge_inv;
        let signed = size * (hedge_side as f64);
        self.hedge_inv += signed;
        self.hedge_fills.push(fill.clone());
        self.next_fill_id += 1;

        self.decisions.push(HedgeDecision {
            t_ns,
            pre_inv: self.inv,
            post_inv: self.inv,
            hedge_size: size,
            hedge_side,
            hedge_px: px,
            pre_hedge_inv,
            post_hedge_inv: self.hedge_inv,
            net_delta_pre: self.inv + pre_hedge_inv,
            net_delta_post: self.inv + self.hedge_inv,
        });
        Some(fill)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fill(ts: i64, side: i32, size: f64) -> Fill {
        Fill {
            fill_id: ts as u64,
            order_id: 0,
            ts_ns: ts,
            price: 100.0,
            size,
            side,
            is_maker: true,
        }
    }

    fn book(bid: f64, ask: f64) -> Book {
        Book {
            ts_ns: 0,
            bids: vec![(bid, 1.0)],
            asks: vec![(ask, 1.0)],
        }
    }

    #[test]
    fn does_not_hedge_below_threshold() {
        let mut he = HedgeEngine::new(0.5, 1.0, "perp");
        he.observe_fill(&fill(1, 1, 0.1));
        assert!(!he.should_hedge());
        let b = book(99.0, 101.0);
        assert!(he.make_hedge(Some(&b), 10).is_none());
        assert_eq!(he.n_hedge_fires(), 0);
    }

    #[test]
    fn hedges_when_long_at_threshold() {
        let mut he = HedgeEngine::new(0.5, 1.0, "perp");
        // Cumulate to +0.5 -> hits threshold.
        he.observe_fill(&fill(1, 1, 0.3));
        he.observe_fill(&fill(2, 1, 0.2));
        assert!((he.inv - 0.5).abs() < 1e-12);
        assert!(he.should_hedge());
        let b = book(99.0, 101.0);
        let hf = he.make_hedge(Some(&b), 10).expect("should fire");
        // Long -> sell hedge at best bid.
        assert_eq!(hf.side, -1);
        assert!((hf.price - 99.0).abs() < 1e-12);
        assert!((hf.size - 0.5).abs() < 1e-12);
        assert!(!hf.is_maker);
        assert_eq!(hf.order_id, HEDGE_ORDER_ID);
        // Net delta now ~0.
        assert!(he.net_delta().abs() < 1e-12);
    }

    #[test]
    fn hedges_when_short_buys_at_best_ask() {
        let mut he = HedgeEngine::new(0.5, 1.0, "perp");
        he.observe_fill(&fill(1, -1, 0.6));   // inv = -0.6
        assert!(he.should_hedge());
        let b = book(99.0, 101.0);
        let hf = he.make_hedge(Some(&b), 10).expect("should fire");
        assert_eq!(hf.side, 1);                // buy
        assert!((hf.price - 101.0).abs() < 1e-12);
        assert!((hf.size - 0.6).abs() < 1e-12);
        assert!(he.net_delta().abs() < 1e-12);
    }

    #[test]
    fn partial_hedge_uses_pct() {
        let mut he = HedgeEngine::new(0.5, 0.5, "perp");
        he.observe_fill(&fill(1, 1, 1.0));   // inv = +1.0
        let b = book(99.0, 101.0);
        let hf = he.make_hedge(Some(&b), 10).expect("fire");
        // 50% of |1.0| -> 0.5 hedge size.
        assert!((hf.size - 0.5).abs() < 1e-12);
        // Net delta = 1.0 - 0.5 = 0.5.
        assert!((he.net_delta() - 0.5).abs() < 1e-12);
    }

    #[test]
    fn ignores_hedge_fills_fed_back() {
        let mut he = HedgeEngine::new(0.5, 1.0, "perp");
        he.observe_fill(&fill(1, 1, 0.6));
        let b = book(99.0, 101.0);
        let hf = he.make_hedge(Some(&b), 10).unwrap();
        // Re-feeding the hedge fill must NOT touch primary inv.
        let before = he.inv;
        he.observe_fill(&hf);
        assert_eq!(he.inv, before);
    }

    #[test]
    fn missing_book_returns_none() {
        let mut he = HedgeEngine::new(0.5, 1.0, "perp");
        he.observe_fill(&fill(1, 1, 0.6));
        assert!(he.make_hedge(None, 10).is_none());
        assert_eq!(he.n_hedge_fires(), 0);
    }

    #[test]
    fn empty_side_of_book_returns_none() {
        let mut he = HedgeEngine::new(0.5, 1.0, "perp");
        he.observe_fill(&fill(1, 1, 0.6));   // long -> needs best bid
        let b = Book { ts_ns: 0, bids: vec![], asks: vec![(101.0, 1.0)] };
        assert!(he.make_hedge(Some(&b), 10).is_none());
        assert_eq!(he.n_hedge_fires(), 0);
    }

    #[test]
    fn decision_log_captures_pre_post() {
        let mut he = HedgeEngine::new(0.5, 1.0, "perp");
        he.observe_fill(&fill(1, 1, 0.6));
        let b = book(99.0, 101.0);
        he.make_hedge(Some(&b), 10);
        assert_eq!(he.decisions.len(), 1);
        let d = &he.decisions[0];
        assert!((d.net_delta_pre - 0.6).abs() < 1e-12);
        assert!(d.net_delta_post.abs() < 1e-12);
        assert!((d.hedge_size - 0.6).abs() < 1e-12);
        assert_eq!(d.hedge_side, -1);
    }
}
