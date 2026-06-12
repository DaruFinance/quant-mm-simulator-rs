//! Symmetric quoting model.
//!
//! Fixed half-spread, ref-price-centred bid + ask of fixed size.
//!
//! Formula
//! -------
//! Given a reference price `s(t)` (chosen by the caller's `ref_fn`):
//!     bid_px = s(t) - half_spread
//!     ask_px = s(t) + half_spread
//! Both sides at the same `size`.

#![cfg(feature = "models")]

use crate::ingest::Book;
use crate::quoter::{Decision, Quoter};
use crate::sim::sim_loop::QuoteRequest;

/// Reference-price strategy enum.  Kept simple (vs. closure with
/// lifetimes) so the struct stays `Clone + Send` and parity-friendly.
#[derive(Debug, Clone, Copy)]
pub enum RefStrategy {
    /// `top_mid(book) = (best_bid + best_ask) / 2`.
    TopMid,
    /// `weighted_mid(book)` — size-weighted mid using TOB queue sizes.
    WeightedMid,
    /// `microprice(book)` — Stoikov microprice.
    Microprice,
}

impl RefStrategy {
    pub fn eval(self, book: Option<&Book>) -> Option<f64> {
        use crate::quoter::refprice::{microprice, top_mid as tm, weighted_mid};
        match self {
            RefStrategy::TopMid => tm(book),
            RefStrategy::WeightedMid => weighted_mid(book),
            RefStrategy::Microprice => microprice(book),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SymmetricQuoter {
    pub half_spread: f64,
    pub size: f64,
    pub ref_strategy: RefStrategy,
}

impl SymmetricQuoter {
    /// Default: top_mid ref.
    pub fn new(half_spread: f64, size: f64) -> Self {
        Self {
            half_spread,
            size,
            ref_strategy: RefStrategy::TopMid,
        }
    }

    pub fn with_ref(mut self, strategy: RefStrategy) -> Self {
        self.ref_strategy = strategy;
        self
    }
}

impl Quoter for SymmetricQuoter {
    fn quote(&mut self, book: Option<&Book>, _inv: f64, _t_ns: i64) -> Vec<Decision> {
        let ref_px = match self.ref_strategy.eval(book) {
            Some(p) => p,
            None => return Vec::new(),
        };
        vec![
            Decision::Maker(QuoteRequest {
                side: 1,
                price: ref_px - self.half_spread,
                size: self.size,
            }),
            Decision::Maker(QuoteRequest {
                side: -1,
                price: ref_px + self.half_spread,
                size: self.size,
            }),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn book(ts: i64, bids: Vec<(f64, f64)>, asks: Vec<(f64, f64)>) -> Book {
        Book { ts_ns: ts, bids, asks }
    }

    #[test]
    fn basic_emit_with_top_mid() {
        let mut q = SymmetricQuoter::new(0.5, 0.001);
        let b = book(0, vec![(100.0, 5.0)], vec![(101.0, 5.0)]);
        let out = q.quote(Some(&b), 0.0, 0);
        assert_eq!(out.len(), 2);
        let bid = match &out[0] { Decision::Maker(q) => q, _ => panic!() };
        let ask = match &out[1] { Decision::Maker(q) => q, _ => panic!() };
        assert!((bid.price - 100.0).abs() < 1e-12);
        assert!((ask.price - 101.0).abs() < 1e-12);
    }

    #[test]
    fn empty_when_no_book() {
        let mut q = SymmetricQuoter::new(0.5, 0.001);
        let out = q.quote(None, 0.0, 0);
        assert!(out.is_empty());
    }

    #[test]
    fn inv_does_not_affect_prices() {
        let mut q = SymmetricQuoter::new(0.5, 0.001);
        let b = book(0, vec![(100.0, 5.0)], vec![(101.0, 5.0)]);
        let a = q.quote(Some(&b), 0.0, 0);
        let bb = q.quote(Some(&b), 100.0, 0);
        // Symmetric model has no inv skew.
        for (x, y) in a.iter().zip(bb.iter()) {
            match (x, y) {
                (Decision::Maker(x), Decision::Maker(y)) => assert_eq!(x.price, y.price),
                _ => panic!(),
            }
        }
    }

    #[test]
    fn microprice_ref_shifts_with_imbalance() {
        let mut q = SymmetricQuoter::new(0.5, 0.001)
            .with_ref(RefStrategy::Microprice);
        // bid-heavy book → microprice > mid → both quotes shift up.
        let b = book(0, vec![(100.0, 9.0)], vec![(101.0, 1.0)]);
        let out = q.quote(Some(&b), 0.0, 0);
        let bid = match &out[0] { Decision::Maker(q) => q, _ => panic!() };
        // mid=100.5, microprice = 100.5 + (9-1)/10 * 0.5 = 100.9
        // bid = 100.4
        assert!((bid.price - 100.4).abs() < 1e-12);
    }

    #[test]
    fn weighted_mid_ref_matches_microprice_at_tob() {
        let mut qm = SymmetricQuoter::new(0.5, 0.001).with_ref(RefStrategy::Microprice);
        let mut qw = SymmetricQuoter::new(0.5, 0.001).with_ref(RefStrategy::WeightedMid);
        let b = book(0, vec![(100.0, 3.0)], vec![(102.0, 7.0)]);
        let m = qm.quote(Some(&b), 0.0, 0);
        let w = qw.quote(Some(&b), 0.0, 0);
        // microprice == weighted_mid at TOB-only depth → identical quotes.
        for (x, y) in m.iter().zip(w.iter()) {
            match (x, y) {
                (Decision::Maker(x), Decision::Maker(y)) => {
                    assert!((x.price - y.price).abs() < 1e-12);
                }
                _ => panic!(),
            }
        }
    }
}
