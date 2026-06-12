//! Ladder quoting model.
//!
//! Wraps the `ladder()` shape primitive inside the
//! `Quoter` trait.  For each level k = 0..n_levels-1:
//!     offset_k = half_spread + k · step
//!     bid_k_px = ref(t) - offset_k
//!     ask_k_px = ref(t) + offset_k
//! All levels at the same `size_per_level`.

#![cfg(feature = "models")]

use crate::ingest::Book;
use crate::quoter::shapes::{ladder, LadderSpec};
use crate::quoter::{Decision, Quoter};

use super::symmetric::RefStrategy;

#[derive(Debug, Clone)]
pub struct LadderQuoter {
    pub half_spread: f64,
    pub step: f64,
    pub n_levels: usize,
    pub size_per_level: f64,
    pub ref_strategy: RefStrategy,
}

impl LadderQuoter {
    pub fn new(half_spread: f64, step: f64, n_levels: usize, size_per_level: f64) -> Self {
        Self {
            half_spread,
            step,
            n_levels,
            size_per_level,
            ref_strategy: RefStrategy::TopMid,
        }
    }

    pub fn with_ref(mut self, strategy: RefStrategy) -> Self {
        self.ref_strategy = strategy;
        self
    }
}

impl Quoter for LadderQuoter {
    fn quote(&mut self, book: Option<&Book>, inv: f64, _t_ns: i64) -> Vec<Decision> {
        let ref_px = match self.ref_strategy.eval(book) {
            Some(p) => p,
            None => return Vec::new(),
        };
        let spec = LadderSpec {
            half_spread: self.half_spread,
            step: self.step,
            n_levels: self.n_levels,
            size_per_level: self.size_per_level,
        };
        ladder(&spec, ref_px, inv)
            .into_iter()
            .map(Decision::Maker)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn book(ts: i64, bids: Vec<(f64, f64)>, asks: Vec<(f64, f64)>) -> Book {
        Book { ts_ns: ts, bids, asks }
    }

    #[test]
    fn basic_emit_three_levels() {
        let mut q = LadderQuoter::new(0.5, 0.25, 3, 0.001);
        let b = book(0, vec![(100.0, 5.0)], vec![(101.0, 5.0)]);
        let out = q.quote(Some(&b), 0.0, 0);
        assert_eq!(out.len(), 6);  // 3 × 2 sides
    }

    #[test]
    fn no_book_no_quotes() {
        let mut q = LadderQuoter::new(0.5, 0.25, 3, 0.001);
        assert!(q.quote(None, 0.0, 0).is_empty());
    }

    #[test]
    fn inv_does_not_affect_prices() {
        let mut q = LadderQuoter::new(0.5, 0.25, 3, 0.001);
        let b = book(0, vec![(100.0, 5.0)], vec![(101.0, 5.0)]);
        let a = q.quote(Some(&b), 0.0, 0);
        let bb = q.quote(Some(&b), 100.0, 0);
        for (x, y) in a.iter().zip(bb.iter()) {
            match (x, y) {
                (Decision::Maker(x), Decision::Maker(y)) => assert_eq!(x.price, y.price),
                _ => panic!(),
            }
        }
    }

    #[test]
    fn ref_at_100_5_half_0_5_step_0_25_levels_match_python() {
        let mut q = LadderQuoter::new(0.5, 0.25, 3, 0.001);
        let b = book(0, vec![(100.0, 5.0)], vec![(101.0, 5.0)]);
        let out = q.quote(Some(&b), 0.0, 0);
        // ref=100.5; level 0: bid=100.0, ask=101.0
        // level 1: bid=99.75, ask=101.25; level 2: bid=99.5, ask=101.5
        let bid_prices: Vec<f64> = out.iter().filter_map(|d| match d {
            Decision::Maker(q) if q.side == 1 => Some(q.price),
            _ => None,
        }).collect();
        assert!((bid_prices[0] - 100.0).abs() < 1e-12);
        assert!((bid_prices[1] - 99.75).abs() < 1e-12);
        assert!((bid_prices[2] - 99.50).abs() < 1e-12);
    }

    #[test]
    fn zero_levels_empty() {
        let mut q = LadderQuoter::new(0.5, 0.25, 0, 0.001);
        let b = book(0, vec![(100.0, 5.0)], vec![(101.0, 5.0)]);
        assert!(q.quote(Some(&b), 0.0, 0).is_empty());
    }
}
