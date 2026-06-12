//! Fair-anchored quoting model.
//!
//! Anchors on an `EWMAFairTracker`: the reference price
//! is the EWMA of the observed mid, fed once per quoter call.  The
//! model quotes symmetrically around the tracker.
//!
//! Formula
//! -------
//!   f(t) = ewma(mid(t))                  # half-life decay updated each call
//!   bid_px = f(t) - half_spread
//!   ask_px = f(t) + half_spread

#![cfg(feature = "models")]

use crate::ingest::Book;
use crate::quoter::refprice::{top_mid, EWMAFairTracker};
use crate::quoter::{Decision, Quoter};
use crate::sim::sim_loop::QuoteRequest;

pub struct FairAnchoredQuoter {
    pub half_spread: f64,
    pub size: f64,
    tracker: EWMAFairTracker,
}

impl FairAnchoredQuoter {
    pub fn new(half_spread: f64, size: f64, half_life_ns: i64) -> Self {
        Self {
            half_spread,
            size,
            tracker: EWMAFairTracker::new(half_life_ns),
        }
    }
}

impl Quoter for FairAnchoredQuoter {
    fn quote(&mut self, book: Option<&Book>, _inv: f64, t_ns: i64) -> Vec<Decision> {
        if let Some(mid) = top_mid(book) {
            self.tracker.observe(t_ns, mid);
        }
        let f = match self.tracker.value(t_ns) {
            Some(v) => v,
            None => return Vec::new(),
        };
        vec![
            Decision::Maker(QuoteRequest {
                side: 1, price: f - self.half_spread, size: self.size,
            }),
            Decision::Maker(QuoteRequest {
                side: -1, price: f + self.half_spread, size: self.size,
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
    fn first_call_seeds_ewma_to_mid() {
        let mut q = FairAnchoredQuoter::new(0.5, 0.001, 1_000);
        let b = book(0, vec![(100.0, 5.0)], vec![(101.0, 5.0)]);
        let out = q.quote(Some(&b), 0.0, 0);
        let bid = match &out[0] { Decision::Maker(q) => q, _ => panic!() };
        // EWMA seeded to mid=100.5; bid = 100.0
        assert!((bid.price - 100.0).abs() < 1e-12);
    }

    #[test]
    fn ewma_decays_at_half_life() {
        let mut q = FairAnchoredQuoter::new(0.5, 0.001, 1_000);
        q.quote(Some(&book(0, vec![(100.0, 5.0)], vec![(101.0, 5.0)])), 0.0, 0);
        // Second call at dt=half_life: ewma = 0.5*100.5 + 0.5*102.5 = 101.5
        let out = q.quote(
            Some(&book(1_000, vec![(102.0, 5.0)], vec![(103.0, 5.0)])),
            0.0, 1_000);
        let bid = match &out[0] { Decision::Maker(q) => q, _ => panic!() };
        assert!((bid.price - 101.0).abs() < 1e-12);
    }

    #[test]
    fn no_book_no_quotes_before_warmup() {
        let mut q = FairAnchoredQuoter::new(0.5, 0.001, 1_000);
        assert!(q.quote(None, 0.0, 0).is_empty());
    }

    #[test]
    fn deterministic_reruns() {
        // Same inputs → identical outputs.
        let mut q1 = FairAnchoredQuoter::new(0.5, 0.001, 1_000);
        let mut q2 = FairAnchoredQuoter::new(0.5, 0.001, 1_000);
        let books = [
            book(0, vec![(100.0, 5.0)], vec![(101.0, 5.0)]),
            book(500, vec![(101.0, 5.0)], vec![(102.0, 5.0)]),
            book(1_500, vec![(99.0, 5.0)], vec![(100.0, 5.0)]),
        ];
        for (i, b) in books.iter().enumerate() {
            let t = i as i64 * 500;
            let a = q1.quote(Some(b), 0.0, t);
            let bb = q2.quote(Some(b), 0.0, t);
            for (x, y) in a.iter().zip(bb.iter()) {
                match (x, y) {
                    (Decision::Maker(x), Decision::Maker(y)) => {
                        assert!((x.price - y.price).abs() < 1e-12);
                    }
                    _ => panic!(),
                }
            }
        }
    }
}
