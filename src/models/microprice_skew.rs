//! Microprice-skew quoting model.
//!
//! Uses the Stoikov `microprice()` as the reference price; bid/ask
//! are symmetric around it.  When the book is heavy on the bid
//! (likely upward drift) the microprice sits above the simple mid
//! and our quotes shift up.
//!
//! Formula
//! -------
//!   s(t) = microprice(book) = mid + ((bid_sz - ask_sz)/(bid_sz + ask_sz)) · half_book_spread
//!   bid_px = s(t) - half_spread
//!   ask_px = s(t) + half_spread

#![cfg(feature = "models")]

use crate::ingest::Book;
use crate::quoter::refprice::microprice;
use crate::quoter::{Decision, Quoter};
use crate::sim::sim_loop::QuoteRequest;

#[derive(Debug, Clone)]
pub struct MicropriceSkewQuoter {
    pub half_spread: f64,
    pub size: f64,
}

impl MicropriceSkewQuoter {
    pub fn new(half_spread: f64, size: f64) -> Self {
        Self { half_spread, size }
    }
}

impl Quoter for MicropriceSkewQuoter {
    fn quote(&mut self, book: Option<&Book>, _inv: f64, _t_ns: i64) -> Vec<Decision> {
        let ref_px = match microprice(book) {
            Some(p) => p,
            None => return Vec::new(),
        };
        vec![
            Decision::Maker(QuoteRequest {
                side: 1, price: ref_px - self.half_spread, size: self.size,
            }),
            Decision::Maker(QuoteRequest {
                side: -1, price: ref_px + self.half_spread, size: self.size,
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
    fn balanced_book_quotes_at_mid_plus_minus_half() {
        let mut q = MicropriceSkewQuoter::new(0.5, 0.001);
        let b = book(0, vec![(100.0, 5.0)], vec![(101.0, 5.0)]);
        let out = q.quote(Some(&b), 0.0, 0);
        let bid = match &out[0] { Decision::Maker(q) => q, _ => panic!() };
        let ask = match &out[1] { Decision::Maker(q) => q, _ => panic!() };
        assert!((bid.price - 100.0).abs() < 1e-12);
        assert!((ask.price - 101.0).abs() < 1e-12);
    }

    #[test]
    fn bid_heavy_shifts_quotes_up() {
        let mut q = MicropriceSkewQuoter::new(0.5, 0.001);
        let b = book(0, vec![(100.0, 9.0)], vec![(101.0, 1.0)]);
        let out = q.quote(Some(&b), 0.0, 0);
        let bid = match &out[0] { Decision::Maker(q) => q, _ => panic!() };
        let ask = match &out[1] { Decision::Maker(q) => q, _ => panic!() };
        // microprice = 100.9; bid=100.4, ask=101.4
        assert!((bid.price - 100.4).abs() < 1e-12);
        assert!((ask.price - 101.4).abs() < 1e-12);
    }

    #[test]
    fn ask_heavy_shifts_quotes_down() {
        let mut q = MicropriceSkewQuoter::new(0.5, 0.001);
        let b = book(0, vec![(100.0, 1.0)], vec![(101.0, 9.0)]);
        let out = q.quote(Some(&b), 0.0, 0);
        let bid = match &out[0] { Decision::Maker(q) => q, _ => panic!() };
        // microprice = 100.5 + (1-9)/10*0.5 = 100.1; bid = 99.6
        assert!((bid.price - 99.6).abs() < 1e-12);
    }

    #[test]
    fn no_book_no_quotes() {
        let mut q = MicropriceSkewQuoter::new(0.5, 0.001);
        assert!(q.quote(None, 0.0, 0).is_empty());
    }

    #[test]
    fn inv_does_not_affect_quotes() {
        let mut q = MicropriceSkewQuoter::new(0.5, 0.001);
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
}
