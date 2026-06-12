//! Ho-Stoll quoting model.
//!
//! Classical 1981 inventory-driven dealer model from Ho & Stoll (1981).
//!
//! Total spread is `α + β · |q| · σ²`; reservation centres on mid plus
//! a linear inventory skew:
//!     half_spread = (α + β · |q| · σ²) / 2
//!     skew         = -β · q · σ²             # signed; long inv ⇒ shift DOWN
//!     bid_px = s + skew - half_spread
//!     ask_px = s + skew + half_spread

#![cfg(feature = "models")]

use crate::ingest::Book;
use crate::quoter::refprice::top_mid;
use crate::quoter::{Decision, Quoter};
use crate::sim::sim_loop::QuoteRequest;

use super::vol::RollingSigma;

pub struct HoStollQuoter {
    pub alpha: f64,
    pub beta: f64,
    pub size: f64,
    pub sigma_floor: f64,
    sigma: RollingSigma,
}

impl HoStollQuoter {
    pub fn new(alpha: f64, beta: f64, size: f64, vol_window_ns: i64) -> Self {
        assert!(alpha >= 0.0, "alpha must be >= 0");
        assert!(beta > 0.0, "beta must be > 0");
        assert!(size > 0.0, "size must be > 0");
        assert!(vol_window_ns > 0, "vol_window_ns must be > 0");
        Self {
            alpha, beta, size, sigma_floor: 1e-9,
            sigma: RollingSigma::new(vol_window_ns),
        }
    }

    pub fn with_sigma_floor(mut self, floor: f64) -> Self {
        assert!(floor >= 0.0, "sigma_floor must be >= 0");
        self.sigma_floor = floor;
        self
    }
}

impl Quoter for HoStollQuoter {
    fn quote(&mut self, book: Option<&Book>, inv: f64, t_ns: i64) -> Vec<Decision> {
        let mid = match top_mid(book) {
            Some(m) => m,
            None => return Vec::new(),
        };
        self.sigma.observe(t_ns, mid);
        let sigma = match self.sigma.value(t_ns) {
            Some(s) => s.max(self.sigma_floor),
            None => return Vec::new(),
        };
        let var = sigma * sigma;
        let spread = self.alpha + self.beta * inv.abs() * var;
        let half_spread = spread / 2.0;
        let skew = -self.beta * inv * var;
        let center = mid + skew;
        vec![
            Decision::Maker(QuoteRequest {
                side: 1, price: center - half_spread, size: self.size,
            }),
            Decision::Maker(QuoteRequest {
                side: -1, price: center + half_spread, size: self.size,
            }),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn book(ts: i64, bid_px: f64, bid_sz: f64, ask_px: f64, ask_sz: f64) -> Book {
        Book { ts_ns: ts, bids: vec![(bid_px, bid_sz)], asks: vec![(ask_px, ask_sz)] }
    }

    #[test]
    fn warmup_no_quotes_until_sigma_ready() {
        let mut q = HoStollQuoter::new(0.5, 1.0, 0.001, 1_000_000_000);
        assert!(q.quote(Some(&book(0, 100.0, 5.0, 101.0, 5.0)), 0.0, 0).is_empty());
        let out = q.quote(Some(&book(1_000_000, 100.1, 5.0, 101.1, 5.0)), 0.0, 1_000_000);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn no_book_no_quotes() {
        let mut q = HoStollQuoter::new(0.5, 1.0, 0.001, 1_000_000_000);
        assert!(q.quote(None, 0.0, 0).is_empty());
    }

    #[test]
    fn long_inv_skews_both_down() {
        let mut q = HoStollQuoter::new(0.5, 10.0, 0.001, 1_000_000_000_000);
        q.quote(Some(&book(0, 100.0, 5.0, 101.0, 5.0)), 0.0, 0);
        q.quote(Some(&book(1_000_000, 110.0, 5.0, 111.0, 5.0)), 0.0, 1_000_000);
        let out_flat = q.quote(Some(&book(2_000_000, 100.0, 5.0, 101.0, 5.0)), 0.0, 2_000_000);
        let out_long = q.quote(Some(&book(3_000_000, 100.0, 5.0, 101.0, 5.0)), 5.0, 3_000_000);
        let bid_flat = match &out_flat[0] { Decision::Maker(q) => q.price, _ => panic!() };
        let bid_long = match &out_long[0] { Decision::Maker(q) => q.price, _ => panic!() };
        assert!(bid_long < bid_flat);
    }

    #[test]
    fn large_inv_widens_spread() {
        let mut q_s = HoStollQuoter::new(0.5, 10.0, 0.001, 1_000_000_000_000);
        let mut q_l = HoStollQuoter::new(0.5, 10.0, 0.001, 1_000_000_000_000);
        for q in [&mut q_s, &mut q_l].iter_mut() {
            q.quote(Some(&book(0, 100.0, 5.0, 101.0, 5.0)), 0.0, 0);
            q.quote(Some(&book(1_000_000, 110.0, 5.0, 111.0, 5.0)), 0.0, 1_000_000);
        }
        let out_s = q_s.quote(Some(&book(2_000_000, 100.0, 5.0, 101.0, 5.0)), 0.1, 2_000_000);
        let out_l = q_l.quote(Some(&book(2_000_000, 100.0, 5.0, 101.0, 5.0)), 5.0, 2_000_000);
        let sp_s = match (&out_s[0], &out_s[1]) {
            (Decision::Maker(b), Decision::Maker(a)) => a.price - b.price, _ => panic!() };
        let sp_l = match (&out_l[0], &out_l[1]) {
            (Decision::Maker(b), Decision::Maker(a)) => a.price - b.price, _ => panic!() };
        assert!(sp_l > sp_s);
    }

    #[test]
    fn deterministic_reruns() {
        let mut q1 = HoStollQuoter::new(0.5, 1.0, 0.001, 1_000_000_000);
        let mut q2 = HoStollQuoter::new(0.5, 1.0, 0.001, 1_000_000_000);
        for (i, p) in [100.0_f64, 100.5, 101.0, 100.5, 100.0].iter().enumerate() {
            let t = i as i64 * 1_000_000;
            let a = q1.quote(Some(&book(t, *p, 5.0, *p + 1.0, 5.0)), 0.0, t);
            let bb = q2.quote(Some(&book(t, *p, 5.0, *p + 1.0, 5.0)), 0.0, t);
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
