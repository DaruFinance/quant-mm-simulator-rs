//! Avellaneda-Stoikov quoting model.
//!
//! Closed-form maker quotes derived in Avellaneda & Stoikov (2008),
//! "High-Frequency Trading in a Limit Order Book".
//!
//! Reservation:
//!     r(s, q, t) = s - q · γ · σ² · (T - t)
//!
//! Optimal half-spread:
//!     half_spread* = (γ · σ² · (T - t)) / 2 + (1/γ) · ln(1 + γ / k)
//!
//! Quoted prices:
//!     bid_px = r - half_spread*
//!     ask_px = r + half_spread*

#![cfg(feature = "models")]

use crate::ingest::Book;
use crate::quoter::refprice::top_mid;
use crate::quoter::{Decision, Quoter};
use crate::sim::sim_loop::QuoteRequest;

use super::vol::RollingSigma;

pub struct AvellanedaStoikovQuoter {
    pub gamma: f64,
    pub k: f64,
    pub horizon_ns: i64,
    pub size: f64,
    pub sigma_floor: f64,
    sigma: RollingSigma,
    t_start_ns: Option<i64>,
}

impl AvellanedaStoikovQuoter {
    pub fn new(
        gamma: f64, k: f64, horizon_ns: i64, size: f64, vol_window_ns: i64,
    ) -> Self {
        assert!(gamma > 0.0, "gamma must be > 0");
        assert!(k > 0.0, "k must be > 0");
        assert!(horizon_ns > 0, "horizon_ns must be > 0");
        assert!(size > 0.0, "size must be > 0");
        assert!(vol_window_ns > 0, "vol_window_ns must be > 0");
        Self {
            gamma, k, horizon_ns, size, sigma_floor: 1e-9,
            sigma: RollingSigma::new(vol_window_ns),
            t_start_ns: None,
        }
    }

    pub fn with_sigma_floor(mut self, floor: f64) -> Self {
        assert!(floor >= 0.0, "sigma_floor must be >= 0");
        self.sigma_floor = floor;
        self
    }

    fn time_to_go(&mut self, t_ns: i64) -> f64 {
        if self.t_start_ns.is_none() {
            self.t_start_ns = Some(t_ns);
        }
        let elapsed = t_ns - self.t_start_ns.unwrap();
        let remaining = self.horizon_ns as f64 - elapsed as f64;
        remaining.max(1.0)
    }
}

impl Quoter for AvellanedaStoikovQuoter {
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
        let tau = self.time_to_go(t_ns);
        let r = mid - inv * self.gamma * sigma * sigma * tau;
        let spread_half = (self.gamma * sigma * sigma * tau) / 2.0
            + (1.0 / self.gamma) * (1.0 + self.gamma / self.k).ln();
        vec![
            Decision::Maker(QuoteRequest {
                side: 1, price: r - spread_half, size: self.size,
            }),
            Decision::Maker(QuoteRequest {
                side: -1, price: r + spread_half, size: self.size,
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
        let mut q = AvellanedaStoikovQuoter::new(0.1, 1.5, 1_000_000_000_000, 0.001, 1_000_000_000);
        assert!(q.quote(Some(&book(0, 100.0, 5.0, 101.0, 5.0)), 0.0, 0).is_empty());
        let out = q.quote(Some(&book(1_000_000, 100.1, 5.0, 101.1, 5.0)), 0.0, 1_000_000);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn no_book_no_quotes() {
        let mut q = AvellanedaStoikovQuoter::new(0.1, 1.5, 1_000_000_000_000, 0.001, 1_000_000_000);
        assert!(q.quote(None, 0.0, 0).is_empty());
    }

    #[test]
    fn inv_shifts_reservation_down_when_long() {
        let mut q = AvellanedaStoikovQuoter::new(0.5, 1.5, 1_000_000_000_000, 0.001, 1_000_000_000_000);
        q.quote(Some(&book(0, 100.0, 5.0, 101.0, 5.0)), 0.0, 0);
        let out_flat = q.quote(Some(&book(1_000_000, 100.1, 5.0, 101.1, 5.0)), 0.0, 1_000_000);
        let out_long = q.quote(Some(&book(2_000_000, 100.1, 5.0, 101.1, 5.0)), 10.0, 2_000_000);
        let bid_flat = match &out_flat[0] { Decision::Maker(q) => q.price, _ => panic!() };
        let bid_long = match &out_long[0] { Decision::Maker(q) => q.price, _ => panic!() };
        assert!(bid_long < bid_flat);
    }

    #[test]
    fn higher_sigma_widens_spread() {
        let mut q_low = AvellanedaStoikovQuoter::new(0.5, 1.5, 1_000_000_000_000, 0.001, 1_000_000_000_000);
        let mut q_hi = AvellanedaStoikovQuoter::new(0.5, 1.5, 1_000_000_000_000, 0.001, 1_000_000_000_000);
        // Low-vol: small swings.
        for (i, p) in [100.0_f64, 100.001, 100.0, 100.001].iter().enumerate() {
            q_low.quote(Some(&book(i as i64 * 1_000_000, *p, 5.0, *p + 1.0, 5.0)), 0.0, i as i64 * 1_000_000);
        }
        // High-vol: big swings.
        for (i, p) in [100.0_f64, 110.0, 100.0, 110.0].iter().enumerate() {
            q_hi.quote(Some(&book(i as i64 * 1_000_000, *p, 5.0, *p + 1.0, 5.0)), 0.0, i as i64 * 1_000_000);
        }
        let out_low = q_low.quote(Some(&book(10_000_000, 100.0, 5.0, 101.0, 5.0)), 0.0, 10_000_000);
        let out_hi = q_hi.quote(Some(&book(10_000_000, 100.0, 5.0, 101.0, 5.0)), 0.0, 10_000_000);
        let sp_low = match (&out_low[0], &out_low[1]) {
            (Decision::Maker(b), Decision::Maker(a)) => a.price - b.price, _ => panic!() };
        let sp_hi = match (&out_hi[0], &out_hi[1]) {
            (Decision::Maker(b), Decision::Maker(a)) => a.price - b.price, _ => panic!() };
        assert!(sp_hi > sp_low);
    }

    #[test]
    fn deterministic_reruns() {
        let mut q1 = AvellanedaStoikovQuoter::new(0.5, 1.5, 1_000_000_000_000, 0.001, 1_000_000_000_000);
        let mut q2 = AvellanedaStoikovQuoter::new(0.5, 1.5, 1_000_000_000_000, 0.001, 1_000_000_000_000);
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
