//! GLFT (Guéant-Lehalle-Fernandez-Tapia) quoting model.
//!
//! Closed-form market-making quotes from Guéant, Lehalle &
//! Fernandez-Tapia (2013).  Extends Avellaneda-Stoikov by giving the
//! spread a market-order arrival-intensity term (parameter A) in
//! addition to the AS intensity-decay term (parameter k).
//!
//! Reservation (same as AS):
//!     r(s, q, t) = s - q · γ · σ² · (T - t)
//!
//! Asymptotic GLFT half-spread:
//!     half_spread* = (1/γ) ln(1 + γ/k)
//!                  + sqrt(σ² γ / (2 k A)) · (1 + γ/k)^((1 + k/γ) / 2)

#![cfg(feature = "models")]

use crate::ingest::Book;
use crate::quoter::refprice::top_mid;
use crate::quoter::{Decision, Quoter};
use crate::sim::sim_loop::QuoteRequest;

use super::vol::RollingSigma;

pub struct GLFTQuoter {
    pub gamma: f64,
    pub k: f64,
    /// Market-order baseline arrival intensity (> 0).
    pub a_param: f64,
    pub horizon_ns: i64,
    pub size: f64,
    pub sigma_floor: f64,
    sigma: RollingSigma,
    t_start_ns: Option<i64>,
}

impl GLFTQuoter {
    pub fn new(
        gamma: f64, k: f64, a_param: f64, horizon_ns: i64,
        size: f64, vol_window_ns: i64,
    ) -> Self {
        assert!(gamma > 0.0, "gamma must be > 0");
        assert!(k > 0.0, "k must be > 0");
        assert!(a_param > 0.0, "a_param must be > 0");
        assert!(horizon_ns > 0, "horizon_ns must be > 0");
        assert!(size > 0.0, "size must be > 0");
        assert!(vol_window_ns > 0, "vol_window_ns must be > 0");
        Self {
            gamma, k, a_param, horizon_ns, size, sigma_floor: 1e-9,
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
        (self.horizon_ns as f64 - elapsed as f64).max(1.0)
    }
}

impl Quoter for GLFTQuoter {
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
        // AS reservation
        let r = mid - inv * self.gamma * sigma * sigma * tau;
        // GLFT half-spread
        let as_term = (1.0 / self.gamma) * (1.0 + self.gamma / self.k).ln();
        let glft_extra = (sigma * sigma * self.gamma / (2.0 * self.k * self.a_param)).sqrt()
            * (1.0 + self.gamma / self.k).powf((1.0 + self.k / self.gamma) / 2.0);
        let spread_half = as_term + glft_extra;
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
        let mut q = GLFTQuoter::new(0.1, 1.5, 140.0, 1_000_000_000_000, 0.001, 1_000_000_000);
        assert!(q.quote(Some(&book(0, 100.0, 5.0, 101.0, 5.0)), 0.0, 0).is_empty());
        let out = q.quote(Some(&book(1_000_000, 100.1, 5.0, 101.1, 5.0)), 0.0, 1_000_000);
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn no_book_no_quotes() {
        let mut q = GLFTQuoter::new(0.1, 1.5, 140.0, 1_000_000_000_000, 0.001, 1_000_000_000);
        assert!(q.quote(None, 0.0, 0).is_empty());
    }

    #[test]
    fn inv_shifts_reservation_down_when_long() {
        let mut q = GLFTQuoter::new(0.5, 1.5, 140.0, 1_000_000_000_000, 0.001, 1_000_000_000_000);
        q.quote(Some(&book(0, 100.0, 5.0, 101.0, 5.0)), 0.0, 0);
        let out_flat = q.quote(Some(&book(1_000_000, 100.1, 5.0, 101.1, 5.0)), 0.0, 1_000_000);
        let out_long = q.quote(Some(&book(2_000_000, 100.1, 5.0, 101.1, 5.0)), 10.0, 2_000_000);
        let bid_flat = match &out_flat[0] { Decision::Maker(q) => q.price, _ => panic!() };
        let bid_long = match &out_long[0] { Decision::Maker(q) => q.price, _ => panic!() };
        assert!(bid_long < bid_flat);
    }

    #[test]
    fn spread_exceeds_as_floor() {
        let gamma = 0.5_f64;
        let k = 1.5_f64;
        let mut q = GLFTQuoter::new(gamma, k, 140.0, 1_000_000_000_000, 0.001, 1_000_000_000_000);
        for (i, p) in [100.0_f64, 100.5, 101.0, 100.5, 100.0].iter().enumerate() {
            let t = i as i64 * 1_000_000;
            q.quote(Some(&book(t, *p, 5.0, *p + 1.0, 5.0)), 0.0, t);
        }
        let out = q.quote(Some(&book(10_000_000, 100.0, 5.0, 101.0, 5.0)), 0.0, 10_000_000);
        let (bid, ask) = match (&out[0], &out[1]) {
            (Decision::Maker(b), Decision::Maker(a)) => (b.price, a.price),
            _ => panic!(),
        };
        let spread_half = (ask - bid) / 2.0;
        let floor = (1.0 / gamma) * (1.0 + gamma / k).ln();
        assert!(spread_half > floor);
    }

    #[test]
    fn deterministic_reruns() {
        let mut q1 = GLFTQuoter::new(0.5, 1.5, 140.0, 1_000_000_000_000, 0.001, 1_000_000_000_000);
        let mut q2 = GLFTQuoter::new(0.5, 1.5, 140.0, 1_000_000_000_000, 0.001, 1_000_000_000_000);
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
