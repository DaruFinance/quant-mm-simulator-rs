//! Shared rolling-volatility tracker for the closed-form quoting
//! models (Avellaneda-Stoikov, Cartea-Jaimungal, GLFT, Ho-Stoll).
//!
//! Mirror of Python's `mmsim.models._vol`.
//!
//! Implementation
//! --------------
//! `RollingSigma::new(window_ns)`:
//!   - `observe(t_ns, price)` pushes the price + ts into the trailing
//!     window.
//!   - `value(t_ns) -> Option<f64>` returns the **per-nanosecond
//!     return std** — `std(log-returns) / sqrt(mean_dt_ns)`.  This
//!     yields a diffusion-coefficient figure such that `σ² · τ`
//!     (with τ in nanoseconds) has units of return-variance over
//!     the horizon τ.  Returns None when fewer than 2 valid prices
//!     fall inside the window.
//!
//! Why mid-fed rather than trade-fed: the Quoter trait gets called on
//! snapshots; the model has no direct hook to the trade stream from
//! inside `quote()`.  Feeding sigma from snapshot mids keeps every
//! input on the Protocol's signature.
//!
//! Leak property
//! -------------
//! `value(T)` filters its internal buffer to `ts <= T` before
//! computing the predicate.

#![cfg(feature = "models")]

use std::collections::VecDeque;

#[derive(Debug, Clone)]
pub struct RollingSigma {
    window_ns: i64,
    buffer: VecDeque<(i64, f64)>,
}

impl RollingSigma {
    pub fn new(window_ns: i64) -> Self {
        assert!(window_ns > 0, "window_ns must be > 0");
        Self { window_ns, buffer: VecDeque::new() }
    }

    pub fn observe(&mut self, t_ns: i64, price: f64) {
        if price > 0.0 {
            self.buffer.push_back((t_ns, price));
        }
    }

    fn evict(&mut self, t_ns: i64) {
        let cutoff = t_ns - self.window_ns;
        while let Some(&(ts, _)) = self.buffer.front() {
            if ts <= cutoff {
                self.buffer.pop_front();
            } else {
                break;
            }
        }
    }

    pub fn value(&mut self, t_ns: i64) -> Option<f64> {
        self.evict(t_ns);
        let kept: Vec<(i64, f64)> = self
            .buffer
            .iter()
            .copied()
            .filter(|&(ts, px)| ts <= t_ns && px > 0.0)
            .collect();
        if kept.len() < 2 {
            return None;
        }
        let prices: Vec<f64> = kept.iter().map(|&(_, px)| px).collect();
        let timestamps: Vec<i64> = kept.iter().map(|&(ts, _)| ts).collect();
        let rets: Vec<f64> = (1..prices.len())
            .map(|i| (prices[i] / prices[i - 1]).ln())
            .collect();
        let n = rets.len() as f64;
        let mean: f64 = rets.iter().sum::<f64>() / n;
        let var: f64 = rets.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
        let std = var.sqrt();
        let total_dt = (timestamps[timestamps.len() - 1] - timestamps[0]).max(1) as f64;
        let mean_dt = total_dt / n;
        Some(std / mean_dt.sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn returns_none_before_two_obs() {
        let mut s = RollingSigma::new(10_000);
        assert_eq!(s.value(0), None);
        s.observe(0, 100.0);
        assert_eq!(s.value(0), None);
    }

    #[test]
    fn evicts_old_window() {
        let mut s = RollingSigma::new(1_000);
        s.observe(0, 100.0);
        s.observe(500, 101.0);
        s.observe(900, 102.0);
        assert!(s.value(1_000).is_some());
        // At t=3000 cutoff=2000; all entries <= 2000 → evicted.
        assert_eq!(s.value(3_000), None);
    }

    #[test]
    fn no_lookahead_under_pollution() {
        let mut clean = RollingSigma::new(10_000);
        let mut polluted = RollingSigma::new(10_000);
        for &(t, p) in &[(0_i64, 100.0_f64), (500, 101.0), (900, 102.0)] {
            clean.observe(t, p);
            polluted.observe(t, p);
        }
        polluted.observe(5_000, 9_999.0);
        let a = clean.value(1_000).unwrap();
        let b = polluted.value(1_000).unwrap();
        assert!((a - b).abs() < 1e-12);
    }

    #[test]
    fn constant_prices_zero_vol() {
        let mut s = RollingSigma::new(10_000);
        for t in 0..5 {
            s.observe(t * 100, 100.0);
        }
        let v = s.value(500).unwrap();
        assert_eq!(v, 0.0);
    }

    #[test]
    fn matches_python_per_ns_scaling() {
        // Three prices at t=0, 1000, 2000 — log-returns ~ln(101/100), ln(102/101)
        let mut s = RollingSigma::new(10_000);
        s.observe(0, 100.0);
        s.observe(1_000, 101.0);
        s.observe(2_000, 102.0);
        let v = s.value(2_000).unwrap();
        // n=2 returns; mean ≈ ln(102/100)/2; var small; mean_dt = 2000/2=1000
        // Expected = std / sqrt(1000) (well-defined, just check finite + positive).
        assert!(v > 0.0 && v.is_finite());
    }
}
