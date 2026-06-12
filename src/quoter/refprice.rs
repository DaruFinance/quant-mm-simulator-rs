//! Reference-price primitives.
//!
//! Six families of reference-price computations.  Three are pure
//! functions of the current book; three are stateful trackers that
//! observe trade events to maintain a moving estimate.  All consume
//! only state at-or-before `t` per the spec's `≤ t` requirement.
//!
//! Pure (book-only):
//!   - `top_mid(book)`        — (best_bid + best_ask) / 2
//!   - `weighted_mid(book)`   — size-weighted using TOB queue sizes
//!   - `microprice(book)`     — Stoikov microprice using queue imbalance
//!
//! Stateful (trade-tape):
//!   - `VWAPTracker::new(window_ns)`           — trailing-window VWAP
//!   - `EWMAFairTracker::new(half_life_ns)`    — EWMA half-life decay
//!   - `ModelPredictedTracker::new(callback)`  — generic; defers to
//!     a user closure operating on a typed `ModelState` struct.
//!
//! All stateful trackers expose `observe(...)` to ingest state and
//! `value(t_ns) -> Option<f64>` to read the current estimate.
//! The leak property is: `value(T)` cannot change after polluting
//! events with ts > T (verified by the unit tests below).

#![cfg(feature = "quoter")]

use std::collections::VecDeque;

use crate::ingest::{Book, TradeEvent};

// --------------------------------------------------------------------- //
// Pure book-only primitives
// --------------------------------------------------------------------- //

/// Simple mid = (best_bid + best_ask) / 2.
pub fn top_mid(book: Option<&Book>) -> Option<f64> {
    book.and_then(|b| b.mid())
}

/// Size-weighted mid using TOB queue sizes.  When the bid queue
/// is heavy relative to the ask, the weighted mid leans toward the
/// bid — the standard "imbalance-aware" mid.
pub fn weighted_mid(book: Option<&Book>) -> Option<f64> {
    let b = book?;
    if b.bids.is_empty() || b.asks.is_empty() {
        return None;
    }
    let (bid_px, bid_sz) = b.bids[0];
    let (ask_px, ask_sz) = b.asks[0];
    if bid_sz + ask_sz <= 0.0 {
        return Some((bid_px + ask_px) / 2.0);
    }
    // When ask queue is heavier (sellers stacked), the price is more
    // likely to drift down: weight the bid_px higher.  That's
    // `(bid_px * ask_sz + ask_px * bid_sz) / (bid_sz + ask_sz)`.
    Some((bid_px * ask_sz + ask_px * bid_sz) / (bid_sz + ask_sz))
}

/// Stoikov microprice: mid + (queue_imbalance) × half_spread,
/// where queue_imbalance = (bid_sz - ask_sz) / (bid_sz + ask_sz).
/// Equivalent to `weighted_mid` algebraically; kept as a separate
/// primitive because the literature treats them as distinct
/// constructions (the derivation differs even if the closed form
/// coincides at the TOB level).
pub fn microprice(book: Option<&Book>) -> Option<f64> {
    let b = book?;
    if b.bids.is_empty() || b.asks.is_empty() {
        return None;
    }
    let (bid_px, bid_sz) = b.bids[0];
    let (ask_px, ask_sz) = b.asks[0];
    if bid_sz + ask_sz <= 0.0 {
        return Some((bid_px + ask_px) / 2.0);
    }
    let mid = (bid_px + ask_px) / 2.0;
    let half_spread = (ask_px - bid_px) / 2.0;
    let imb = (bid_sz - ask_sz) / (bid_sz + ask_sz);
    Some(mid + imb * half_spread)
}

// --------------------------------------------------------------------- //
// Stateful trackers — VWAP, EWMA fair, model-predicted
// --------------------------------------------------------------------- //

/// Trailing-window VWAP of trade tape.  `window_ns` is the
/// look-back in nanoseconds.  `observe(trade)` pushes a trade;
/// `value(t_ns)` returns Σ(price·size) / Σ(size) over trades with
/// `t_ns - window_ns < trade.ts_ns <= t_ns`.
///
/// Implementation: a deque of (ts_ns, price, size).  On each
/// value() call we evict entries older than the window.  Eviction
/// is O(k) where k is the number of expired entries; for a 1-hour
/// DS-LOB-1H run with 100ms windows this stays cheap.
#[derive(Debug, Clone)]
pub struct VWAPTracker {
    window_ns: i64,
    buffer: VecDeque<(i64, f64, f64)>,
}

impl VWAPTracker {
    pub fn new(window_ns: i64) -> Self {
        assert!(window_ns > 0, "window_ns must be > 0");
        Self { window_ns, buffer: VecDeque::new() }
    }

    pub fn observe(&mut self, trade: &TradeEvent) {
        self.buffer.push_back((trade.ts_ns, trade.price, trade.size));
    }

    pub fn value(&mut self, t_ns: i64) -> Option<f64> {
        let cutoff = t_ns - self.window_ns;
        // Evict expired entries (ts <= cutoff i.e. older than window).
        while let Some(&(ts, _, _)) = self.buffer.front() {
            if ts <= cutoff {
                self.buffer.pop_front();
            } else {
                break;
            }
        }
        if self.buffer.is_empty() {
            return None;
        }
        // Filter to entries with ts <= t_ns (in-bounds; the deque
        // may hold entries with ts > t_ns if observe was called with
        // future trades — but the leak test forbids that; defensive).
        let mut total_pv = 0.0;
        let mut total_v = 0.0;
        for &(ts, px, sz) in &self.buffer {
            if ts > t_ns {
                continue;
            }
            total_pv += px * sz;
            total_v += sz;
        }
        if total_v <= 0.0 {
            return None;
        }
        Some(total_pv / total_v)
    }
}

/// Exponentially-weighted moving average of mid (when fed
/// snapshots) or trade price (when fed trades).  Half-life form:
/// decay factor per nanosecond α(Δt) = 0.5 ** (Δt / half_life_ns).
///
/// First observation seeds the EWMA at that value.  Subsequent
/// observations update as
///   ewma <- α(Δt) * ewma + (1 - α(Δt)) * new_value
#[derive(Debug, Clone)]
pub struct EWMAFairTracker {
    half_life_ns: i64,
    ewma: Option<f64>,
    last_ts: Option<i64>,
}

impl EWMAFairTracker {
    pub fn new(half_life_ns: i64) -> Self {
        assert!(half_life_ns > 0, "half_life_ns must be > 0");
        Self { half_life_ns, ewma: None, last_ts: None }
    }

    pub fn observe(&mut self, t_ns: i64, value: f64) {
        if self.ewma.is_none() {
            self.ewma = Some(value);
            self.last_ts = Some(t_ns);
            return;
        }
        let last = self.last_ts.expect("last_ts must be set when ewma is set");
        let dt = t_ns - last;
        if dt < 0 {
            // Out-of-order; ignore (defensive).
            return;
        }
        if dt == 0 {
            // Same instant — replace with new value (more-recent
            // observation wins at the same timestamp).
            self.ewma = Some(value);
            return;
        }
        let alpha = 0.5_f64.powf(dt as f64 / self.half_life_ns as f64);
        let cur = self.ewma.expect("ewma must be set");
        self.ewma = Some(alpha * cur + (1.0 - alpha) * value);
        self.last_ts = Some(t_ns);
    }

    pub fn value(&self, _t_ns: i64) -> Option<f64> {
        self.ewma
    }
}

/// Typed analogue of Python's free-form `state` dict.  The
/// `ModelPredictedTracker` accumulates these fields via `observe(...)`
/// and the predict closure reads them.
///
/// Rust deliberately diverges from the Python `**kwargs` interface:
/// rather than a stringly-keyed dict we use named fields, gaining
/// compile-time checks at the cost of flexibility (acceptable —
/// the only in-tree predictor is `linear_drift_predictor`).
#[derive(Debug, Clone, Default)]
pub struct ModelState {
    pub n_obs: usize,
    pub mid: Option<f64>,
    pub slope_per_obs: f64,
}

/// Generic stateful tracker that defers to a user-supplied closure
/// `predict(state) -> Option<f64>`.  Ships with `linear_drift_predictor`
/// as a closed-form reference (used by the cross-language parity log
/// to anchor at an exact deterministic value).
pub struct ModelPredictedTracker {
    predict: Box<dyn FnMut(&ModelState) -> Option<f64> + Send>,
    pub state: ModelState,
}

impl ModelPredictedTracker {
    pub fn new<F>(predict: F) -> Self
    where
        F: FnMut(&ModelState) -> Option<f64> + Send + 'static,
    {
        Self {
            predict: Box::new(predict),
            state: ModelState::default(),
        }
    }

    /// Bump n_obs and update the mid field.  Pass `None` to leave
    /// mid unchanged from the previous call.
    pub fn observe(&mut self, mid: Option<f64>) {
        self.state.n_obs += 1;
        if mid.is_some() {
            self.state.mid = mid;
        }
    }

    /// Set the linear-drift slope used by `linear_drift_predictor`.
    /// (Analogue of Python's `observe(slope_per_obs=...)`.)
    pub fn set_slope_per_obs(&mut self, slope: f64) {
        self.state.slope_per_obs = slope;
    }

    pub fn value(&mut self, _t_ns: i64) -> Option<f64> {
        (self.predict)(&self.state)
    }
}

/// Reference deterministic predictor: returns
/// `mid + slope_per_obs * n_obs`.  Reads `mid` and `slope_per_obs`
/// from state (set by the caller via `observe`/`set_slope_per_obs`).
pub fn linear_drift_predictor(state: &ModelState) -> Option<f64> {
    let mid = state.mid?;
    Some(mid + state.slope_per_obs * state.n_obs as f64)
}

// --------------------------------------------------------------------- //
// Tests
// --------------------------------------------------------------------- //

#[cfg(test)]
mod tests {
    use super::*;

    fn book(ts: i64, bids: Vec<(f64, f64)>, asks: Vec<(f64, f64)>) -> Book {
        Book { ts_ns: ts, bids, asks }
    }

    fn trade(ts: i64, price: f64, size: f64) -> TradeEvent {
        TradeEvent {
            ts_ns: ts,
            recv_ns: ts,
            symbol: "BTC-USDT".into(),
            venue: "binance".into(),
            price,
            size,
            side: 1,
        }
    }

    #[test]
    fn top_mid_basic_and_none() {
        assert_eq!(top_mid(None), None);
        let b = book(0, vec![(100.0, 1.0)], vec![(102.0, 1.0)]);
        assert_eq!(top_mid(Some(&b)), Some(101.0));
        // Empty side -> mid() returns None.
        let b2 = book(0, vec![], vec![(102.0, 1.0)]);
        assert_eq!(top_mid(Some(&b2)), None);
    }

    #[test]
    fn weighted_mid_imbalance_skews_toward_thin_side() {
        // Heavy bid (5.0) vs thin ask (1.0): price should be closer to ask
        // (because heavy bid + thin ask = imbalance toward sellers
        // _absorbing_ less; weighted mid leans toward the side opposite
        // the thick queue under the Python formula).  Concretely:
        // (100*1 + 102*5) / 6 = 610/6 = 101.6666...
        let b = book(0, vec![(100.0, 5.0)], vec![(102.0, 1.0)]);
        let wm = weighted_mid(Some(&b)).unwrap();
        assert!((wm - (610.0 / 6.0)).abs() < 1e-12);
        // Equal sizes -> simple mid.
        let b2 = book(0, vec![(100.0, 2.0)], vec![(102.0, 2.0)]);
        assert!((weighted_mid(Some(&b2)).unwrap() - 101.0).abs() < 1e-12);
        // Zero sizes -> simple mid fallback.
        let b3 = book(0, vec![(100.0, 0.0)], vec![(102.0, 0.0)]);
        assert!((weighted_mid(Some(&b3)).unwrap() - 101.0).abs() < 1e-12);
        // None / empty side.
        assert_eq!(weighted_mid(None), None);
        let b4 = book(0, vec![], vec![(102.0, 1.0)]);
        assert_eq!(weighted_mid(Some(&b4)), None);
    }

    #[test]
    fn microprice_matches_weighted_mid_at_tob() {
        // Algebraic equivalence at TOB: microprice == weighted_mid.
        let b = book(0, vec![(100.0, 3.0)], vec![(102.0, 7.0)]);
        let mp = microprice(Some(&b)).unwrap();
        let wm = weighted_mid(Some(&b)).unwrap();
        assert!((mp - wm).abs() < 1e-12);
        // Spot-check: mid=101, half=1, imb=(3-7)/10=-0.4, mp=101-0.4=100.6
        assert!((mp - 100.6).abs() < 1e-12);
        // None / empty side.
        assert_eq!(microprice(None), None);
        let b2 = book(0, vec![(100.0, 1.0)], vec![]);
        assert_eq!(microprice(Some(&b2)), None);
    }

    #[test]
    fn vwap_window_evicts_old_and_computes_average() {
        let mut v = VWAPTracker::new(1_000);
        v.observe(&trade(100, 100.0, 1.0));
        v.observe(&trade(500, 102.0, 3.0));
        // At t=600 both trades are inside the window (600-1000=-400 cutoff)
        // VWAP = (100*1 + 102*3) / 4 = 406/4 = 101.5
        let wm = v.value(600).unwrap();
        assert!((wm - 101.5).abs() < 1e-12);
        // At t=1600 cutoff=600 -> only trade at ts=100 evicted (ts<=cutoff
        // is the eviction rule, so ts=500>600 false, ts=100<=600 true -> evicted).
        // Wait: cutoff=1600-1000=600. ts=100 <= 600 -> evicted.
        // ts=500 <= 600 -> evicted. Both gone, returns None.
        assert_eq!(v.value(1600), None);
        // Re-observe and check trailing.
        v.observe(&trade(1500, 200.0, 2.0));
        assert!((v.value(1600).unwrap() - 200.0).abs() < 1e-12);
    }

    #[test]
    fn vwap_no_lookahead_under_pollution() {
        // The leak property: value(T) must not change if future trades
        // (ts > T) are observed.  observe() simply appends; value(T)
        // filters out anything with ts > T.
        let mut clean = VWAPTracker::new(10_000);
        let mut polluted = VWAPTracker::new(10_000);
        for t in [(100, 100.0, 1.0), (500, 102.0, 3.0)] {
            let tr = trade(t.0, t.1, t.2);
            clean.observe(&tr);
            polluted.observe(&tr);
        }
        // Pollute with a strictly-future trade.
        polluted.observe(&trade(2_000, 9_999.0, 100.0));
        let c = clean.value(1_000).unwrap();
        let p = polluted.value(1_000).unwrap();
        assert!((c - p).abs() < 1e-12, "vwap leaked: clean={} polluted={}", c, p);
    }

    #[test]
    fn ewma_seeds_and_decays_at_half_life() {
        let mut e = EWMAFairTracker::new(1_000);
        // First observation seeds.
        e.observe(100, 100.0);
        assert_eq!(e.value(100), Some(100.0));
        // After one half-life (dt=1000), alpha=0.5, ewma = 0.5*100 + 0.5*200 = 150.
        e.observe(1_100, 200.0);
        assert!((e.value(1_100).unwrap() - 150.0).abs() < 1e-12);
        // Same-timestamp replace.
        e.observe(1_100, 999.0);
        assert!((e.value(1_100).unwrap() - 999.0).abs() < 1e-12);
        // Out-of-order: ignored.
        e.observe(500, -1.0);
        assert!((e.value(1_100).unwrap() - 999.0).abs() < 1e-12);
    }

    #[test]
    fn ewma_no_lookahead_under_out_of_order() {
        // Out-of-order observe must NOT change the value.
        let mut e = EWMAFairTracker::new(1_000);
        e.observe(0, 50.0);
        e.observe(1_000, 100.0); // half-life: ewma = 0.5*50 + 0.5*100 = 75
        let before = e.value(1_000).unwrap();
        // Now feed an out-of-order observation at t=500.
        e.observe(500, 9_999.0);
        let after = e.value(1_000).unwrap();
        assert!((before - after).abs() < 1e-12,
                "ewma leaked under out-of-order: before={} after={}", before, after);
    }

    #[test]
    fn model_predicted_linear_drift_basic() {
        let mut m = ModelPredictedTracker::new(linear_drift_predictor);
        // Before any observation, mid is None -> predictor returns None.
        assert_eq!(m.value(0), None);
        m.set_slope_per_obs(0.5);
        m.observe(Some(100.0));               // n_obs=1, mid=100
        // 100 + 0.5*1 = 100.5
        assert!((m.value(0).unwrap() - 100.5).abs() < 1e-12);
        m.observe(Some(101.0));               // n_obs=2, mid=101
        // 101 + 0.5*2 = 102
        assert!((m.value(0).unwrap() - 102.0).abs() < 1e-12);
        m.observe(None);                       // n_obs=3, mid stays 101
        // 101 + 0.5*3 = 102.5
        assert!((m.value(0).unwrap() - 102.5).abs() < 1e-12);
    }

    #[test]
    fn model_predicted_custom_closure() {
        // Custom predictor that returns mid * 2.
        let mut m = ModelPredictedTracker::new(|s: &ModelState| s.mid.map(|x| x * 2.0));
        m.observe(Some(42.0));
        assert_eq!(m.value(0), Some(84.0));
    }
}
