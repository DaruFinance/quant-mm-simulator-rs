//! Quote-shape primitives.
//!
//! Mirror of Python's `mmsim.quoter.shapes`.  Five pure-function
//! shape primitives, each `(spec, ref_price, inv) -> Vec<QuoteRequest>`.
//! They never consult time, never consult future state, and never
//! close over mutable book state — leak-freedom by signature.
//!
//! The five shapes:
//!   - `single`        — one bid + one ask at ref +/- half_spread
//!   - `paired`        — N bid + N ask with per-level (offset, size)
//!   - `ladder`        — N linearly-spaced levels per side
//!   - `geometric`     — N geometrically-spaced levels per side
//!   - `dynamic_depth` — N tapers down as |inv| grows past threshold

#![cfg(feature = "quoter")]

use crate::sim::sim_loop::QuoteRequest;

// --------------------------------------------------------------------- //
// Spec types
// --------------------------------------------------------------------- //

/// One bid + one ask of equal size at `ref +/- half_spread`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SingleSpec {
    pub size: f64,
    pub half_spread: f64,
}

/// N levels per side with per-level `(offset_from_ref, size)`.
///
/// `levels_bid` and `levels_ask` are each a vec of `(offset, size)`
/// where offset is the absolute distance from ref (positive; bids
/// sit at `ref - offset`, asks at `ref + offset`).  Sizes per level
/// can differ across sides; the two arms are independent.
#[derive(Debug, Clone, PartialEq)]
pub struct PairedSpec {
    pub levels_bid: Vec<(f64, f64)>,
    pub levels_ask: Vec<(f64, f64)>,
}

/// N linear-spaced levels per side.  Level k sits at
/// `ref +/- (half_spread + k * step)` with size `size_per_level`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LadderSpec {
    pub half_spread: f64,
    pub step: f64,
    pub n_levels: usize,
    pub size_per_level: f64,
}

/// N geometrically-spaced levels per side.  Level k sits at
/// `ref +/- half_spread * ratio^k` with size `size_per_level`.
/// `ratio > 1` means expanding spacing; `ratio == 1` collapses to a
/// degenerate stack at the same price.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GeometricSpec {
    pub half_spread: f64,
    pub ratio: f64,
    pub n_levels: usize,
    pub size_per_level: f64,
}

/// Like `LadderSpec` but the active level count tapers when
/// `|inv|` exceeds `inv_taper_threshold`.  At `|inv| <= thresh`
/// the shape posts `max_levels` per side; at `|inv| >= 2 * thresh`
/// it tapers down to 1 level; linear in between.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DynamicDepthSpec {
    pub half_spread: f64,
    pub step: f64,
    pub max_levels: usize,
    pub inv_taper_threshold: f64,
    pub size_per_level: f64,
}

// --------------------------------------------------------------------- //
// Primitives
// --------------------------------------------------------------------- //

/// One bid + one ask centred on `ref_price`.
pub fn single(spec: &SingleSpec, ref_price: f64, _inv: f64) -> Vec<QuoteRequest> {
    vec![
        QuoteRequest { side: 1, price: ref_price - spec.half_spread, size: spec.size },
        QuoteRequest { side: -1, price: ref_price + spec.half_spread, size: spec.size },
    ]
}

/// Per-level (offset, size) on each side, independent arms.
pub fn paired(spec: &PairedSpec, ref_price: f64, _inv: f64) -> Vec<QuoteRequest> {
    let mut out: Vec<QuoteRequest> =
        Vec::with_capacity(spec.levels_bid.len() + spec.levels_ask.len());
    for &(offset, size) in &spec.levels_bid {
        out.push(QuoteRequest { side: 1, price: ref_price - offset, size });
    }
    for &(offset, size) in &spec.levels_ask {
        out.push(QuoteRequest { side: -1, price: ref_price + offset, size });
    }
    out
}

/// N linear-spaced levels per side, bid + ask interleaved by level.
pub fn ladder(spec: &LadderSpec, ref_price: f64, _inv: f64) -> Vec<QuoteRequest> {
    if spec.n_levels < 1 {
        return Vec::new();
    }
    let mut out: Vec<QuoteRequest> = Vec::with_capacity(spec.n_levels * 2);
    for k in 0..spec.n_levels {
        let offset = spec.half_spread + (k as f64) * spec.step;
        out.push(QuoteRequest {
            side: 1,
            price: ref_price - offset,
            size: spec.size_per_level,
        });
        out.push(QuoteRequest {
            side: -1,
            price: ref_price + offset,
            size: spec.size_per_level,
        });
    }
    out
}

/// N geometrically-spaced levels per side.
pub fn geometric(spec: &GeometricSpec, ref_price: f64, _inv: f64) -> Vec<QuoteRequest> {
    if spec.n_levels < 1 {
        return Vec::new();
    }
    let mut out: Vec<QuoteRequest> = Vec::with_capacity(spec.n_levels * 2);
    let mut offset = spec.half_spread;
    for _ in 0..spec.n_levels {
        out.push(QuoteRequest {
            side: 1,
            price: ref_price - offset,
            size: spec.size_per_level,
        });
        out.push(QuoteRequest {
            side: -1,
            price: ref_price + offset,
            size: spec.size_per_level,
        });
        offset *= spec.ratio;
    }
    out
}

/// Ladder whose active depth tapers with `|inv|`.
pub fn dynamic_depth(
    spec: &DynamicDepthSpec,
    ref_price: f64,
    inv: f64,
) -> Vec<QuoteRequest> {
    let abs_inv = inv.abs();
    let thr = spec.inv_taper_threshold;
    let n_active: usize = if abs_inv <= thr {
        spec.max_levels
    } else if abs_inv >= 2.0 * thr {
        1
    } else {
        // Linear taper from max_levels down to 1 across [thr, 2*thr].
        let frac = (abs_inv - thr) / thr; // in [0, 1]
        let scaled = (spec.max_levels as f64) * (1.0 - frac);
        // Python: max(1, int(round(scaled))).  Use banker's-rounding-free
        // round-half-away-from-zero to match Python's `round()` on
        // half-integer corners we don't expect to hit in practice; for
        // non-half values the behaviour is identical.
        let rounded = scaled.round() as i64;
        if rounded < 1 { 1 } else { rounded as usize }
    };
    let mut out: Vec<QuoteRequest> = Vec::with_capacity(n_active * 2);
    for k in 0..n_active {
        let offset = spec.half_spread + (k as f64) * spec.step;
        out.push(QuoteRequest {
            side: 1,
            price: ref_price - offset,
            size: spec.size_per_level,
        });
        out.push(QuoteRequest {
            side: -1,
            price: ref_price + offset,
            size: spec.size_per_level,
        });
    }
    out
}

// --------------------------------------------------------------------- //
// Tests
// --------------------------------------------------------------------- //

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-12;

    #[test]
    fn single_emits_bid_and_ask_at_half_spread() {
        let spec = SingleSpec { size: 0.5, half_spread: 0.25 };
        let out = single(&spec, 100.0, 0.0);
        assert_eq!(out.len(), 2);
        // Bid at ref - half_spread.
        assert_eq!(out[0].side, 1);
        assert!((out[0].price - 99.75).abs() < EPS);
        assert!((out[0].size - 0.5).abs() < EPS);
        // Ask at ref + half_spread.
        assert_eq!(out[1].side, -1);
        assert!((out[1].price - 100.25).abs() < EPS);
        assert!((out[1].size - 0.5).abs() < EPS);
    }

    #[test]
    fn paired_independent_arms_with_per_level_sizes() {
        let spec = PairedSpec {
            levels_bid: vec![(0.10, 1.0), (0.30, 2.0)],
            levels_ask: vec![(0.20, 3.0)],
        };
        let out = paired(&spec, 50.0, 0.0);
        assert_eq!(out.len(), 3);
        // Two bids first.
        assert_eq!(out[0].side, 1);
        assert!((out[0].price - 49.90).abs() < EPS);
        assert!((out[0].size - 1.0).abs() < EPS);
        assert_eq!(out[1].side, 1);
        assert!((out[1].price - 49.70).abs() < EPS);
        assert!((out[1].size - 2.0).abs() < EPS);
        // Then one ask.
        assert_eq!(out[2].side, -1);
        assert!((out[2].price - 50.20).abs() < EPS);
        assert!((out[2].size - 3.0).abs() < EPS);
    }

    #[test]
    fn ladder_three_levels_linear_spacing() {
        let spec = LadderSpec {
            half_spread: 0.5,
            step: 0.25,
            n_levels: 3,
            size_per_level: 1.5,
        };
        let out = ladder(&spec, 10.0, 0.0);
        // 3 levels x 2 sides = 6 requests.
        assert_eq!(out.len(), 6);
        // k=0: offset 0.50  -> bid 9.50, ask 10.50
        // k=1: offset 0.75  -> bid 9.25, ask 10.75
        // k=2: offset 1.00  -> bid 9.00, ask 11.00
        let expected_bids = [9.50, 9.25, 9.00];
        let expected_asks = [10.50, 10.75, 11.00];
        for k in 0..3 {
            let bid = &out[2 * k];
            let ask = &out[2 * k + 1];
            assert_eq!(bid.side, 1);
            assert_eq!(ask.side, -1);
            assert!((bid.price - expected_bids[k]).abs() < EPS);
            assert!((ask.price - expected_asks[k]).abs() < EPS);
            assert!((bid.size - 1.5).abs() < EPS);
            assert!((ask.size - 1.5).abs() < EPS);
        }
    }

    #[test]
    fn geometric_three_levels_ratio_two() {
        let spec = GeometricSpec {
            half_spread: 0.10,
            ratio: 2.0,
            n_levels: 3,
            size_per_level: 0.25,
        };
        let out = geometric(&spec, 100.0, 0.0);
        assert_eq!(out.len(), 6);
        // k=0: offset 0.10  -> bid 99.90,  ask 100.10
        // k=1: offset 0.20  -> bid 99.80,  ask 100.20
        // k=2: offset 0.40  -> bid 99.60,  ask 100.40
        let expected_bids = [99.90, 99.80, 99.60];
        let expected_asks = [100.10, 100.20, 100.40];
        for k in 0..3 {
            let bid = &out[2 * k];
            let ask = &out[2 * k + 1];
            assert_eq!(bid.side, 1);
            assert_eq!(ask.side, -1);
            assert!((bid.price - expected_bids[k]).abs() < EPS);
            assert!((ask.price - expected_asks[k]).abs() < EPS);
            assert!((bid.size - 0.25).abs() < EPS);
            assert!((ask.size - 0.25).abs() < EPS);
        }
    }

    #[test]
    fn dynamic_depth_tapers_with_inventory() {
        let spec = DynamicDepthSpec {
            half_spread: 0.10,
            step: 0.05,
            max_levels: 4,
            inv_taper_threshold: 1.0,
            size_per_level: 0.5,
        };
        // |inv| <= thresh -> full depth (4 levels per side = 8 requests).
        let out = dynamic_depth(&spec, 100.0, 0.5);
        assert_eq!(out.len(), 8);
        // |inv| >= 2*thresh -> collapse to 1 level per side = 2 requests.
        let out = dynamic_depth(&spec, 100.0, 2.5);
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].side, 1);
        assert!((out[0].price - 99.90).abs() < EPS);
        assert_eq!(out[1].side, -1);
        assert!((out[1].price - 100.10).abs() < EPS);
        // Linear taper midpoint: |inv| = 1.5 -> frac = 0.5 ->
        // n = round(4 * 0.5) = 2.
        let out = dynamic_depth(&spec, 100.0, 1.5);
        assert_eq!(out.len(), 4);
        // Negative inventory hits the |.| branch identically.
        let out_neg = dynamic_depth(&spec, 100.0, -1.5);
        assert_eq!(out_neg.len(), 4);
    }
}
