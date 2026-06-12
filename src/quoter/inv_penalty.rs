//! Inventory-penalty primitives.
//!
//! Mirror of Python's `mmsim.quoter.inv_penalty`.  Six pure functions
//! of `(inv, params)` that compute a price skew (in the same currency
//! unit as `ref_price`) and an optional size scale applied to bid vs
//! ask quotes.  Output is a `Skew` record with three scalar fields:
//!
//!   - `price_offset`: signed currency offset; **positive shifts both
//!     quotes UP** (skewing in favour of selling — discourages buying).
//!     Bid price becomes `ref - half_spread + price_offset`;
//!     ask price becomes `ref + half_spread + price_offset`.
//!   - `size_scale_bid` / `size_scale_ask`: multiplicative factors in
//!     `[0.0, 1.0]` applied to each side's quote sizes.  1.0 = no
//!     change; 0.0 = refuse to quote that side (hard-cap path).
//!
//! All primitives are pure functions of `inv` and the spec — no time,
//! no book, no future state.  Leak-freedom is the signature itself.

#![cfg(feature = "quoter")]

/// Output of every inventory-penalty primitive.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Skew {
    /// Signed currency offset; `+ve` shifts both quotes UP.
    pub price_offset: f64,
    /// Multiplier in `[0.0, 1.0]` for the bid-side quote size.
    pub size_scale_bid: f64,
    /// Multiplier in `[0.0, 1.0]` for the ask-side quote size.
    pub size_scale_ask: f64,
}

// --------------------------------------------------------------------- //
// Primitives
// --------------------------------------------------------------------- //

/// Price skew = `-gamma * inv` (long inv -> shift quotes DOWN to
/// encourage selling).  No size skew.
pub fn linear(inv: f64, gamma: f64) -> Skew {
    Skew {
        price_offset: -gamma * inv,
        size_scale_bid: 1.0,
        size_scale_ask: 1.0,
    }
}

/// Quadratic price skew: `-sign(inv) * gamma * inv^2`.  Penalty grows
/// non-linearly with position size.
pub fn quadratic(inv: f64, gamma: f64) -> Skew {
    let sign = if inv > 0.0 {
        -1.0
    } else if inv < 0.0 {
        1.0
    } else {
        0.0
    };
    Skew {
        price_offset: sign * gamma * inv * inv,
        size_scale_bid: 1.0,
        size_scale_ask: 1.0,
    }
}

/// Exponential price skew: `sign(-inv) * gamma * (exp(|inv|/scale) - 1)`.
/// Skew explodes as `|inv|` grows; useful for hard-skew at large
/// positions while staying small near zero.
pub fn exponential(inv: f64, gamma: f64, scale: f64) -> Skew {
    let abs_inv = inv.abs();
    let base = (abs_inv / scale).exp() - 1.0;
    let sign = if inv > 0.0 {
        -1.0
    } else if inv < 0.0 {
        1.0
    } else {
        0.0
    };
    Skew {
        price_offset: sign * gamma * base,
        size_scale_bid: 1.0,
        size_scale_ask: 1.0,
    }
}

/// Different linear coefficient for long vs short inventory.  Useful
/// when one side is structurally harder to hedge.
pub fn asymmetric(inv: f64, gamma_long: f64, gamma_short: f64) -> Skew {
    let offset = if inv > 0.0 {
        -gamma_long * inv
    } else if inv < 0.0 {
        -gamma_short * inv // inv<0 -> offset>0 -> shifts UP
    } else {
        0.0
    };
    Skew {
        price_offset: offset,
        size_scale_bid: 1.0,
        size_scale_ask: 1.0,
    }
}

/// Linear skew plus an additional quadratic ramp as `|inv|` approaches
/// `cap`.  Smoothly slows accumulation without an abrupt cliff.
///
/// Panics if `cap <= 0`.
pub fn soft_cap(inv: f64, gamma: f64, cap: f64) -> Skew {
    if cap <= 0.0 {
        panic!("cap must be > 0");
    }
    let base = -gamma * inv;
    let abs_inv = inv.abs();
    let offset = if abs_inv >= cap {
        // Quadratic ramp: extra penalty proportional to (|inv|/cap - 1)^2
        let ratio = abs_inv / cap - 1.0;
        let extra = gamma * cap * ratio * ratio;
        let sign_extra = if inv > 0.0 { -1.0 } else { 1.0 };
        base + sign_extra * extra
    } else {
        base
    };
    Skew {
        price_offset: offset,
        size_scale_bid: 1.0,
        size_scale_ask: 1.0,
    }
}

/// No price skew; instead, refuses to quote on the side that would
/// worsen position.  `inv >= cap`: drop the bid (`size_bid=0`);
/// `inv <= -cap`: drop the ask (`size_ask=0`).
///
/// Panics if `cap <= 0`.
pub fn hard_cap(inv: f64, cap: f64) -> Skew {
    if cap <= 0.0 {
        panic!("cap must be > 0");
    }
    let bid_scale = if inv >= cap { 0.0 } else { 1.0 };
    let ask_scale = if inv <= -cap { 0.0 } else { 1.0 };
    Skew {
        price_offset: 0.0,
        size_scale_bid: bid_scale,
        size_scale_ask: ask_scale,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-12;

    #[test]
    fn linear_long_shifts_down_short_shifts_up() {
        let s = linear(2.0, 0.5);
        assert!((s.price_offset - (-1.0)).abs() < EPS);
        assert_eq!(s.size_scale_bid, 1.0);
        assert_eq!(s.size_scale_ask, 1.0);

        let s = linear(-3.0, 0.5);
        assert!((s.price_offset - 1.5).abs() < EPS);
    }

    #[test]
    fn linear_zero_inv_no_offset() {
        let s = linear(0.0, 1.0);
        assert_eq!(s.price_offset, 0.0);
        assert_eq!(s.size_scale_bid, 1.0);
        assert_eq!(s.size_scale_ask, 1.0);
    }

    #[test]
    fn quadratic_grows_with_square_of_inv() {
        // inv=2, gamma=0.5 -> sign=-1, offset = -1 * 0.5 * 4 = -2.0
        let s = quadratic(2.0, 0.5);
        assert!((s.price_offset - (-2.0)).abs() < EPS);
        // inv=-3, gamma=0.5 -> sign=+1, offset = +1 * 0.5 * 9 = +4.5
        let s = quadratic(-3.0, 0.5);
        assert!((s.price_offset - 4.5).abs() < EPS);
        // inv=0 -> 0
        let s = quadratic(0.0, 10.0);
        assert_eq!(s.price_offset, 0.0);
    }

    #[test]
    fn exponential_zero_inv_zero_offset() {
        let s = exponential(0.0, 1.0, 5.0);
        assert_eq!(s.price_offset, 0.0);
        assert_eq!(s.size_scale_bid, 1.0);
        assert_eq!(s.size_scale_ask, 1.0);
    }

    #[test]
    fn exponential_signs_and_magnitude() {
        // inv=5, gamma=2.0, scale=5.0 -> sign=-1, base = e^1 - 1
        let s = exponential(5.0, 2.0, 5.0);
        let expected = -1.0 * 2.0 * ((1.0_f64).exp() - 1.0);
        assert!((s.price_offset - expected).abs() < EPS);

        // negative inv -> positive offset (shifts UP)
        let s = exponential(-5.0, 2.0, 5.0);
        let expected = 2.0 * ((1.0_f64).exp() - 1.0);
        assert!((s.price_offset - expected).abs() < EPS);
    }

    #[test]
    fn asymmetric_uses_separate_coefficients() {
        // inv>0 -> uses gamma_long
        let s = asymmetric(2.0, 0.5, 1.5);
        assert!((s.price_offset - (-1.0)).abs() < EPS);
        // inv<0 -> uses gamma_short, offset = -gamma_short * inv = +3.0
        let s = asymmetric(-2.0, 0.5, 1.5);
        assert!((s.price_offset - 3.0).abs() < EPS);
        // inv=0 -> 0
        let s = asymmetric(0.0, 0.5, 1.5);
        assert_eq!(s.price_offset, 0.0);
    }

    #[test]
    fn soft_cap_below_cap_is_just_linear() {
        // |inv| < cap -> only base linear skew
        let s = soft_cap(1.0, 0.5, 5.0);
        assert!((s.price_offset - (-0.5)).abs() < EPS);
    }

    #[test]
    fn soft_cap_above_cap_adds_quadratic_ramp() {
        // inv=10, gamma=0.5, cap=5 -> base = -5.0
        // ratio = 10/5 - 1 = 1; extra = 0.5 * 5 * 1 = 2.5
        // sign_extra=-1 -> offset = -5.0 - 2.5 = -7.5
        let s = soft_cap(10.0, 0.5, 5.0);
        assert!((s.price_offset - (-7.5)).abs() < EPS);

        // negative inv side: inv=-10, gamma=0.5, cap=5 -> base=+5.0
        // extra=2.5, sign_extra=+1 -> offset=+7.5
        let s = soft_cap(-10.0, 0.5, 5.0);
        assert!((s.price_offset - 7.5).abs() < EPS);
    }

    #[test]
    fn soft_cap_at_cap_boundary_includes_ramp() {
        // |inv| == cap: ratio=0, extra=0, offset == base
        let s = soft_cap(5.0, 0.5, 5.0);
        assert!((s.price_offset - (-2.5)).abs() < EPS);
    }

    #[test]
    #[should_panic(expected = "cap must be > 0")]
    fn soft_cap_panics_on_nonpositive_cap() {
        let _ = soft_cap(1.0, 0.5, 0.0);
    }

    #[test]
    fn hard_cap_drops_bid_when_long_at_cap() {
        let s = hard_cap(5.0, 5.0);
        assert_eq!(s.price_offset, 0.0);
        assert_eq!(s.size_scale_bid, 0.0);
        assert_eq!(s.size_scale_ask, 1.0);
    }

    #[test]
    fn hard_cap_drops_ask_when_short_at_cap() {
        let s = hard_cap(-5.0, 5.0);
        assert_eq!(s.price_offset, 0.0);
        assert_eq!(s.size_scale_bid, 1.0);
        assert_eq!(s.size_scale_ask, 0.0);
    }

    #[test]
    fn hard_cap_inside_cap_no_change() {
        let s = hard_cap(0.0, 5.0);
        assert_eq!(s.size_scale_bid, 1.0);
        assert_eq!(s.size_scale_ask, 1.0);
        let s = hard_cap(4.99, 5.0);
        assert_eq!(s.size_scale_bid, 1.0);
        assert_eq!(s.size_scale_ask, 1.0);
    }

    #[test]
    #[should_panic(expected = "cap must be > 0")]
    fn hard_cap_panics_on_nonpositive_cap() {
        let _ = hard_cap(0.0, -1.0);
    }
}
