//! T4 Market-Making structural combo grid.
//!
//! Mirror of Python's `mmsim.t4_corpus.combos`. 7 axes × cardinalities
//! `{8, 6, 7, 4, 6, 5, 5}` = 201,600 structural combos.
//!
//! Sampler is deterministic-by-seed but does NOT match Python's
//! `numpy.random.default_rng.choice(replace=False)` ordering — see
//! the divergence note in `src/t4_corpus/mod.rs`.

#![cfg(feature = "t4-corpus")]

use rand::{Rng, SeedableRng};
use rand_pcg::Pcg64;

pub const QUOTING_MODELS: [&str; 8] = [
    "avellaneda_stoikov", "cartea_jaimungal", "glft", "ho_stoll",
    "symmetric", "ladder", "microprice_skew", "fair_anchored",
];

pub const INVENTORY_PENALTIES: [&str; 6] = [
    "linear", "quadratic", "exponential",
    "asymmetric", "soft_cap", "hard_cap",
];

pub const ADVERSE_FILTERS: [&str; 7] = [
    "none", "ofi", "toxicity", "vol_surge",
    "microprice_dev", "queue_imb", "hybrid",
];

pub const HEDGE_MODES: [&str; 4] = [
    "none", "perp", "basket", "options_vega_stub",
];

pub const REFERENCE_PRICES: [&str; 6] = [
    "mid", "microprice", "weighted_mid",
    "ewma_fair", "model_pred", "vwap",
];

pub const QUOTE_SHAPES: [&str; 5] = [
    "single", "paired", "ladder", "geometric", "dynamic_depth",
];

pub const REFRESH_TRIGGERS: [&str; 5] = [
    "time", "mid_move", "inv_change", "book_event", "hybrid",
];

/// Total structural cardinality: 8 * 6 * 7 * 4 * 6 * 5 * 5 = 201,600.
pub const GRID_SIZE: usize =
    QUOTING_MODELS.len() * INVENTORY_PENALTIES.len()
    * ADVERSE_FILTERS.len() * HEDGE_MODES.len()
    * REFERENCE_PRICES.len() * QUOTE_SHAPES.len()
    * REFRESH_TRIGGERS.len();

/// One structural T4 combo. Mirrors Python's frozen `Combo` dataclass.
///
/// Stores `&'static str` references to the axis-constant arrays for
/// memory efficiency and exact-equality semantics.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Combo {
    pub quoting_model: &'static str,
    pub inventory_penalty: &'static str,
    pub adverse_filter: &'static str,
    pub hedge_mode: &'static str,
    pub reference_price: &'static str,
    pub quote_shape: &'static str,
    pub refresh_trigger: &'static str,
}

const AXIS_SIZES: [usize; 7] = [
    QUOTING_MODELS.len(),
    INVENTORY_PENALTIES.len(),
    ADVERSE_FILTERS.len(),
    HEDGE_MODES.len(),
    REFERENCE_PRICES.len(),
    QUOTE_SHAPES.len(),
    REFRESH_TRIGGERS.len(),
];

fn token_for(axis: usize, idx: usize) -> &'static str {
    match axis {
        0 => QUOTING_MODELS[idx],
        1 => INVENTORY_PENALTIES[idx],
        2 => ADVERSE_FILTERS[idx],
        3 => HEDGE_MODES[idx],
        4 => REFERENCE_PRICES[idx],
        5 => QUOTE_SHAPES[idx],
        6 => REFRESH_TRIGGERS[idx],
        _ => unreachable!("axis index out of range"),
    }
}

/// Materialize the combo at linear index `idx` without enumerating the
/// whole grid. Uses mixed-radix decomposition matching Python's
/// `itertools.product` ordering (last axis varies fastest).
pub fn combo_at(idx: usize) -> Combo {
    assert!(idx < GRID_SIZE, "combo index {} out of range [0, {})", idx, GRID_SIZE);
    // Strides matching slow-to-fast layout (refresh_trigger fastest).
    let mut strides = [1_usize; 7];
    for i in (0..6).rev() {
        strides[i] = strides[i + 1] * AXIS_SIZES[i + 1];
    }
    let mut remaining = idx;
    let mut parts = [0_usize; 7];
    for i in 0..7 {
        parts[i] = remaining / strides[i];
        remaining %= strides[i];
    }
    Combo {
        quoting_model: token_for(0, parts[0]),
        inventory_penalty: token_for(1, parts[1]),
        adverse_filter: token_for(2, parts[2]),
        hedge_mode: token_for(3, parts[3]),
        reference_price: token_for(4, parts[4]),
        quote_shape: token_for(5, parts[5]),
        refresh_trigger: token_for(6, parts[6]),
    }
}

pub fn iter_all_combos() -> impl Iterator<Item = Combo> {
    (0..GRID_SIZE).map(combo_at)
}

/// Deterministic unique sample of `n` combos using a Pcg64 seeded RNG.
///
/// Divergence note (parity): the underlying PRNG transforms differ between
/// numpy's PCG64 and `rand_pcg::Pcg64`, so the LISTS do not match Python
/// bit-for-bit. The shape contract holds: deterministic given the seed,
/// every drawn index lies in `[0, GRID_SIZE)`, no duplicates.
pub fn sample_combos(n: usize, seed: u64) -> Vec<Combo> {
    assert!(n <= GRID_SIZE, "requested n={} exceeds grid size {}", n, GRID_SIZE);
    let mut idx: Vec<usize> = (0..GRID_SIZE).collect();
    let mut rng = Pcg64::seed_from_u64(seed);
    // Fisher-Yates partial shuffle: for i in 0..n, swap idx[i] with a
    // uniform random element in idx[i..]. After n swaps, idx[..n] is
    // a uniform sample without replacement.
    for i in 0..n {
        let j = rng.random_range(i..GRID_SIZE);
        idx.swap(i, j);
    }
    idx[..n].iter().map(|&i| combo_at(i)).collect()
}

/// Per-axis short-code resolver — disambiguates tokens shared across
/// axes (`none`, `hybrid`, `ladder`) by axis position so a combo's
/// strategy name has 7 underscore-separated parts uniquely keyed by
/// axis.
fn axis_short_code(axis_name: &str, token: &str) -> &'static str {
    match (axis_name, token) {
        ("adverse_filter", "none") => "nofilt",
        ("adverse_filter", "hybrid") => "hybA",
        ("hedge_mode", "none") => "nohed",
        ("refresh_trigger", "hybrid") => "hybR",
        ("quote_shape", "ladder") => "ladS",
        _ => match token {
            // quoting_model
            "avellaneda_stoikov" => "AS",
            "cartea_jaimungal" => "CJ",
            "glft" => "GLFT",
            "ho_stoll" => "HS",
            "symmetric" => "sym",
            "ladder" => "lad",
            "microprice_skew" => "mps",
            "fair_anchored" => "fair",
            // inventory_penalty
            "linear" => "lin",
            "quadratic" => "quad",
            "exponential" => "exp",
            "asymmetric" => "asym",
            "soft_cap" => "scap",
            "hard_cap" => "hcap",
            // adverse_filter (none/hybrid handled above)
            "ofi" => "ofi",
            "toxicity" => "tox",
            "vol_surge" => "vsurge",
            "microprice_dev" => "mpdev",
            "queue_imb" => "qimb",
            // hedge_mode (none handled above)
            "perp" => "perp",
            "basket" => "bask",
            "options_vega_stub" => "opvg",
            // reference_price
            "mid" => "mid",
            "microprice" => "micro",
            "weighted_mid" => "wmid",
            "ewma_fair" => "ewma",
            "model_pred" => "mpred",
            "vwap" => "vwap",
            // quote_shape (ladder handled above)
            "single" => "sgl",
            "paired" => "par",
            "geometric" => "geo",
            "dynamic_depth" => "dyn",
            // refresh_trigger (hybrid handled above)
            "time" => "time",
            "mid_move" => "mvmid",
            "inv_change" => "invch",
            "book_event" => "bkev",
            other => panic!("unknown short code for token: {} (axis {})", other, axis_name),
        },
    }
}

pub fn combo_to_strategy_name(c: &Combo) -> String {
    let parts: [&str; 7] = [
        axis_short_code("quoting_model", c.quoting_model),
        axis_short_code("inventory_penalty", c.inventory_penalty),
        axis_short_code("adverse_filter", c.adverse_filter),
        axis_short_code("hedge_mode", c.hedge_mode),
        axis_short_code("reference_price", c.reference_price),
        axis_short_code("quote_shape", c.quote_shape),
        axis_short_code("refresh_trigger", c.refresh_trigger),
    ];
    parts.join("_")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn grid_size_is_201600() {
        assert_eq!(GRID_SIZE, 201_600);
    }

    #[test]
    fn combo_at_first_and_last() {
        let c0 = combo_at(0);
        assert_eq!(c0.quoting_model, "avellaneda_stoikov");
        assert_eq!(c0.refresh_trigger, "time");
        let c_last = combo_at(GRID_SIZE - 1);
        assert_eq!(c_last.quoting_model, "fair_anchored");
        assert_eq!(c_last.refresh_trigger, "hybrid");
    }

    #[test]
    fn iter_count_matches_grid_size() {
        let count: usize = iter_all_combos().count();
        assert_eq!(count, GRID_SIZE);
    }

    #[test]
    fn sample_deterministic_and_dedup() {
        let a = sample_combos(50, 2026);
        let b = sample_combos(50, 2026);
        assert_eq!(a, b);
        let set: HashSet<_> = a.iter().collect();
        assert_eq!(set.len(), 50);
        let c = sample_combos(50, 2027);
        assert_ne!(a, c);
    }

    #[test]
    fn sample_uniqueness_at_100() {
        let s = sample_combos(100, 2026);
        let set: HashSet<_> = s.into_iter().collect();
        assert_eq!(set.len(), 100);
    }

    #[test]
    fn strategy_name_has_seven_parts() {
        let s = sample_combos(20, 2026);
        for c in &s {
            let name = combo_to_strategy_name(c);
            assert_eq!(name.split('_').count(), 7);
        }
    }
}
