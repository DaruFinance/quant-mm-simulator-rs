//! T4 Market-Making corpus generator layer.
//!
//! Direct port of Python's `mmsim.t4_corpus`. The T4 corpus layer
//! drives the existing mmsim engine primitives at scale by enumerating
//! the cross-product of 7 structural [B] axes:
//!
//!   QUOTING_MODELS (8) × INVENTORY_PENALTIES (6) × ADVERSE_FILTERS (7)
//!   × HEDGE_MODES (4) × REFERENCE_PRICES (6) × QUOTE_SHAPES (5)
//!   × REFRESH_TRIGGERS (5) = 201,600 combos
//!
//! Module layout (1:1 with Python):
//!
//! | Python module       | Rust module       | Purpose |
//! |---|---|---|
//! | `combos.py`         | [`combos`]        | 201,600 structural-combo grid |
//! | `search_space.py`   | [`search_space`]  | 7 IS-tune axes |
//! | `t4_composer.py`    | [`composer`]      | combo+params -> strategy bundle |
//! | `t4_runner.py`      | [`runner`]        | engine driver (per-fill ledger) |
//! | `t4_corpus_gen.py`  | [`corpus_gen`]    | resumable (asset × combo) writer |
//!
//! # Parity divergences (documented, intentional)
//!
//! 1. [`combos::sample_combos`] — same divergence pattern as T5/T6:
//!    `np.random.default_rng(seed)` and `rand_pcg::Pcg64::seed_from_u64`
//!    have different SeedSequence mixing. The Rust sample is deterministic
//!    given the Rust seed and every index lies in `[0, GRID_SIZE)`, but
//!    the LISTS do not match Python bit-for-bit. The parity script
//!    applies a shape-only check for `sample_combos`.
//! 2. [`search_space::SearchSpace::sample`] — same divergence; shape-only
//!    parity assertion.
//! 3. `options_vega_stub` hedge mode — same fallback in both: routes to
//!    no-op (None / Option::None). T3 options book is deferred per spec.
//! 4. `model_pred` reference price — Python has no ML layer either; both
//!    sides route to a deterministic mid-fallback closure. Bit-identical.

#![cfg(feature = "t4-corpus")]

pub mod combos;
pub mod search_space;
pub mod composer;
pub mod runner;
pub mod corpus_gen;

pub use combos::{
    Combo, GRID_SIZE,
    QUOTING_MODELS, INVENTORY_PENALTIES, ADVERSE_FILTERS,
    HEDGE_MODES, REFERENCE_PRICES, QUOTE_SHAPES, REFRESH_TRIGGERS,
    iter_all_combos, sample_combos, combo_to_strategy_name,
};
pub use search_space::{Axis, SearchSpace, search_space_t4};
pub use composer::{T4StrategyConfig, build_mm_strategy};
pub use runner::{
    T4RunResult, run_t4_combo, LEG_COLS,
    MAKER_FEE_PCT, TAKER_FEE_PCT, SLIP_PCT,
};
pub use corpus_gen::{T4CorpusResult, generate_t4_corpus};
