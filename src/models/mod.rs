//! quoting-model library.
//!
//! Mirror of Python's `mmsim.models`.  Eight concrete quoting models,
//! each implementing the `Quoter` trait from `crate::quoter`.  The
//! library is the capstone composition over the quoting
//! primitives: it builds on top of refprices, shapes,
//! inv-penalty, and adverse filters — all verified
//! individually before this item lands.
//!
//! Models
//! ------
//! | Type                       | Family             | Stateful? |
//! |----------------------------|--------------------|-----------|
//! | `SymmetricQuoter`          | fixed half-spread  | No        |
//! | `LadderQuoter`             | multi-level ladder | No        |
//! | `MicropriceSkewQuoter`     | book-imbalance ref | No        |
//! | `FairAnchoredQuoter`       | EWMA-fair anchor   | Yes (EWMA)|
//! | `AvellanedaStoikovQuoter`  | AS closed-form     | Yes (σ)   |
//! | `CarteaJaimungalQuoter`    | CJ variant         | Yes (σ)   |
//! | `GLFTQuoter`               | GLFT closed-form   | Yes (σ)   |
//! | `HoStollQuoter`            | classical (1981)   | Yes (σ)   |
//!
//! All 8 implement `Quoter::quote(book, inv, t_ns) -> Vec<Decision>`
//! and plug directly into `run_sim_with_quoter`.
//!
//! Helpers
//! -------
//! `RollingSigma::new(window_ns)` — shared trailing-window log-return
//! std tracker (mid-fed) used by the AS-family models.  Deterministic,
//! leak-free, and parity-friendly.

#![cfg(feature = "models")]

pub mod vol;
pub use vol::RollingSigma;

pub mod symmetric;
pub use symmetric::SymmetricQuoter;

pub mod ladder;
pub use ladder::LadderQuoter;

pub mod microprice_skew;
pub use microprice_skew::MicropriceSkewQuoter;

pub mod fair_anchored;
pub use fair_anchored::FairAnchoredQuoter;

pub mod avellaneda_stoikov;
pub use avellaneda_stoikov::AvellanedaStoikovQuoter;

pub mod cartea_jaimungal;
pub use cartea_jaimungal::CarteaJaimungalQuoter;

pub mod glft;
pub use glft::GLFTQuoter;

pub mod ho_stoll;
pub use ho_stoll::HoStollQuoter;
