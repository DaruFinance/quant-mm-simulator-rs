//! Hedge subpackage: hedge engine.
//!
//! Mirror of Python's `mmsim.hedge`.  Lives behind the `hedge`
//! cargo feature (which itself depends on `sim` for the shared
//! `Fill` record + `Book`).

#![cfg(feature = "hedge")]

pub mod engine;

pub use engine::{HedgeDecision, HedgeEngine, HEDGE_ORDER_ID};
