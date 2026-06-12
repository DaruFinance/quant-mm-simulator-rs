//! quant-mm-simulator-rs — high-performance event-driven market-making simulator.
//!
//! Replays an L2 book + trade tape with queue-position-aware fills, continuous
//! inventory, a library of microstructure quoting models, a hedge engine, and a
//! costed multi-leg trade ledger. Numeric output is held to bit-level parity
//! against an independent reference implementation.
//!
//! Each subsystem lands behind a cargo feature so consumers compile only what
//! they use.

#[cfg(feature = "ingest")]
pub mod ingest;

#[cfg(feature = "sim")]
pub mod sim;

#[cfg(feature = "quoter")]
pub mod quoter;

#[cfg(feature = "hedge")]
pub mod hedge;

#[cfg(feature = "models")]
pub mod models;

#[cfg(feature = "logs")]
pub mod logs;

#[cfg(feature = "t4-corpus")]
pub mod t4_corpus;
