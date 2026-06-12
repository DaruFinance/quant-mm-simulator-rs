//! L2 book + trade-tape ingestion.
//!
//! Mirror of Python's `mmsim.ingest.lob`.  Public surface:
//!   - `SnapshotEvent` / `TradeEvent` — frozen-style records
//!   - `Event` — sum type wrapping both kinds
//!   - `Book` — reconstructed L2 state at an instant
//!   - `load_lob(snapshots_csv, trades_csv)` — chronologically sorted stream
//!   - `reconstruct_book_at(stream, t_ns)` — most-recent snapshot at-or-before `t_ns`
//!
//! CSV-first by design: the parity script converts the canonical
//! parquet fixtures to wide-format CSV (`bid_px_0, bid_sz_0, ...`)
//! at runtime, mirroring the carry/pairs convention in
//! `quant-research-framework-rs`.  No parquet dep on the Rust side.

#![cfg(feature = "ingest")]

pub mod lob;

pub use lob::{
    load_lob, reconstruct_book_at,
    Book, Event, EventStream, SnapshotEvent, TradeEvent,
};
