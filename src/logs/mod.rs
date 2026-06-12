//! logs subpackage.
//!
//! Auxiliary log streams (fill-rate / inventory / queue-pos) that
//! ride alongside the trade ledger.  Mirrors Python's `mmsim.logs`.

#![cfg(feature = "logs")]

pub mod aux;

pub use aux::{
    AuxLogs, FillRateBucket, FillRateLogger,
    InventoryBucket, InventoryLogger,
    QueuePosLogger, QueuePosSample,
    run_sim_with_aux_logs, NS_PER_S,
};
