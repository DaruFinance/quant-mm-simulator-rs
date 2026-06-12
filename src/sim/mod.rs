//! Sim subpackage: event-driven loop, fills, queue, inventory.
//!
//! The event loop (`loop.rs`) drives the queue, fill model, and
//! inventory tracker.

#![cfg(feature = "sim")]

pub mod sim_loop;
pub mod queue;
pub mod fills;
pub mod inventory;

pub use sim_loop::{
    run_sim, run_sim_with_model, Fill, Order, QuoteRequest, SimResult,
};
pub use queue::{track_queue_position, QueueSample, QueueTrace, QueueTracker};
pub use fills::{FillModel, QueueAwareFillModel, TakerRequest};
pub use inventory::{
    inventory_path, InventorySample, InventoryTrace, InventoryTracker,
};
