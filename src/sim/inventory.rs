//! Continuous fractional inventory tracker.
//!
//! Mirror of Python's `mmsim.sim.inventory`.  Tracks net inventory
//! as f64 — every fill (maker or taker) shifts inventory by
//! `fill.size * fill.side`.

#![cfg(feature = "sim")]

use crate::sim::sim_loop::Fill;

#[derive(Debug, Clone, Copy)]
pub struct InventorySample {
    pub ts_ns: i64,
    pub inv: f64,
    pub fill_id: u64,
    pub fill_signed_size: f64,
}

#[derive(Debug, Clone)]
pub struct InventoryTrace {
    pub samples: Vec<InventorySample>,
    pub final_inv: f64,
    pub peak_long: f64,
    pub peak_short: f64,
    pub n_fills: usize,
}

#[derive(Debug, Clone, Default)]
pub struct InventoryTracker {
    pub inv: f64,
    pub peak_long: f64,
    pub peak_short: f64,
    pub samples: Vec<InventorySample>,
}

impl InventoryTracker {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn observe(&mut self, fill: &Fill) {
        let signed = fill.size * (fill.side as f64);
        self.inv += signed;
        if self.inv > self.peak_long {
            self.peak_long = self.inv;
        }
        if self.inv < self.peak_short {
            self.peak_short = self.inv;
        }
        self.samples.push(InventorySample {
            ts_ns: fill.ts_ns,
            inv: self.inv,
            fill_id: fill.fill_id,
            fill_signed_size: signed,
        });
    }

    pub fn trace(&self) -> InventoryTrace {
        InventoryTrace {
            samples: self.samples.clone(),
            final_inv: self.inv,
            peak_long: self.peak_long,
            peak_short: self.peak_short,
            n_fills: self.samples.len(),
        }
    }
}

pub fn inventory_path(fills: &[Fill]) -> InventoryTrace {
    let mut tr = InventoryTracker::new();
    for f in fills {
        tr.observe(f);
    }
    tr.trace()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fill(ts: i64, side: i32, size: f64) -> Fill {
        Fill {
            fill_id: ts as u64, order_id: 0, ts_ns: ts,
            price: 100.0, size, side, is_maker: true,
        }
    }

    #[test]
    fn signed_accumulation() {
        let mut t = InventoryTracker::new();
        t.observe(&fill(1, 1, 0.5));
        t.observe(&fill(2, -1, 0.3));
        t.observe(&fill(3, 1, 0.2));
        assert!((t.inv - 0.4).abs() < 1e-12);
    }

    #[test]
    fn peaks_track_extrema() {
        let mut t = InventoryTracker::new();
        t.observe(&fill(1, 1, 1.5));
        t.observe(&fill(2, -1, 4.0));
        t.observe(&fill(3, 1, 1.0));
        assert!((t.peak_long - 1.5).abs() < 1e-12);
        assert!((t.peak_short - (-2.5)).abs() < 1e-12);
    }
}
