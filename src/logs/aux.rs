//! Auxiliary log streams.
//!
//! Mirror of Python's `mmsim.logs.aux`.  Three sidecar logs:
//!   - `FillRateLogger` — per-second `(n_fills, total_qty)` buckets.
//!   - `InventoryLogger` — per-second inventory snapshots with
//!     carry-forward into empty buckets.
//!   - `QueuePosLogger` — per-(snapshot, active-order) queue position
//!     samples.
//!
//! All three are downstream-only: every row at time `t` derives from
//! sim state at or before `t`.  Bucket key = `ts_ns / 1_000_000_000`.

#![cfg(feature = "logs")]

use std::collections::BTreeMap;
use std::fs::{create_dir_all, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use crate::sim::fills::FillModel;
use crate::sim::sim_loop::{Fill, Order};

pub const NS_PER_S: i64 = 1_000_000_000;

// ===================================================================== //
// FillRateLogger
// ===================================================================== //

#[derive(Debug, Clone, PartialEq)]
pub struct FillRateBucket {
    pub bucket_s: i64,
    pub t_ns_start: i64,
    pub n_fills: u64,
    pub total_qty: f64,
}

#[derive(Debug, Clone, Default)]
pub struct FillRateLogger {
    n: BTreeMap<i64, u64>,
    qty: BTreeMap<i64, f64>,
    min_bucket: Option<i64>,
    max_bucket: Option<i64>,
}

impl FillRateLogger {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn observe_fill(&mut self, fill: &Fill) {
        let b = fill.ts_ns.div_euclid(NS_PER_S);
        *self.n.entry(b).or_insert(0) += 1;
        *self.qty.entry(b).or_insert(0.0) += fill.size;
        self.min_bucket = Some(self.min_bucket.map_or(b, |x| x.min(b)));
        self.max_bucket = Some(self.max_bucket.map_or(b, |x| x.max(b)));
    }

    pub fn buckets(&self, start_s: Option<i64>, end_s: Option<i64>) -> Vec<FillRateBucket> {
        let start = start_s.unwrap_or(self.min_bucket.unwrap_or(0));
        let end = end_s.unwrap_or(self.max_bucket.unwrap_or(start - 1));
        let mut out = Vec::new();
        let mut b = start;
        while b <= end {
            out.push(FillRateBucket {
                bucket_s: b,
                t_ns_start: b * NS_PER_S,
                n_fills: *self.n.get(&b).unwrap_or(&0),
                total_qty: *self.qty.get(&b).unwrap_or(&0.0),
            });
            b += 1;
        }
        out
    }
}

// ===================================================================== //
// InventoryLogger
// ===================================================================== //

#[derive(Debug, Clone, PartialEq)]
pub struct InventoryBucket {
    pub bucket_s: i64,
    pub t_ns_start: i64,
    pub inv: f64,
    pub n_fills_so_far: u64,
}

#[derive(Debug, Clone, Default)]
pub struct InventoryLogger {
    inv: f64,
    cum_fills: u64,
    end_inv: BTreeMap<i64, f64>,
    end_count: BTreeMap<i64, u64>,
    min_bucket: Option<i64>,
    max_bucket: Option<i64>,
}

impl InventoryLogger {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn observe_fill(&mut self, fill: &Fill) {
        let signed = fill.size * (fill.side as f64);
        self.inv += signed;
        self.cum_fills += 1;
        let b = fill.ts_ns.div_euclid(NS_PER_S);
        self.end_inv.insert(b, self.inv);
        self.end_count.insert(b, self.cum_fills);
        self.min_bucket = Some(self.min_bucket.map_or(b, |x| x.min(b)));
        self.max_bucket = Some(self.max_bucket.map_or(b, |x| x.max(b)));
    }

    pub fn buckets(&self, start_s: Option<i64>, end_s: Option<i64>) -> Vec<InventoryBucket> {
        let start = start_s.unwrap_or(self.min_bucket.unwrap_or(0));
        let end = end_s.unwrap_or(self.max_bucket.unwrap_or(start - 1));
        let mut out = Vec::new();
        let mut cur_inv = 0.0_f64;
        let mut cur_count = 0u64;
        let mut b = start;
        while b <= end {
            if let Some(v) = self.end_inv.get(&b) {
                cur_inv = *v;
                cur_count = *self.end_count.get(&b).unwrap();
            }
            out.push(InventoryBucket {
                bucket_s: b,
                t_ns_start: b * NS_PER_S,
                inv: cur_inv,
                n_fills_so_far: cur_count,
            });
            b += 1;
        }
        out
    }
}

// ===================================================================== //
// QueuePosLogger
// ===================================================================== //

#[derive(Debug, Clone)]
pub struct QueuePosSample {
    pub ts_ns: i64,
    pub order_id: u64,
    pub side: i32,
    pub price: f64,
    pub queue_pos: f64,
    pub frozen: bool,
}

#[derive(Debug, Clone, Default)]
pub struct QueuePosLogger {
    samples: Vec<QueuePosSample>,
}

impl QueuePosLogger {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn observe_queue_pos(
        &mut self,
        ts_ns: i64,
        order_id: u64,
        side: i32,
        price: f64,
        queue_pos: f64,
        frozen: bool,
    ) {
        self.samples.push(QueuePosSample {
            ts_ns, order_id, side, price, queue_pos, frozen,
        });
    }

    pub fn samples(&self) -> &[QueuePosSample] {
        &self.samples
    }
}

// ===================================================================== //
// AuxLogs aggregate
// ===================================================================== //

#[derive(Debug, Clone, Default)]
pub struct AuxLogs {
    pub fill_rate: FillRateLogger,
    pub inventory: InventoryLogger,
    pub queue_pos: QueuePosLogger,
}

impl AuxLogs {
    pub fn new() -> Self {
        Self::default()
    }

    /// Dispatch a fill to both fill-rate and inventory loggers.
    /// Queue-pos requires per-snapshot calls, which the driver
    /// issues directly.
    pub fn observe_fill(&mut self, fill: &Fill) {
        self.fill_rate.observe_fill(fill);
        self.inventory.observe_fill(fill);
    }

    /// Write the three CSVs to `out_dir`.  Returns the three paths.
    pub fn to_csvs(
        &self,
        out_dir: &Path,
        start_s: Option<i64>,
        end_s: Option<i64>,
    ) -> std::io::Result<(PathBuf, PathBuf, PathBuf)> {
        create_dir_all(out_dir)?;

        let fr_path = out_dir.join("fill_rate.csv");
        {
            let mut w = BufWriter::new(File::create(&fr_path)?);
            writeln!(w, "bucket_s,t_ns_start,n_fills,total_qty")?;
            for b in self.fill_rate.buckets(start_s, end_s) {
                writeln!(w, "{},{},{},{:.12}",
                         b.bucket_s, b.t_ns_start, b.n_fills, b.total_qty)?;
            }
        }

        let inv_path = out_dir.join("inventory.csv");
        {
            let mut w = BufWriter::new(File::create(&inv_path)?);
            writeln!(w, "bucket_s,t_ns_start,inv,n_fills_so_far")?;
            for b in self.inventory.buckets(start_s, end_s) {
                writeln!(w, "{},{},{:.12},{}",
                         b.bucket_s, b.t_ns_start, b.inv, b.n_fills_so_far)?;
            }
        }

        let qp_path = out_dir.join("queue_pos.csv");
        {
            let mut w = BufWriter::new(File::create(&qp_path)?);
            writeln!(w, "ts_ns,order_id,side,price,queue_pos,frozen")?;
            for s in self.queue_pos.samples() {
                writeln!(w, "{},{},{},{:.12},{:.12},{}",
                         s.ts_ns, s.order_id, s.side, s.price, s.queue_pos,
                         if s.frozen { 1 } else { 0 })?;
            }
        }

        Ok((fr_path, inv_path, qp_path))
    }
}

// ===================================================================== //
// Sim driver
// ===================================================================== //

/// Drive an event stream through the model-based sim loop while
/// collecting the three aux log streams.  Returns `(SimResult, AuxLogs)`.
///
/// Mechanics mirror the Python sibling:
///   - At each snapshot, the FillModel's on_snapshot fires *first*
///     (loop convention), then we log queue positions of every
///     active order from the underlying QueueAware tracker map.
///   - At the end, we walk `result.fills` once to drive fill-rate /
///     inventory loggers.
///
/// Causality: snapshot-time queue_pos reads only the tracker state
/// at the just-applied snapshot, which itself only saw events with
/// ts_ns <= snap.ts_ns.  Fill records are produced by the loop in
/// ts-monotonic order.
pub fn run_sim_with_aux_logs<Q>(
    events: &crate::ingest::EventStream,
    mut quoter: Q,
    model: &mut FillModel,
) -> (crate::sim::sim_loop::SimResult, AuxLogs)
where
    Q: FnMut(Option<&crate::ingest::Book>, &[Order], i64)
        -> Vec<crate::sim::sim_loop::QuoterItem>,
{
    use crate::ingest::{Book, Event};
    use crate::sim::sim_loop::{
        Fill, QuoterItem, SimResult,
    };
    use crate::sim::fills::TakerRequest;

    let mut aux = AuxLogs::new();
    let mut active: Vec<Order> = Vec::new();
    let mut out_fills: Vec<Fill> = Vec::new();
    let mut next_order_id: u64 = 0;
    let mut next_fill_id: u64 = 0;
    let mut n_snap = 0usize;
    let mut n_trade = 0usize;
    let mut n_quoter_calls = 0usize;
    let mut n_maker_fills = 0usize;
    let mut n_taker_fills = 0usize;

    for ev in events {
        match ev {
            Event::Snapshot(s) => {
                n_snap += 1;
                let book = Book {
                    ts_ns: s.ts_ns, bids: s.bids.clone(), asks: s.asks.clone(),
                };
                // 1) Push snapshot to model (cancel attribution).
                model.on_snapshot(s);
                // 2) Log queue positions of every active order from
                //    the QueueAware tracker map (read-only borrow).
                if let Some(trackers) = model.queue_trackers() {
                    for o in &active {
                        if let Some(tr) = trackers.get(&o.order_id) {
                            aux.queue_pos.observe_queue_pos(
                                s.ts_ns,
                                o.order_id,
                                o.side,
                                o.price,
                                tr.queue_pos,
                                tr.frozen,
                            );
                        }
                    }
                }
                // 3) Quoter decides.
                let items = quoter(Some(&book), &active, s.ts_ns);
                n_quoter_calls += 1;
                let mut makers: Vec<crate::sim::sim_loop::QuoteRequest> = Vec::new();
                let mut takers: Vec<TakerRequest> = Vec::new();
                for it in items {
                    match it {
                        QuoterItem::Maker(q) => makers.push(q),
                        QuoterItem::Taker(t) => takers.push(t),
                    }
                }
                // 4) Replace active makers; notify the model.
                let removed: Vec<u64> = active.iter().map(|o| o.order_id).collect();
                if !removed.is_empty() {
                    model.on_orders_removed(&removed);
                }
                let mut new_active: Vec<Order> = Vec::with_capacity(makers.len());
                for q in makers {
                    let o = Order {
                        order_id: next_order_id,
                        side: q.side, price: q.price, size: q.size,
                        placed_at_ns: s.ts_ns,
                    };
                    model.on_order_placed(&o, &book);
                    new_active.push(o);
                    next_order_id += 1;
                }
                active = new_active;
                // 5) Fire takers immediately.
                for req in &takers {
                    let rows = model.fill_taker(req, &book, s.ts_ns);
                    for (px, sz) in rows {
                        if sz <= 0.0 { continue; }
                        out_fills.push(Fill {
                            fill_id: next_fill_id,
                            order_id: u64::MAX,
                            ts_ns: s.ts_ns,
                            price: px, size: sz, side: req.side,
                            is_maker: false,
                        });
                        next_fill_id += 1;
                        n_taker_fills += 1;
                    }
                }
            }
            Event::Trade(t) => {
                n_trade += 1;
                if active.is_empty() { continue; }
                let hits = model.on_trade(t, &active);
                for (oid, fsize) in hits {
                    let pos = active.iter().position(|o| o.order_id == oid)
                        .unwrap_or_else(|| panic!(
                            "run_sim_with_aux_logs: order_id={} not in active set at ts={}",
                            oid, t.ts_ns));
                    if fsize <= 0.0 {
                        panic!("run_sim_with_aux_logs: non-positive fill size {}", fsize);
                    }
                    let actual = fsize.min(active[pos].size);
                    out_fills.push(Fill {
                        fill_id: next_fill_id,
                        order_id: active[pos].order_id,
                        ts_ns: t.ts_ns,
                        price: active[pos].price,
                        size: actual,
                        side: active[pos].side,
                        is_maker: true,
                    });
                    next_fill_id += 1;
                    n_maker_fills += 1;
                    active[pos].size -= actual;
                }
                active.retain(|o| o.size > 0.0);
            }
        }
    }

    // Walk fills for fill-rate + inventory.
    for f in &out_fills {
        aux.observe_fill(f);
    }

    let res = SimResult {
        fills: out_fills,
        n_events_processed: events.len(),
        n_snapshot_events: n_snap,
        n_trade_events: n_trade,
        n_quoter_calls,
        n_maker_fills,
        n_taker_fills,
        final_orders: active,
    };
    (res, aux)
}

// ===================================================================== //
// Unit tests
// ===================================================================== //

#[cfg(test)]
mod tests {
    use super::*;

    fn mkfill(ts: i64, side: i32, size: f64) -> Fill {
        Fill {
            fill_id: ts as u64, order_id: 0, ts_ns: ts,
            price: 100.0, size, side, is_maker: true,
        }
    }

    #[test]
    fn fill_rate_single_second() {
        let mut log = FillRateLogger::new();
        log.observe_fill(&mkfill(1_000_000_001, 1, 0.5));
        log.observe_fill(&mkfill(1_500_000_000, -1, 0.25));
        log.observe_fill(&mkfill(1_999_999_999, 1, 0.1));
        let bs = log.buckets(None, None);
        assert_eq!(bs.len(), 1);
        assert_eq!(bs[0].bucket_s, 1);
        assert_eq!(bs[0].n_fills, 3);
        assert!((bs[0].total_qty - 0.85).abs() < 1e-12);
        assert_eq!(bs[0].t_ns_start, 1_000_000_000);
    }

    #[test]
    fn fill_rate_spans_multiple_seconds_with_empty_in_middle() {
        let mut log = FillRateLogger::new();
        log.observe_fill(&mkfill(5_000_000_000, 1, 1.0));
        log.observe_fill(&mkfill(7_500_000_000, 1, 2.0));
        let bs = log.buckets(None, None);
        assert_eq!(bs.iter().map(|b| b.bucket_s).collect::<Vec<_>>(),
                   vec![5, 6, 7]);
        assert_eq!(bs.iter().map(|b| b.n_fills).collect::<Vec<_>>(),
                   vec![1, 0, 1]);
    }

    #[test]
    fn fill_rate_explicit_grid_overrides_observed_range() {
        let mut log = FillRateLogger::new();
        log.observe_fill(&mkfill(5_000_000_000, 1, 1.0));
        let bs = log.buckets(Some(3), Some(7));
        assert_eq!(bs.len(), 5);
        assert_eq!(bs[0].bucket_s, 3);
        assert_eq!(bs[4].bucket_s, 7);
        assert_eq!(bs.iter().map(|b| b.n_fills).collect::<Vec<_>>(),
                   vec![0, 0, 1, 0, 0]);
    }

    #[test]
    fn inventory_carry_forward_into_empty_bucket() {
        let mut log = InventoryLogger::new();
        log.observe_fill(&mkfill(2_000_000_000, 1, 0.5));   // b2 inv=+0.5
        log.observe_fill(&mkfill(2_500_000_000, -1, 0.2));  // b2 inv=+0.3
        log.observe_fill(&mkfill(4_000_000_000, 1, 0.4));   // b4 inv=+0.7
        let bs = log.buckets(None, None);
        assert_eq!(bs.iter().map(|b| b.bucket_s).collect::<Vec<_>>(),
                   vec![2, 3, 4]);
        let expected = [0.3, 0.3, 0.7];
        for (b, e) in bs.iter().zip(expected.iter()) {
            assert!((b.inv - e).abs() < 1e-12);
        }
        assert_eq!(bs.iter().map(|b| b.n_fills_so_far).collect::<Vec<_>>(),
                   vec![2, 2, 3]);
    }

    #[test]
    fn inventory_leading_empty_buckets_default_zero() {
        let mut log = InventoryLogger::new();
        log.observe_fill(&mkfill(3_000_000_000, 1, 0.5));
        let bs = log.buckets(Some(1), Some(4));
        assert_eq!(bs.len(), 4);
        assert_eq!(bs[0].inv, 0.0);
        assert_eq!(bs[1].inv, 0.0);
        assert!((bs[2].inv - 0.5).abs() < 1e-12);
        assert!((bs[3].inv - 0.5).abs() < 1e-12);
    }

    #[test]
    fn queue_pos_records_each_observation() {
        let mut log = QueuePosLogger::new();
        log.observe_queue_pos(1_000, 1, 1, 100.0, 5.0, false);
        log.observe_queue_pos(2_000, 1, 1, 100.0, 3.5, false);
        log.observe_queue_pos(3_000, 2, -1, 101.0, 0.0, true);
        let ss = log.samples();
        assert_eq!(ss.len(), 3);
        assert_eq!(ss[2].order_id, 2);
        assert!(ss[2].frozen);
    }

    #[test]
    fn aux_logs_to_csvs_round_trip() {
        let mut aux = AuxLogs::new();
        aux.observe_fill(&mkfill(1_000_000_000, 1, 0.5));
        aux.observe_fill(&mkfill(2_000_000_000, -1, 0.2));
        aux.queue_pos.observe_queue_pos(1_000_000_000, 1, 1, 100.0, 5.0, false);
        let td = tempdir();
        let (fr, inv, qp) = aux.to_csvs(&td, None, None).unwrap();
        assert!(fr.exists());
        assert!(inv.exists());
        assert!(qp.exists());
        // Quick header check.
        let fr_text = std::fs::read_to_string(&fr).unwrap();
        assert!(fr_text.starts_with("bucket_s,t_ns_start,n_fills,total_qty\n"));
        let lines: Vec<&str> = fr_text.lines().collect();
        // Header + 2 data rows (buckets 1 and 2).
        assert_eq!(lines.len(), 3);
        // Cleanup.
        let _ = std::fs::remove_dir_all(&td);
    }

    fn tempdir() -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("mmsim_logs_aux_test_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        p
    }
}
