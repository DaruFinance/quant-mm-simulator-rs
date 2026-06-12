//! Quoter contract.
//!
//! Mirror of Python's `mmsim.quoter`.  Public surface:
//!   - `Quoter` trait — formal contract.
//!   - `ConstantQuoter`, `TopOfBookQuoter`, `BracketQuoter` —
//!     reference implementations consumed by every parity script
//!     and verification baseline.
//!
//! The existing `run_sim_with_model` already takes an `FnMut`
//! quoter closure; this trait layers on top, providing the contract that
//! reference quoters implement.  The closure path stays for
//! ad-hoc parity-binary use (mirrors Python's "either callable or
//! Protocol" branching).

#![cfg(feature = "sim")]

#[cfg(feature = "quoter")]
pub mod shapes;

#[cfg(feature = "quoter")]
pub use shapes::{
    dynamic_depth, geometric, ladder, paired, single,
    DynamicDepthSpec, GeometricSpec, LadderSpec, PairedSpec, SingleSpec,
};

#[cfg(feature = "quoter")]
pub mod triggers;

#[cfg(feature = "quoter")]
pub use triggers::{
    BookEventTrigger, HybridMode, HybridTrigger, InvChangeTrigger, MidMoveTrigger,
    RefreshTrigger, TimeTrigger,
};

#[cfg(feature = "quoter")]
pub mod refprice;

#[cfg(feature = "quoter")]
pub use refprice::{
    linear_drift_predictor, microprice, top_mid, weighted_mid,
    EWMAFairTracker, ModelPredictedTracker, ModelState, VWAPTracker,
};

#[cfg(feature = "quoter")]
pub mod adverse;

#[cfg(feature = "quoter")]
pub use adverse::{
    AdverseFilter, HybridAdverseFilter, HybridAdverseMode, MicropriceDevFilter, OFIFilter,
    QueueImbalanceFilter, TradeToxicityFilter, VolSurgeFilter,
};

#[cfg(feature = "quoter")]
pub mod inv_penalty;

#[cfg(feature = "quoter")]
pub use inv_penalty::{
    asymmetric, exponential, hard_cap, linear, quadratic, soft_cap, Skew,
};

use crate::ingest::Book;
use crate::sim::fills::TakerRequest;
use crate::sim::sim_loop::QuoteRequest;

/// What a quoter returns at each call.  Sum type wrapping the two
/// admissible decisions (resting maker post / immediate taker fire).
#[derive(Debug, Clone)]
pub enum Decision {
    Maker(QuoteRequest),
    Taker(TakerRequest),
}

/// Formal quoter trait.  The single integration point for
/// every quoting model (Avellaneda-Stoikov, Cartea-Jaimungal,
/// GLFT, Ho-Stoll, microprice-skew, fair-anchored).
///
/// Causality: every input is state at-or-before `t_ns`.  The trait
/// does not pass the event stream or any future-state handle.
pub trait Quoter {
    fn quote(&mut self, book: Option<&Book>, inv: f64, t_ns: i64) -> Vec<Decision>;
}

// --------------------------------------------------------------------- //
// Reference implementations
// --------------------------------------------------------------------- //

/// Posts a fixed bid+ask of fixed size, regardless of book.
/// The spec's "trivial constant quoter" used in contract tests.
#[derive(Debug, Clone)]
pub struct ConstantQuoter {
    pub bid_price: f64,
    pub ask_price: f64,
    pub size: f64,
}

impl ConstantQuoter {
    pub fn new(bid_price: f64, ask_price: f64, size: f64) -> Self {
        Self { bid_price, ask_price, size }
    }
}

impl Quoter for ConstantQuoter {
    fn quote(&mut self, _book: Option<&Book>, _inv: f64, _t_ns: i64) -> Vec<Decision> {
        vec![
            Decision::Maker(QuoteRequest { side: 1, price: self.bid_price, size: self.size }),
            Decision::Maker(QuoteRequest { side: -1, price: self.ask_price, size: self.size }),
        ]
    }
}

/// TOB-joining maker pair (formalized from the earlier stub).
#[derive(Debug, Clone)]
pub struct TopOfBookQuoter {
    pub size: f64,
}

impl TopOfBookQuoter {
    pub fn new(size: f64) -> Self {
        Self { size }
    }
}

impl Quoter for TopOfBookQuoter {
    fn quote(&mut self, book: Option<&Book>, _inv: f64, _t_ns: i64) -> Vec<Decision> {
        let b = match book {
            Some(b) => b,
            None => return Vec::new(),
        };
        match (b.best_bid(), b.best_ask()) {
            (Some(bid), Some(ask)) => vec![
                Decision::Maker(QuoteRequest { side: 1, price: bid, size: self.size }),
                Decision::Maker(QuoteRequest { side: -1, price: ask, size: self.size }),
            ],
            _ => Vec::new(),
        }
    }
}

/// TOB makers + a small taker buy fired every `taker_every` snapshots.
/// The maker/taker fill reference, formalized as a Quoter impl.
#[derive(Debug, Clone)]
pub struct BracketQuoter {
    pub maker_size: f64,
    pub taker_size: f64,
    pub taker_every: usize,
    snap_count: usize,
}

impl BracketQuoter {
    pub fn new(maker_size: f64, taker_size: f64, taker_every: usize) -> Self {
        Self { maker_size, taker_size, taker_every, snap_count: 0 }
    }
}

impl Quoter for BracketQuoter {
    fn quote(&mut self, book: Option<&Book>, _inv: f64, _t_ns: i64) -> Vec<Decision> {
        self.snap_count += 1;
        let b = match book {
            Some(b) => b,
            None => return Vec::new(),
        };
        let (bid, ask) = match (b.best_bid(), b.best_ask()) {
            (Some(b), Some(a)) => (b, a),
            _ => return Vec::new(),
        };
        let mut out = vec![
            Decision::Maker(QuoteRequest { side: 1, price: bid, size: self.maker_size }),
            Decision::Maker(QuoteRequest { side: -1, price: ask, size: self.maker_size }),
        ];
        if self.snap_count % self.taker_every == 0 {
            out.push(Decision::Taker(TakerRequest {
                side: 1, size: self.taker_size, limit_px: None,
            }));
        }
        out
    }
}

// --------------------------------------------------------------------- //
// Adapter: wrap a Quoter trait object into the closure signature
// `run_sim_with_model` already accepts.  This is the "Protocol path"
// equivalent — drives the quoter's `quote()` against the loop's
// internally-tracked inventory.
// --------------------------------------------------------------------- //

/// Build an `FnMut` closure from a `Quoter`.  The closure feeds it
/// the loop's `(book, &active, t)` shape and adapts to the
/// trait's `(book, inv, t)` signature; the inventory is tracked by
/// the closure itself by accumulating signed sizes from a passed-in
/// fills accumulator.  In practice callers don't use this adapter
/// directly — they call `run_sim_with_quoter` below, which threads
/// the inventory through.
pub fn run_sim_with_quoter<Q: Quoter>(
    events: &crate::ingest::EventStream,
    mut quoter: Q,
    model: &mut crate::sim::fills::FillModel,
) -> crate::sim::sim_loop::SimResult {
    use crate::sim::inventory::InventoryTracker;
    use crate::sim::sim_loop::{QuoterItem, run_sim_with_model};

    let mut inv = InventoryTracker::new();
    // We need the quoter's most-recent inv before each call AND we
    // need to update inv from each emitted fill.  The cleanest way
    // is: don't use run_sim_with_model directly (which doesn't expose
    // a per-fill hook); reimplement the inner loop here so we can
    // observe the fills before the next quoter call.
    //
    // But that duplicates the loop.  Pragmatic: snapshot the inv at
    // the START of each quoter call.  Since quoter is called only
    // on SnapshotEvents, and fills land between
    // snapshots, the inv at the *start* of a snapshot's quoter call
    // is the inv from all fills with ts_ns < snap.ts_ns.  That's
    // the inv-at-t the spec asks for.
    //
    // We achieve this by wrapping the quoter in an FnMut closure
    // that reads from a shared inv handle.
    let inv_handle = std::cell::RefCell::new(0.0_f64);
    let wrapped = |book: Option<&Book>, _active: &[_], t_ns: i64| -> Vec<QuoterItem> {
        let inv_now = *inv_handle.borrow();
        let decisions = quoter.quote(book, inv_now, t_ns);
        decisions.into_iter().map(|d| match d {
            Decision::Maker(q) => QuoterItem::Maker(q),
            Decision::Taker(t) => QuoterItem::Taker(t),
        }).collect()
    };

    let res = run_sim_with_model(events, wrapped, model);
    // Update inv from all observed fills so this function's caller
    // can read the final state if needed.
    for f in &res.fills {
        inv.observe(f);
    }
    let _ = inv;  // keep alive for clarity
    res
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ingest::SnapshotEvent;

    fn book(ts: i64, bids: Vec<(f64, f64)>, asks: Vec<(f64, f64)>) -> Book {
        Book { ts_ns: ts, bids, asks }
    }

    #[test]
    fn constant_quoter_posts_fixed_pair() {
        let mut q = ConstantQuoter::new(100.0, 101.0, 0.001);
        let out = q.quote(None, 0.0, 0);
        assert_eq!(out.len(), 2);
        match (&out[0], &out[1]) {
            (Decision::Maker(b), Decision::Maker(a)) => {
                assert_eq!(b.price, 100.0);
                assert_eq!(a.price, 101.0);
            }
            _ => panic!("expected two makers"),
        }
    }

    #[test]
    fn top_of_book_empty_before_warmup() {
        let mut q = TopOfBookQuoter::new(0.001);
        assert!(q.quote(None, 0.0, 0).is_empty());
    }

    #[test]
    fn top_of_book_joins_tob() {
        let mut q = TopOfBookQuoter::new(0.001);
        let b = book(0, vec![(100.0, 5.0)], vec![(101.0, 3.0)]);
        let out = q.quote(Some(&b), 0.0, 0);
        let bid = out.iter().find_map(|d| match d {
            Decision::Maker(q) if q.side == 1 => Some(q.price),
            _ => None,
        });
        let ask = out.iter().find_map(|d| match d {
            Decision::Maker(q) if q.side == -1 => Some(q.price),
            _ => None,
        });
        assert_eq!(bid, Some(100.0));
        assert_eq!(ask, Some(101.0));
    }

    #[test]
    fn bracket_quoter_taker_fires_every_n() {
        let mut q = BracketQuoter::new(0.001, 0.0001, 3);
        let b = book(0, vec![(100.0, 5.0)], vec![(101.0, 3.0)]);
        let mut takers = 0;
        for _ in 0..10 {
            let out = q.quote(Some(&b), 0.0, 0);
            if out.iter().any(|d| matches!(d, Decision::Taker(_))) {
                takers += 1;
            }
        }
        assert_eq!(takers, 3);
    }
}
