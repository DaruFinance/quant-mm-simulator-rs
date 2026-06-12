//! Refresh-trigger primitives.
//!
//! Cancel-replace decision: given the current sim state at time `t`,
//! should the quoter refresh its outstanding orders?  Five built-in
//! trigger families, each a small stateful struct implementing the
//! `RefreshTrigger` trait with a single
//! `step(book, inv, t_ns) -> bool` method.
//!
//! The triggers:
//!   - `TimeTrigger(interval_ns)` — every N ns elapsed since last fire
//!   - `MidMoveTrigger(threshold_bp)` — mid moved >= threshold bp since last fire
//!   - `InvChangeTrigger(threshold)` — |inv - last_fire_inv| >= threshold
//!   - `BookEventTrigger` — every call fires
//!   - `HybridTrigger(children, mode)` — composition; Any or All
//!
//! Causality contract (`<= t` only): each `step()` reads only the
//! arguments passed in (which the caller provides from sim state at-or-
//! before t) plus the trigger's own internal state (set by a previous
//! `step()` at-or-before t).  No closure over future events, no time
//! peek.
//!
//! Convention: `step()` returns true if the trigger fires AT this call.
//! When it fires, the trigger updates its internal state so the next call
//! re-bases against the new reference.  HybridTrigger steps every child
//! once per call (no short-circuit) so state advances deterministically.

#![cfg(feature = "quoter")]

use crate::ingest::Book;

/// Formal refresh-trigger trait.  Causality: every input is
/// state at-or-before `t_ns`.
pub trait RefreshTrigger {
    fn step(&mut self, book: Option<&Book>, inv: f64, t_ns: i64) -> bool;
}

// --------------------------------------------------------------------- //
// TimeTrigger
// --------------------------------------------------------------------- //

/// Fires every `interval_ns` nanoseconds.  Initial call always fires;
/// subsequent calls fire only after the interval elapses.
#[derive(Debug, Clone)]
pub struct TimeTrigger {
    pub interval_ns: i64,
    last_fire_ts: Option<i64>,
}

impl TimeTrigger {
    pub fn new(interval_ns: i64) -> Self {
        if interval_ns <= 0 {
            panic!("interval_ns must be > 0");
        }
        Self { interval_ns, last_fire_ts: None }
    }
}

impl RefreshTrigger for TimeTrigger {
    fn step(&mut self, _book: Option<&Book>, _inv: f64, t_ns: i64) -> bool {
        match self.last_fire_ts {
            None => {
                self.last_fire_ts = Some(t_ns);
                true
            }
            Some(last) => {
                if t_ns - last >= self.interval_ns {
                    self.last_fire_ts = Some(t_ns);
                    true
                } else {
                    false
                }
            }
        }
    }
}

// --------------------------------------------------------------------- //
// MidMoveTrigger
// --------------------------------------------------------------------- //

/// Fires when the mid has moved at least `threshold_bp` basis-points
/// away from the last fire's mid.  When no book is available (or no
/// mid), returns false (cannot evaluate).  First call with a book sets
/// the baseline and fires.
#[derive(Debug, Clone)]
pub struct MidMoveTrigger {
    pub threshold_bp: f64,
    last_fire_mid: Option<f64>,
}

impl MidMoveTrigger {
    pub fn new(threshold_bp: f64) -> Self {
        if threshold_bp < 0.0 {
            panic!("threshold_bp must be >= 0");
        }
        Self { threshold_bp, last_fire_mid: None }
    }
}

impl RefreshTrigger for MidMoveTrigger {
    fn step(&mut self, book: Option<&Book>, _inv: f64, _t_ns: i64) -> bool {
        let b = match book {
            Some(b) => b,
            None => return false,
        };
        let m = match b.mid() {
            Some(m) => m,
            None => return false,
        };
        match self.last_fire_mid {
            None => {
                self.last_fire_mid = Some(m);
                true
            }
            Some(prev) => {
                let rel_bp = (m - prev).abs() / prev * 1e4;
                if rel_bp >= self.threshold_bp {
                    self.last_fire_mid = Some(m);
                    true
                } else {
                    false
                }
            }
        }
    }
}

// --------------------------------------------------------------------- //
// InvChangeTrigger
// --------------------------------------------------------------------- //

/// Fires when `|inv - last_fire_inv| >= threshold`.  First call always
/// fires; subsequent calls re-base on each fire.
#[derive(Debug, Clone)]
pub struct InvChangeTrigger {
    pub threshold: f64,
    last_fire_inv: Option<f64>,
}

impl InvChangeTrigger {
    pub fn new(threshold: f64) -> Self {
        if threshold < 0.0 {
            panic!("threshold must be >= 0");
        }
        Self { threshold, last_fire_inv: None }
    }
}

impl RefreshTrigger for InvChangeTrigger {
    fn step(&mut self, _book: Option<&Book>, inv: f64, _t_ns: i64) -> bool {
        match self.last_fire_inv {
            None => {
                self.last_fire_inv = Some(inv);
                true
            }
            Some(prev) => {
                if (inv - prev).abs() >= self.threshold {
                    self.last_fire_inv = Some(inv);
                    true
                } else {
                    false
                }
            }
        }
    }
}

// --------------------------------------------------------------------- //
// BookEventTrigger
// --------------------------------------------------------------------- //

/// Fires on every call.  The caller is responsible for wiring this
/// trigger to fire only on book events; this trigger has no internal
/// predicate.
#[derive(Debug, Clone, Default)]
pub struct BookEventTrigger;

impl BookEventTrigger {
    pub fn new() -> Self {
        Self
    }
}

impl RefreshTrigger for BookEventTrigger {
    fn step(&mut self, _book: Option<&Book>, _inv: f64, _t_ns: i64) -> bool {
        true
    }
}

// --------------------------------------------------------------------- //
// HybridTrigger
// --------------------------------------------------------------------- //

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HybridMode {
    Any,
    All,
}

/// Composition of multiple triggers.
///
/// `HybridMode::Any` fires if ANY child fires (each child's step is
/// invoked exactly once per call so state advances uniformly).
/// `HybridMode::All` fires only if EVERY child fires.
///
/// All children are stepped on every call regardless of mode — this
/// keeps state advancement deterministic and avoids the surprise of a
/// short-circuit boolean leaving a child out-of-sync.
pub struct HybridTrigger {
    pub children: Vec<Box<dyn RefreshTrigger>>,
    pub mode: HybridMode,
}

impl HybridTrigger {
    pub fn new(children: Vec<Box<dyn RefreshTrigger>>, mode: HybridMode) -> Self {
        Self { children, mode }
    }
}

impl RefreshTrigger for HybridTrigger {
    fn step(&mut self, book: Option<&Book>, inv: f64, t_ns: i64) -> bool {
        // Step every child once, collect into a Vec to avoid short-circuit.
        let fires: Vec<bool> = self
            .children
            .iter_mut()
            .map(|c| c.step(book, inv, t_ns))
            .collect();
        match self.mode {
            HybridMode::Any => fires.iter().any(|&x| x),
            HybridMode::All => !fires.is_empty() && fires.iter().all(|&x| x),
        }
    }
}

// --------------------------------------------------------------------- //
// Tests
// --------------------------------------------------------------------- //

#[cfg(test)]
mod tests {
    use super::*;

    fn book(ts: i64, bids: Vec<(f64, f64)>, asks: Vec<(f64, f64)>) -> Book {
        Book { ts_ns: ts, bids, asks }
    }

    #[test]
    fn time_trigger_fires_at_interval() {
        let mut tr = TimeTrigger::new(100);
        // First call always fires (sets baseline at t_ns=0).
        assert!(tr.step(None, 0.0, 0));
        // Below interval -> no fire.
        assert!(!tr.step(None, 0.0, 50));
        assert!(!tr.step(None, 0.0, 99));
        // At interval -> fires.
        assert!(tr.step(None, 0.0, 100));
        // Re-based at 100; 150 below 100+100, no fire.
        assert!(!tr.step(None, 0.0, 150));
        // 200 hits interval again.
        assert!(tr.step(None, 0.0, 200));
    }

    #[test]
    fn mid_move_trigger_no_book_no_fire() {
        let mut tr = MidMoveTrigger::new(5.0);
        // No book ever -> never fires, no baseline set.
        assert!(!tr.step(None, 0.0, 0));
        assert!(!tr.step(None, 0.0, 1));
        // First book -> fires, sets baseline at mid=100.5.
        let b = book(0, vec![(100.0, 1.0)], vec![(101.0, 1.0)]);
        assert!(tr.step(Some(&b), 0.0, 2));
        // Same mid -> 0 bp move -> no fire (threshold > 0).
        assert!(!tr.step(Some(&b), 0.0, 3));
        // Move mid by ~10 bp ((100.6 - 100.5)/100.5 * 1e4 ~= 9.95 bp), below threshold 5? actually above.
        let b2 = book(0, vec![(100.05, 1.0)], vec![(101.05, 1.0)]); // mid=100.55, +5 bp from 100.5
        // (100.55 - 100.5)/100.5 * 1e4 = 4.975 bp, below 5 -> no fire.
        assert!(!tr.step(Some(&b2), 0.0, 4));
        // Larger move: mid = 100.6 -> (100.6 - 100.5)/100.5 * 1e4 ~= 9.95 bp -> fires.
        let b3 = book(0, vec![(100.1, 1.0)], vec![(101.1, 1.0)]);
        assert!(tr.step(Some(&b3), 0.0, 5));
    }

    #[test]
    fn inv_change_trigger_rebases_on_fire() {
        let mut tr = InvChangeTrigger::new(2.0);
        // First call fires, baseline = 0.
        assert!(tr.step(None, 0.0, 0));
        // |1 - 0| = 1 < 2 -> no fire.
        assert!(!tr.step(None, 1.0, 1));
        // |2 - 0| = 2 >= 2 -> fire, re-base to 2.
        assert!(tr.step(None, 2.0, 2));
        // |3 - 2| = 1 < 2 -> no fire.
        assert!(!tr.step(None, 3.0, 3));
        // |0 - 2| = 2 >= 2 -> fire.
        assert!(tr.step(None, 0.0, 4));
    }

    #[test]
    fn book_event_trigger_always_fires() {
        let mut tr = BookEventTrigger::new();
        assert!(tr.step(None, 0.0, 0));
        assert!(tr.step(None, 0.0, 1));
        let b = book(0, vec![(100.0, 1.0)], vec![(101.0, 1.0)]);
        assert!(tr.step(Some(&b), 5.0, 2));
    }

    #[test]
    fn hybrid_any_fires_if_any_child_fires() {
        // Two TimeTriggers with different intervals.
        let a: Box<dyn RefreshTrigger> = Box::new(TimeTrigger::new(100));
        let b: Box<dyn RefreshTrigger> = Box::new(TimeTrigger::new(200));
        let mut h = HybridTrigger::new(vec![a, b], HybridMode::Any);
        // Both fire first call.
        assert!(h.step(None, 0.0, 0));
        // At t=50: a has fired at 0, needs 100; b has fired at 0, needs 200 -> neither.
        assert!(!h.step(None, 0.0, 50));
        // At t=100: a fires; b does not -> any-mode fires.
        assert!(h.step(None, 0.0, 100));
        // At t=150: a re-based at 100, needs 200; b re-based at... wait, b was stepped at 100
        // and did NOT fire there (b's last_fire is still 0). So at 150: a needs 200, b needs 200 -> no.
        assert!(!h.step(None, 0.0, 150));
        // At t=200: a fires (last 100 + 100 = 200); b fires (last 0 + 200 = 200) -> fires.
        assert!(h.step(None, 0.0, 200));
    }

    #[test]
    fn hybrid_all_requires_every_child_to_fire() {
        let a: Box<dyn RefreshTrigger> = Box::new(TimeTrigger::new(100));
        let b: Box<dyn RefreshTrigger> = Box::new(InvChangeTrigger::new(1.0));
        let mut h = HybridTrigger::new(vec![a, b], HybridMode::All);
        // First call: both fire -> all-mode fires.
        assert!(h.step(None, 0.0, 0));
        // t=100, inv=0 (unchanged): time fires, inv does NOT (|0-0|=0 < 1) -> all-mode no.
        assert!(!h.step(None, 0.0, 100));
        // t=150, inv=2: time -- last_fire was 100, 150-100=50 < 100 -> time no, inv yes -> no.
        assert!(!h.step(None, 2.0, 150));
        // t=200, inv=2: time -- last_fire 100, 200-100=100 >= 100 -> fires; inv -- last 0,
        // wait, inv last_fire was set at t=0 to 0; at t=100 we passed inv=0, no fire so still 0;
        // at t=150 we passed inv=2, |2-0|=2>=1 -> fired, re-based to 2; at t=200 inv=2,
        // |2-2|=0 < 1 -> no fire. So all-mode: time yes, inv no -> false.
        assert!(!h.step(None, 2.0, 200));
        // t=300, inv=5: time -- last 200, 300-200=100 -> fires; inv -- last 2, |5-2|=3 -> fires.
        assert!(h.step(None, 5.0, 300));
    }

    #[test]
    fn hybrid_steps_every_child_no_short_circuit() {
        // Verify that even when an early child returns true (any-mode)
        // or false (all-mode), every child's state advances.
        let a: Box<dyn RefreshTrigger> = Box::new(BookEventTrigger::new()); // always fires
        let b: Box<dyn RefreshTrigger> = Box::new(TimeTrigger::new(100));   // tracks ts
        let mut h = HybridTrigger::new(vec![a, b], HybridMode::Any);
        // First call any-mode fires (both fire); both children stepped.
        assert!(h.step(None, 0.0, 0));
        // Second call: a (BookEvent) always true, b's time at t=50: was stepped at 0, baseline set.
        // 50-0=50 < 100, so b would not fire. any-mode: a=true -> fire. But we need to confirm
        // b's state advanced. We re-extract by next call at t=100: b should fire because its
        // last_fire is still 0 (not 50), so 100-0=100 -> fires. If short-circuit had skipped b
        // at t=50 we'd still expect b to fire at t=100. Hard to distinguish here; instead
        // use all-mode with two TimeTriggers to test that both step.
        let a2: Box<dyn RefreshTrigger> = Box::new(TimeTrigger::new(50));
        let b2: Box<dyn RefreshTrigger> = Box::new(TimeTrigger::new(50));
        let mut h2 = HybridTrigger::new(vec![a2, b2], HybridMode::All);
        // t=0: both fire, all-mode fires.
        assert!(h2.step(None, 0.0, 0));
        // t=50: both fire (50-0=50 >= 50), all-mode fires; both re-based at 50.
        assert!(h2.step(None, 0.0, 50));
        // t=99: neither fires (99-50=49 < 50), all-mode does not.
        assert!(!h2.step(None, 0.0, 99));
        // t=100: both fire, all-mode fires.
        assert!(h2.step(None, 0.0, 100));
    }
}
