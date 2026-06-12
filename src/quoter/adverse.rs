//! Adverse-selection filter primitives.
//!
//! Mirror of Python's `mmsim.quoter.adverse`.  Six stateful filters
//! that suppress quoting when the order book or trade tape signals
//! that the market is currently moving against passive makers.  Each
//! filter exposes the same trio: `observe_trade`, `observe_book`,
//! `is_adverse(t_ns) -> bool`.
//!
//! The six:
//!   - `OFIFilter` — order-flow imbalance over a trailing window
//!   - `TradeToxicityFilter` — aggressor-side dominance share
//!   - `VolSurgeFilter` — realized-vol surge from trade-tape log-returns
//!   - `MicropriceDevFilter` — gap between microprice and mid (pure book)
//!   - `QueueImbalanceFilter` — TOB queue imbalance (pure book)
//!   - `HybridAdverseFilter` — any-of / all-of composition
//!
//! Causality contract (`<= t` only): every filter consumes only state
//! that was fed via `observe_*` at-or-before `t_ns`.  Trade-tape filters
//! keep a trailing-window `VecDeque` of observations; pure-book filters
//! store only the most-recent book.
//!
//! Note: `is_adverse` takes `&mut self` because trade-tape filters evict
//! expired entries from the trailing-window deque during the query.

#![cfg(feature = "quoter")]

use std::collections::VecDeque;

use crate::ingest::{Book, TradeEvent};

/// Formal adverse-filter trait.  Causality: every input is
/// state at-or-before `t_ns`.  `is_adverse` is `&mut self` so trade-tape
/// filters can evict expired window entries on query.
pub trait AdverseFilter {
    fn observe_trade(&mut self, trade: &TradeEvent);
    fn observe_book(&mut self, book: &Book);
    fn is_adverse(&mut self, t_ns: i64) -> bool;
}

// --------------------------------------------------------------------- //
// Trade-tape filters
// --------------------------------------------------------------------- //

/// Order-flow imbalance over a trailing window of trades.
/// `OFI = (buy_vol - sell_vol) / total_vol`.  Activates when
/// `|OFI| >= threshold`.  Aggressor side is taken from `trade.side`
/// (`+1` buy aggressor, `-1` sell aggressor, `0` ignored).
#[derive(Debug, Clone)]
pub struct OFIFilter {
    pub window_ns: i64,
    pub threshold: f64,
    // (ts_ns, side, size)
    buffer: VecDeque<(i64, i32, f64)>,
}

impl OFIFilter {
    pub fn new(window_ns: i64, threshold: f64) -> Self {
        if window_ns <= 0 {
            panic!("window_ns must be > 0");
        }
        if threshold < 0.0 {
            panic!("threshold must be >= 0");
        }
        Self { window_ns, threshold, buffer: VecDeque::new() }
    }

    fn evict(&mut self, t_ns: i64) {
        let cutoff = t_ns - self.window_ns;
        while let Some(&(ts, _, _)) = self.buffer.front() {
            if ts <= cutoff {
                self.buffer.pop_front();
            } else {
                break;
            }
        }
    }
}

impl AdverseFilter for OFIFilter {
    fn observe_trade(&mut self, trade: &TradeEvent) {
        if trade.side == 0 {
            return;
        }
        self.buffer.push_back((trade.ts_ns, trade.side, trade.size));
    }

    fn observe_book(&mut self, _book: &Book) {
        // book-blind
    }

    fn is_adverse(&mut self, t_ns: i64) -> bool {
        self.evict(t_ns);
        if self.buffer.is_empty() {
            return false;
        }
        let mut buy_vol = 0.0_f64;
        let mut sell_vol = 0.0_f64;
        for &(ts, side, sz) in &self.buffer {
            if ts <= t_ns {
                if side == 1 {
                    buy_vol += sz;
                } else if side == -1 {
                    sell_vol += sz;
                }
            }
        }
        let total = buy_vol + sell_vol;
        if total <= 0.0 {
            return false;
        }
        let ofi = (buy_vol - sell_vol) / total;
        ofi.abs() >= self.threshold
    }
}

/// Aggressor-side dominance over a trailing window.  Activates when
/// one side's volume share exceeds `threshold` (in `[0.5, 1.0]` — 0.5
/// means perfectly balanced, 1.0 means all on one side).
#[derive(Debug, Clone)]
pub struct TradeToxicityFilter {
    pub window_ns: i64,
    pub threshold: f64,
    buffer: VecDeque<(i64, i32, f64)>,
}

impl TradeToxicityFilter {
    pub fn new(window_ns: i64, threshold: f64) -> Self {
        if window_ns <= 0 {
            panic!("window_ns must be > 0");
        }
        if !(0.5..=1.0).contains(&threshold) {
            panic!("threshold must be in [0.5, 1.0]");
        }
        Self { window_ns, threshold, buffer: VecDeque::new() }
    }

    fn evict(&mut self, t_ns: i64) {
        let cutoff = t_ns - self.window_ns;
        while let Some(&(ts, _, _)) = self.buffer.front() {
            if ts <= cutoff {
                self.buffer.pop_front();
            } else {
                break;
            }
        }
    }
}

impl AdverseFilter for TradeToxicityFilter {
    fn observe_trade(&mut self, trade: &TradeEvent) {
        if trade.side == 0 {
            return;
        }
        self.buffer.push_back((trade.ts_ns, trade.side, trade.size));
    }

    fn observe_book(&mut self, _book: &Book) {}

    fn is_adverse(&mut self, t_ns: i64) -> bool {
        self.evict(t_ns);
        if self.buffer.is_empty() {
            return false;
        }
        let mut buy_vol = 0.0_f64;
        let mut sell_vol = 0.0_f64;
        for &(ts, side, sz) in &self.buffer {
            if ts <= t_ns {
                if side == 1 {
                    buy_vol += sz;
                } else if side == -1 {
                    sell_vol += sz;
                }
            }
        }
        let total = buy_vol + sell_vol;
        if total <= 0.0 {
            return false;
        }
        let max_share = buy_vol.max(sell_vol) / total;
        max_share >= self.threshold
    }
}

/// Realized-vol surge from trailing trade prices.  Activates when the
/// rolling std of log-returns exceeds `threshold_bp` (in basis points).
/// Needs at least 3 trades in the window before it can fire.
#[derive(Debug, Clone)]
pub struct VolSurgeFilter {
    pub window_ns: i64,
    pub threshold_bp: f64,
    // (ts_ns, price)
    buffer: VecDeque<(i64, f64)>,
}

impl VolSurgeFilter {
    pub fn new(window_ns: i64, threshold_bp: f64) -> Self {
        if window_ns <= 0 {
            panic!("window_ns must be > 0");
        }
        if threshold_bp < 0.0 {
            panic!("threshold_bp must be >= 0");
        }
        Self { window_ns, threshold_bp, buffer: VecDeque::new() }
    }

    fn evict(&mut self, t_ns: i64) {
        let cutoff = t_ns - self.window_ns;
        while let Some(&(ts, _)) = self.buffer.front() {
            if ts <= cutoff {
                self.buffer.pop_front();
            } else {
                break;
            }
        }
    }
}

impl AdverseFilter for VolSurgeFilter {
    fn observe_trade(&mut self, trade: &TradeEvent) {
        self.buffer.push_back((trade.ts_ns, trade.price));
    }

    fn observe_book(&mut self, _book: &Book) {}

    fn is_adverse(&mut self, t_ns: i64) -> bool {
        self.evict(t_ns);
        let prices: Vec<f64> = self
            .buffer
            .iter()
            .filter_map(|&(ts, px)| if ts <= t_ns && px > 0.0 { Some(px) } else { None })
            .collect();
        if prices.len() < 3 {
            return false;
        }
        let rets: Vec<f64> = (1..prices.len())
            .map(|i| (prices[i] / prices[i - 1]).ln())
            .collect();
        let n = rets.len() as f64;
        let mean: f64 = rets.iter().sum::<f64>() / n;
        let var: f64 = rets.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
        let std_bp = var.sqrt() * 1e4;
        std_bp >= self.threshold_bp
    }
}

// --------------------------------------------------------------------- //
// Pure-book filters (no trade-tape state needed)
// --------------------------------------------------------------------- //

/// Activates when `|microprice - mid| / mid * 1e4 >= threshold_bp`.
/// Pure book function on the most-recent observed book.
#[derive(Debug, Clone)]
pub struct MicropriceDevFilter {
    pub threshold_bp: f64,
    last_book: Option<Book>,
}

impl MicropriceDevFilter {
    pub fn new(threshold_bp: f64) -> Self {
        if threshold_bp < 0.0 {
            panic!("threshold_bp must be >= 0");
        }
        Self { threshold_bp, last_book: None }
    }
}

impl AdverseFilter for MicropriceDevFilter {
    fn observe_trade(&mut self, _trade: &TradeEvent) {}

    fn observe_book(&mut self, book: &Book) {
        self.last_book = Some(book.clone());
    }

    fn is_adverse(&mut self, _t_ns: i64) -> bool {
        let b = match &self.last_book {
            Some(b) => b,
            None => return false,
        };
        if b.bids.is_empty() || b.asks.is_empty() {
            return false;
        }
        let (bid_px, bid_sz) = b.bids[0];
        let (ask_px, ask_sz) = b.asks[0];
        if bid_sz + ask_sz <= 0.0 {
            return false;
        }
        let mid = (bid_px + ask_px) / 2.0;
        if mid <= 0.0 {
            return false;
        }
        let half_spread = (ask_px - bid_px) / 2.0;
        let imb = (bid_sz - ask_sz) / (bid_sz + ask_sz);
        let microprice = mid + imb * half_spread;
        (microprice - mid).abs() / mid * 1e4 >= self.threshold_bp
    }
}

/// Activates when the absolute queue imbalance at TOB exceeds
/// `threshold`: `|bid_sz - ask_sz| / (bid_sz + ask_sz) >= threshold`.
#[derive(Debug, Clone)]
pub struct QueueImbalanceFilter {
    pub threshold: f64,
    last_book: Option<Book>,
}

impl QueueImbalanceFilter {
    pub fn new(threshold: f64) -> Self {
        if !(0.0..=1.0).contains(&threshold) {
            panic!("threshold must be in [0.0, 1.0]");
        }
        Self { threshold, last_book: None }
    }
}

impl AdverseFilter for QueueImbalanceFilter {
    fn observe_trade(&mut self, _trade: &TradeEvent) {}

    fn observe_book(&mut self, book: &Book) {
        self.last_book = Some(book.clone());
    }

    fn is_adverse(&mut self, _t_ns: i64) -> bool {
        let b = match &self.last_book {
            Some(b) => b,
            None => return false,
        };
        if b.bids.is_empty() || b.asks.is_empty() {
            return false;
        }
        let bid_sz = b.bids[0].1;
        let ask_sz = b.asks[0].1;
        let total = bid_sz + ask_sz;
        if total <= 0.0 {
            return false;
        }
        let imb = (bid_sz - ask_sz).abs() / total;
        imb >= self.threshold
    }
}

// --------------------------------------------------------------------- //
// Composition
// --------------------------------------------------------------------- //

/// Composition mode for `HybridAdverseFilter`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HybridAdverseMode {
    Any,
    All,
}

/// Composes multiple adverse filters via any-of or all-of.
///
/// No short-circuit: every child's `is_adverse` is invoked exactly
/// once per call.  This mirrors the Python sibling and keeps state
/// advancement deterministic (symmetric with the triggers convention).
pub struct HybridAdverseFilter {
    pub children: Vec<Box<dyn AdverseFilter>>,
    pub mode: HybridAdverseMode,
}

impl HybridAdverseFilter {
    pub fn new(children: Vec<Box<dyn AdverseFilter>>, mode: HybridAdverseMode) -> Self {
        Self { children, mode }
    }
}

impl AdverseFilter for HybridAdverseFilter {
    fn observe_trade(&mut self, trade: &TradeEvent) {
        for c in self.children.iter_mut() {
            c.observe_trade(trade);
        }
    }

    fn observe_book(&mut self, book: &Book) {
        for c in self.children.iter_mut() {
            c.observe_book(book);
        }
    }

    fn is_adverse(&mut self, t_ns: i64) -> bool {
        // No short-circuit: collect every child's vote first.
        let votes: Vec<bool> = self
            .children
            .iter_mut()
            .map(|c| c.is_adverse(t_ns))
            .collect();
        match self.mode {
            HybridAdverseMode::Any => votes.iter().any(|&x| x),
            HybridAdverseMode::All => !votes.is_empty() && votes.iter().all(|&x| x),
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

    fn trade(ts: i64, side: i32, price: f64, size: f64) -> TradeEvent {
        TradeEvent {
            ts_ns: ts,
            recv_ns: ts,
            symbol: "X".to_string(),
            venue: "V".to_string(),
            price,
            size,
            side,
        }
    }

    #[test]
    fn ofi_filter_fires_on_one_sided_flow() {
        let mut f = OFIFilter::new(1_000, 0.5);
        // Empty -> no fire.
        assert!(!f.is_adverse(0));
        // One buy trade of size 1, window includes it -> OFI = 1.0 >= 0.5 -> fire.
        f.observe_trade(&trade(100, 1, 100.0, 1.0));
        assert!(f.is_adverse(200));
        // Add matching sell -> total balanced, OFI = 0 -> no fire.
        f.observe_trade(&trade(150, -1, 100.0, 1.0));
        assert!(!f.is_adverse(200));
        // Window evicts both (cutoff = t - window_ns = 1200 - 1000 = 200 >= 150 >= 100)
        // -> empty -> no fire.
        assert!(!f.is_adverse(1200));
        // side=0 trades are ignored.
        f.observe_trade(&trade(1300, 0, 100.0, 5.0));
        assert!(!f.is_adverse(1400));
    }

    #[test]
    fn trade_toxicity_filter_fires_above_share() {
        let mut f = TradeToxicityFilter::new(1_000, 0.8);
        // 4 buys of size 1, 1 sell of size 1 -> buy share = 0.8 -> fire.
        for ts in [100, 110, 120, 130].iter() {
            f.observe_trade(&trade(*ts, 1, 100.0, 1.0));
        }
        f.observe_trade(&trade(140, -1, 100.0, 1.0));
        assert!(f.is_adverse(200));
        // Add another sell -> buy share = 4/6 ~ 0.667 -> no fire.
        f.observe_trade(&trade(150, -1, 100.0, 1.0));
        assert!(!f.is_adverse(200));
    }

    #[test]
    fn vol_surge_needs_three_trades_then_fires() {
        let mut f = VolSurgeFilter::new(1_000, 1.0); // 1 bp threshold
        // Two trades insufficient.
        f.observe_trade(&trade(100, 1, 100.0, 1.0));
        f.observe_trade(&trade(110, 1, 100.0, 1.0));
        assert!(!f.is_adverse(200));
        // Add a third with a price jump -> log-return non-zero -> fire.
        f.observe_trade(&trade(120, 1, 101.0, 1.0));
        assert!(f.is_adverse(200));
        // Calm regime (flat prices) -> std=0 -> no fire.
        let mut g = VolSurgeFilter::new(1_000, 1.0);
        for ts in [100, 110, 120, 130].iter() {
            g.observe_trade(&trade(*ts, 1, 100.0, 1.0));
        }
        assert!(!g.is_adverse(200));
    }

    #[test]
    fn microprice_dev_filter_pure_book() {
        let mut f = MicropriceDevFilter::new(5.0);
        // No book yet.
        assert!(!f.is_adverse(0));
        // Balanced book -> microprice == mid -> deviation 0 -> no fire.
        let bal = book(0, vec![(100.0, 1.0)], vec![(101.0, 1.0)]);
        f.observe_book(&bal);
        assert!(!f.is_adverse(0));
        // Skewed book: bid_sz=9, ask_sz=1 -> imb=0.8, half_spread=0.5,
        // microprice = 100.5 + 0.8*0.5 = 100.9; |100.9-100.5|/100.5*1e4 ~= 39.8 bp -> fire.
        let skewed = book(1, vec![(100.0, 9.0)], vec![(101.0, 1.0)]);
        f.observe_book(&skewed);
        assert!(f.is_adverse(1));
        // Empty side -> no fire.
        let empty_ask = book(2, vec![(100.0, 1.0)], vec![]);
        f.observe_book(&empty_ask);
        assert!(!f.is_adverse(2));
    }

    #[test]
    fn queue_imbalance_filter_pure_book() {
        let mut f = QueueImbalanceFilter::new(0.6);
        assert!(!f.is_adverse(0));
        // Balanced -> imb=0 -> no fire.
        f.observe_book(&book(0, vec![(100.0, 5.0)], vec![(101.0, 5.0)]));
        assert!(!f.is_adverse(0));
        // 8 vs 2 -> imb = 6/10 = 0.6 -> fire.
        f.observe_book(&book(1, vec![(100.0, 8.0)], vec![(101.0, 2.0)]));
        assert!(f.is_adverse(1));
        // Zero total -> no fire.
        f.observe_book(&book(2, vec![(100.0, 0.0)], vec![(101.0, 0.0)]));
        assert!(!f.is_adverse(2));
    }

    #[test]
    fn hybrid_any_fires_if_any_child_fires() {
        let a: Box<dyn AdverseFilter> = Box::new(QueueImbalanceFilter::new(0.6));
        let b: Box<dyn AdverseFilter> = Box::new(OFIFilter::new(1_000, 0.5));
        let mut h = HybridAdverseFilter::new(vec![a, b], HybridAdverseMode::Any);
        // No state -> neither fires.
        assert!(!h.is_adverse(0));
        // Feed a one-sided trade -> OFI fires; queue still neutral (no book).
        h.observe_trade(&trade(100, 1, 100.0, 1.0));
        assert!(h.is_adverse(200));
        // Feed a balanced book and balanced trade -> neither fires.
        let mut h2 = HybridAdverseFilter::new(
            vec![
                Box::new(QueueImbalanceFilter::new(0.6)),
                Box::new(OFIFilter::new(1_000, 0.5)),
            ],
            HybridAdverseMode::Any,
        );
        h2.observe_book(&book(0, vec![(100.0, 5.0)], vec![(101.0, 5.0)]));
        h2.observe_trade(&trade(100, 1, 100.0, 1.0));
        h2.observe_trade(&trade(110, -1, 100.0, 1.0));
        assert!(!h2.is_adverse(200));
    }

    #[test]
    fn hybrid_all_requires_every_child_to_fire() {
        let a: Box<dyn AdverseFilter> = Box::new(QueueImbalanceFilter::new(0.6));
        let b: Box<dyn AdverseFilter> = Box::new(OFIFilter::new(1_000, 0.5));
        let mut h = HybridAdverseFilter::new(vec![a, b], HybridAdverseMode::All);
        // Skewed book fires queue; no trades -> OFI does not fire -> all-mode no.
        h.observe_book(&book(0, vec![(100.0, 8.0)], vec![(101.0, 2.0)]));
        assert!(!h.is_adverse(0));
        // Add a one-sided trade -> OFI fires too -> all-mode fires.
        h.observe_trade(&trade(100, 1, 100.0, 1.0));
        assert!(h.is_adverse(200));
    }

    #[test]
    #[should_panic(expected = "window_ns must be > 0")]
    fn ofi_filter_rejects_nonpositive_window() {
        let _ = OFIFilter::new(0, 0.5);
    }

    #[test]
    #[should_panic(expected = "threshold must be in [0.5, 1.0]")]
    fn trade_toxicity_rejects_low_threshold() {
        let _ = TradeToxicityFilter::new(1_000, 0.3);
    }

    #[test]
    #[should_panic(expected = "threshold must be in [0.0, 1.0]")]
    fn queue_imbalance_rejects_out_of_range_threshold() {
        let _ = QueueImbalanceFilter::new(1.5);
    }
}
