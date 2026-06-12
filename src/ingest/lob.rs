//! L2 book + tape loader & book-reconstruction primitives.
//!
//! Causality contract (matches Python sibling):
//!   `reconstruct_book_at(stream, t_ns)` consults only events with
//!   `ts_ns <= t_ns`.  Polluting events strictly past `t_ns` does
//!   not change the return value.  Verified by the cross-language
//!   parity script's leak-pollute battery.

#![cfg(feature = "ingest")]

use std::path::Path;

#[derive(Debug, Clone)]
pub struct SnapshotEvent {
    pub ts_ns: i64,
    pub recv_ns: i64,
    pub symbol: String,
    pub venue: String,
    pub depth: u32,
    /// Best-first.  Each level is `(price, size)`.
    pub bids: Vec<(f64, f64)>,
    pub asks: Vec<(f64, f64)>,
}

#[derive(Debug, Clone)]
pub struct TradeEvent {
    pub ts_ns: i64,
    pub recv_ns: i64,
    pub symbol: String,
    pub venue: String,
    pub price: f64,
    pub size: f64,
    /// `+1` buy aggressor, `-1` sell aggressor, `0` unknown.
    pub side: i32,
}

#[derive(Debug, Clone)]
pub enum Event {
    Snapshot(SnapshotEvent),
    Trade(TradeEvent),
}

impl Event {
    pub fn ts_ns(&self) -> i64 {
        match self {
            Event::Snapshot(s) => s.ts_ns,
            Event::Trade(t) => t.ts_ns,
        }
    }
    pub fn recv_ns(&self) -> i64 {
        match self {
            Event::Snapshot(s) => s.recv_ns,
            Event::Trade(t) => t.recv_ns,
        }
    }
    /// Same tiebreaker as the Python sibling: snapshot < trade at
    /// identical (ts_ns, recv_ns).
    fn kind_rank(&self) -> u8 {
        match self {
            Event::Snapshot(_) => 0,
            Event::Trade(_) => 1,
        }
    }
}

pub type EventStream = Vec<Event>;

#[derive(Debug, Clone, PartialEq)]
pub struct Book {
    pub ts_ns: i64,
    pub bids: Vec<(f64, f64)>,
    pub asks: Vec<(f64, f64)>,
}

impl Book {
    pub fn best_bid(&self) -> Option<f64> {
        self.bids.first().map(|b| b.0)
    }
    pub fn best_ask(&self) -> Option<f64> {
        self.asks.first().map(|a| a.0)
    }
    pub fn mid(&self) -> Option<f64> {
        match (self.best_bid(), self.best_ask()) {
            (Some(b), Some(a)) => Some((b + a) / 2.0),
            _ => None,
        }
    }
}

// --------------------------------------------------------------------- //
// CSV loaders
// --------------------------------------------------------------------- //

fn parse_snapshot_csv<P: AsRef<Path>>(path: P) -> Result<Vec<SnapshotEvent>, String> {
    let mut rdr = csv::ReaderBuilder::new()
        .from_path(path.as_ref())
        .map_err(|e| format!("snapshots: open {:?}: {}", path.as_ref(), e))?;
    let headers: Vec<String> = rdr
        .headers()
        .map_err(|e| format!("snapshots: headers: {}", e))?
        .iter()
        .map(|s| s.to_string())
        .collect();

    // Map flat columns once: ts_ns, recv_ns, symbol, venue, depth,
    // bid_px_0, bid_sz_0, ..., ask_px_0, ask_sz_0, ...
    let pos = |name: &str| -> Result<usize, String> {
        headers
            .iter()
            .position(|h| h == name)
            .ok_or_else(|| format!("snapshots: missing column '{}'", name))
    };
    let ts_idx = pos("ts_ns")?;
    let recv_idx = pos("recv_ns")?;
    let sym_idx = pos("symbol")?;
    let venue_idx = pos("venue")?;
    let depth_idx = pos("depth")?;

    // Discover depth N from the wide columns.  Robust to depth changing
    // between fixtures: scan headers for the highest bid_px_K present.
    let mut depth_n: usize = 0;
    for h in &headers {
        if let Some(rest) = h.strip_prefix("bid_px_") {
            if let Ok(k) = rest.parse::<usize>() {
                depth_n = depth_n.max(k + 1);
            }
        }
    }
    if depth_n == 0 {
        return Err("snapshots: no bid_px_* columns found".into());
    }
    let bid_px_idx: Vec<usize> = (0..depth_n)
        .map(|k| pos(&format!("bid_px_{}", k)))
        .collect::<Result<_, _>>()?;
    let bid_sz_idx: Vec<usize> = (0..depth_n)
        .map(|k| pos(&format!("bid_sz_{}", k)))
        .collect::<Result<_, _>>()?;
    let ask_px_idx: Vec<usize> = (0..depth_n)
        .map(|k| pos(&format!("ask_px_{}", k)))
        .collect::<Result<_, _>>()?;
    let ask_sz_idx: Vec<usize> = (0..depth_n)
        .map(|k| pos(&format!("ask_sz_{}", k)))
        .collect::<Result<_, _>>()?;

    let mut out: Vec<SnapshotEvent> = Vec::new();
    for (row, rec) in rdr.records().enumerate() {
        let rec = rec.map_err(|e| format!("snapshots: row {}: {}", row, e))?;
        let g = |i: usize| rec.get(i).unwrap_or("");
        let ts_ns: i64 = g(ts_idx).parse()
            .map_err(|e| format!("snapshots: row {} ts_ns: {}", row, e))?;
        let recv_ns: i64 = g(recv_idx).parse()
            .map_err(|e| format!("snapshots: row {} recv_ns: {}", row, e))?;
        let depth: u32 = g(depth_idx).parse()
            .map_err(|e| format!("snapshots: row {} depth: {}", row, e))?;

        let mut bids = Vec::with_capacity(depth_n);
        for k in 0..depth_n {
            let p: f64 = g(bid_px_idx[k]).parse()
                .map_err(|e| format!("snapshots: row {} bid_px_{}: {}", row, k, e))?;
            let s: f64 = g(bid_sz_idx[k]).parse()
                .map_err(|e| format!("snapshots: row {} bid_sz_{}: {}", row, k, e))?;
            bids.push((p, s));
        }
        let mut asks = Vec::with_capacity(depth_n);
        for k in 0..depth_n {
            let p: f64 = g(ask_px_idx[k]).parse()
                .map_err(|e| format!("snapshots: row {} ask_px_{}: {}", row, k, e))?;
            let s: f64 = g(ask_sz_idx[k]).parse()
                .map_err(|e| format!("snapshots: row {} ask_sz_{}: {}", row, k, e))?;
            asks.push((p, s));
        }
        out.push(SnapshotEvent {
            ts_ns,
            recv_ns,
            symbol: g(sym_idx).to_string(),
            venue: g(venue_idx).to_string(),
            depth,
            bids,
            asks,
        });
    }
    Ok(out)
}

fn parse_trades_csv<P: AsRef<Path>>(path: P) -> Result<Vec<TradeEvent>, String> {
    let mut rdr = csv::ReaderBuilder::new()
        .from_path(path.as_ref())
        .map_err(|e| format!("trades: open {:?}: {}", path.as_ref(), e))?;
    let headers: Vec<String> = rdr
        .headers()
        .map_err(|e| format!("trades: headers: {}", e))?
        .iter()
        .map(|s| s.to_string())
        .collect();
    let pos = |n: &str| -> Result<usize, String> {
        headers
            .iter()
            .position(|h| h == n)
            .ok_or_else(|| format!("trades: missing column '{}'", n))
    };
    let ts = pos("ts_ns")?;
    let recv = pos("recv_ns")?;
    let sym = pos("symbol")?;
    let ven = pos("venue")?;
    let px = pos("price")?;
    let sz = pos("size")?;
    let side = pos("side")?;

    let mut out: Vec<TradeEvent> = Vec::new();
    for (row, rec) in rdr.records().enumerate() {
        let rec = rec.map_err(|e| format!("trades: row {}: {}", row, e))?;
        let g = |i: usize| rec.get(i).unwrap_or("");
        out.push(TradeEvent {
            ts_ns: g(ts).parse()
                .map_err(|e| format!("trades: row {} ts_ns: {}", row, e))?,
            recv_ns: g(recv).parse()
                .map_err(|e| format!("trades: row {} recv_ns: {}", row, e))?,
            symbol: g(sym).to_string(),
            venue: g(ven).to_string(),
            price: g(px).parse()
                .map_err(|e| format!("trades: row {} price: {}", row, e))?,
            size: g(sz).parse()
                .map_err(|e| format!("trades: row {} size: {}", row, e))?,
            side: g(side).parse()
                .map_err(|e| format!("trades: row {} side: {}", row, e))?,
        });
    }
    Ok(out)
}

pub fn load_lob<P: AsRef<Path>>(
    snapshots_csv: P,
    trades_csv: P,
    symbol_filter: Option<&str>,
    venue_filter: Option<&str>,
) -> Result<EventStream, String> {
    let mut snaps = parse_snapshot_csv(snapshots_csv)?;
    let mut trades = parse_trades_csv(trades_csv)?;
    if let Some(sym) = symbol_filter {
        snaps.retain(|s| s.symbol == sym);
        trades.retain(|t| t.symbol == sym);
    }
    if let Some(ven) = venue_filter {
        snaps.retain(|s| s.venue == ven);
        trades.retain(|t| t.venue == ven);
    }
    let mut merged: EventStream = Vec::with_capacity(snaps.len() + trades.len());
    for s in snaps {
        merged.push(Event::Snapshot(s));
    }
    for t in trades {
        merged.push(Event::Trade(t));
    }
    // Stable sort by (ts_ns, recv_ns, kind_rank).  Snapshot before
    // trade at identical timestamps.
    merged.sort_by(|a, b| {
        (a.ts_ns(), a.recv_ns(), a.kind_rank())
            .cmp(&(b.ts_ns(), b.recv_ns(), b.kind_rank()))
    });
    Ok(merged)
}

pub fn reconstruct_book_at(stream: &[Event], t_ns: i64) -> Option<Book> {
    let mut last: Option<&SnapshotEvent> = None;
    for ev in stream {
        if ev.ts_ns() > t_ns {
            break;
        }
        if let Event::Snapshot(s) = ev {
            last = Some(s);
        }
    }
    last.map(|s| Book {
        ts_ns: s.ts_ns,
        bids: s.bids.clone(),
        asks: s.asks.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snap(ts: i64, bids: Vec<(f64, f64)>, asks: Vec<(f64, f64)>) -> Event {
        Event::Snapshot(SnapshotEvent {
            ts_ns: ts,
            recv_ns: ts,
            symbol: "BTC-USDT".into(),
            venue: "binance".into(),
            depth: bids.len() as u32,
            bids,
            asks,
        })
    }

    fn trade(ts: i64, px: f64, side: i32) -> Event {
        Event::Trade(TradeEvent {
            ts_ns: ts,
            recv_ns: ts,
            symbol: "BTC-USDT".into(),
            venue: "binance".into(),
            price: px,
            size: 1.0,
            side,
        })
    }

    #[test]
    fn reconstruct_returns_most_recent_snapshot() {
        let s0 = snap(100, vec![(80000.0, 1.0)], vec![(80001.0, 1.0)]);
        let s1 = snap(200, vec![(80050.0, 1.0)], vec![(80051.0, 1.0)]);
        let stream = vec![s0, trade(150, 80000.5, 1), s1];
        let book_150 = reconstruct_book_at(&stream, 150).unwrap();
        assert_eq!(book_150.ts_ns, 100);
        assert_eq!(book_150.best_bid(), Some(80000.0));
        let book_250 = reconstruct_book_at(&stream, 250).unwrap();
        assert_eq!(book_250.ts_ns, 200);
        assert_eq!(book_250.best_bid(), Some(80050.0));
    }

    #[test]
    fn reconstruct_returns_none_before_first_snapshot() {
        let stream = vec![snap(100, vec![(80000.0, 1.0)], vec![(80001.0, 1.0)])];
        assert!(reconstruct_book_at(&stream, 50).is_none());
        assert!(reconstruct_book_at(&stream, 100).is_some());
    }

    #[test]
    fn reconstruct_no_lookahead_under_pollution() {
        // Pollute snapshots strictly past T; book at T should be unchanged.
        let s0 = snap(100, vec![(80000.0, 1.0)], vec![(80001.0, 1.0)]);
        let s1 = snap(200, vec![(80050.0, 1.0)], vec![(80051.0, 1.0)]);
        let s2 = snap(300, vec![(99999.0, 1.0)], vec![(99999.0, 1.0)]);  // garbage
        let clean = reconstruct_book_at(&[s0.clone(), s1.clone()], 250).unwrap();
        let polluted = reconstruct_book_at(&[s0, s1, s2], 250).unwrap();
        assert_eq!(clean, polluted);
    }
}
