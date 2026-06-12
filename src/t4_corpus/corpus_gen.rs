//! T4 corpus generator.
//!
//! Mirror of Python's `mmsim.t4_corpus.t4_corpus_gen`. Walks
//! `(combos × assets)` calling [`run_t4_combo`] for each combination
//! and writing per-combo CSV + JSON sidecars to
//! `{output_root}/{asset}/{combo_hash}.{csv|json}`.
//!
//! CSV-first by design: matches the framework's "Rust reads/writes
//! CSV; the parity script converts between formats as needed".
//!
//! Resumable via skip-if-exists; single-process serial.

#![cfg(feature = "t4-corpus")]

use std::collections::HashMap;
use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use crate::ingest::EventStream;

use super::combos::{combo_to_strategy_name, sample_combos, Combo};
use super::runner::{run_t4_combo, T4LegRow, T4Metrics, T4RunResult, LEG_COLS};

#[derive(Debug, Clone)]
pub struct T4CorpusResult {
    pub asset: String,
    pub combo_name: String,
    pub n_fills: usize,
    pub n_maker_fills: usize,
    pub n_taker_fills: usize,
    pub total_cost: f64,
    pub net_pnl: f64,
    pub csv_path: PathBuf,
    pub sidecar_path: PathBuf,
}

fn combo_hash(combo: &Combo, params: &HashMap<String, f64>, asset: &str) -> String {
    // Stable 12-hex-char hash based on a JSON-ish canonical string of
    // (combo, sorted-params, asset). Pure FNV-1a for compactness; only
    // needs collision-resistance within one user's corpus so a 64-bit
    // hash truncated to 12 hex chars is plenty.
    let mut entries: Vec<(String, f64)> = params.iter()
        .map(|(k, v)| (k.clone(), *v)).collect();
    entries.sort_by(|a, b| a.0.cmp(&b.0));
    let mut buf = String::new();
    buf.push_str(combo.quoting_model);
    buf.push('|');
    buf.push_str(combo.inventory_penalty);
    buf.push('|');
    buf.push_str(combo.adverse_filter);
    buf.push('|');
    buf.push_str(combo.hedge_mode);
    buf.push('|');
    buf.push_str(combo.reference_price);
    buf.push('|');
    buf.push_str(combo.quote_shape);
    buf.push('|');
    buf.push_str(combo.refresh_trigger);
    buf.push('|');
    for (k, v) in &entries {
        buf.push_str(k);
        buf.push('=');
        buf.push_str(&format!("{:.12e}", v));
        buf.push(',');
    }
    buf.push_str("asset=");
    buf.push_str(asset);
    let mut h: u64 = 0xcbf29ce484222325;
    for b in buf.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    format!("{:012x}", h)
}

fn write_csv(path: &Path, leg_rows: &[T4LegRow]) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut f = fs::File::create(path)?;
    writeln!(f, "{}", LEG_COLS.join(","))?;
    for r in leg_rows {
        writeln!(
            f,
            "{},{},{},{:.9},{:.9},{},{:.9},{:.9},{:.9},{:.9},{:.9},{},{}",
            r.fill_id, r.ts_ns, r.side, r.price, r.size, r.is_maker,
            r.notional, r.fee, r.slippage, r.gross_pnl, r.net_pnl,
            r.order_id, r.trade_group_id,
        )?;
    }
    Ok(())
}

fn write_sidecar(
    path: &Path,
    asset: &str,
    combo: &Combo,
    params: &HashMap<String, f64>,
    metrics: &T4Metrics,
) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut buf = String::new();
    buf.push_str("{\n");
    buf.push_str(&format!("  \"asset\": \"{}\",\n", asset));
    buf.push_str("  \"combo\": {\n");
    buf.push_str(&format!("    \"quoting_model\": \"{}\",\n", combo.quoting_model));
    buf.push_str(&format!("    \"inventory_penalty\": \"{}\",\n", combo.inventory_penalty));
    buf.push_str(&format!("    \"adverse_filter\": \"{}\",\n", combo.adverse_filter));
    buf.push_str(&format!("    \"hedge_mode\": \"{}\",\n", combo.hedge_mode));
    buf.push_str(&format!("    \"reference_price\": \"{}\",\n", combo.reference_price));
    buf.push_str(&format!("    \"quote_shape\": \"{}\",\n", combo.quote_shape));
    buf.push_str(&format!("    \"refresh_trigger\": \"{}\"\n", combo.refresh_trigger));
    buf.push_str("  },\n");
    buf.push_str("  \"params\": {\n");
    let mut keys: Vec<&String> = params.keys().collect();
    keys.sort();
    for (i, k) in keys.iter().enumerate() {
        let v = params.get(*k).unwrap();
        buf.push_str(&format!("    \"{}\": {:.12e}{}\n", k, v,
            if i + 1 < keys.len() { "," } else { "" }));
    }
    buf.push_str("  },\n");
    buf.push_str("  \"metrics\": {\n");
    buf.push_str(&format!("    \"n_fills\": {},\n", metrics.n_fills));
    buf.push_str(&format!("    \"n_maker_fills\": {},\n", metrics.n_maker_fills));
    buf.push_str(&format!("    \"n_taker_fills\": {},\n", metrics.n_taker_fills));
    buf.push_str(&format!("    \"total_fees\": {:.12e},\n", metrics.total_fees));
    buf.push_str(&format!("    \"total_slippage\": {:.12e},\n", metrics.total_slippage));
    buf.push_str(&format!("    \"total_cost\": {:.12e},\n", metrics.total_cost));
    buf.push_str(&format!("    \"gross_pnl\": {:.12e},\n", metrics.gross_pnl));
    buf.push_str(&format!("    \"net_pnl\": {:.12e}\n", metrics.net_pnl));
    buf.push_str("  },\n");
    buf.push_str(&format!("  \"n_fills\": {},\n", metrics.n_fills));
    buf.push_str("  \"schema_version\": 1\n");
    buf.push_str("}\n");
    fs::write(path, buf)?;
    Ok(())
}

/// Generate one corpus slice. `streams` is a per-asset preloaded
/// EventStream map; `output_root` is the CSV/JSON root.
pub fn generate_t4_corpus(
    n_combos: usize,
    seed: u64,
    streams: &HashMap<String, EventStream>,
    output_root: &Path,
    is_params: HashMap<String, f64>,
    skip_if_exists: bool,
) -> Vec<T4CorpusResult> {
    let combos = sample_combos(n_combos, seed);
    let mut results: Vec<T4CorpusResult> = Vec::new();

    for (asset, stream) in streams.iter() {
        for combo in &combos {
            let h = combo_hash(combo, &is_params, asset);
            let out_dir = output_root.join(asset);
            let p_path = out_dir.join(format!("{}.csv", h));
            let s_path = out_dir.join(format!("{}.json", h));
            if skip_if_exists && p_path.exists() && s_path.exists() {
                continue;
            }
            let rr: T4RunResult = run_t4_combo(
                combo.clone(), is_params.clone(), stream, asset);
            let _ = write_csv(&p_path, &rr.leg_rows);
            let _ = write_sidecar(&s_path, asset, combo, &is_params, &rr.metrics);
            results.push(T4CorpusResult {
                asset: asset.clone(),
                combo_name: combo_to_strategy_name(combo),
                n_fills: rr.n_fills,
                n_maker_fills: rr.n_maker_fills,
                n_taker_fills: rr.n_taker_fills,
                total_cost: rr.metrics.total_cost,
                net_pnl: rr.metrics.net_pnl,
                csv_path: p_path,
                sidecar_path: s_path,
            });
        }
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn combo_hash_is_stable() {
        let combo = Combo {
            quoting_model: "symmetric",
            inventory_penalty: "linear",
            adverse_filter: "none",
            hedge_mode: "none",
            reference_price: "mid",
            quote_shape: "single",
            refresh_trigger: "book_event",
        };
        let params: HashMap<String, f64> = HashMap::new();
        let h1 = combo_hash(&combo, &params, "BTCUSDT");
        let h2 = combo_hash(&combo, &params, "BTCUSDT");
        assert_eq!(h1, h2);
        let h3 = combo_hash(&combo, &params, "ETHUSDT");
        assert_ne!(h1, h3);
    }

    #[test]
    fn empty_streams_produces_no_results() {
        let streams: HashMap<String, EventStream> = HashMap::new();
        let tmp = std::env::temp_dir().join("t4_rs_test_corpus_empty");
        let r = generate_t4_corpus(5, 2026, &streams, &tmp, HashMap::new(), true);
        assert!(r.is_empty());
    }
}
