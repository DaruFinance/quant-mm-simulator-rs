//! T4 strategy composer.
//!
//! Mirror of Python's `mmsim.t4_corpus.t4_composer`. Given a structural
//! [`Combo`] and an IS-params map, produces a [`T4StrategyConfig`]
//! bundle holding:
//!
//!   - a boxed [`Quoter`] (from `crate::models`) implementing the
//!     chosen quoting model, parameterized by IS axes
//!   - an inventory-penalty closure (returns [`Skew`])
//!   - an optional boxed [`AdverseFilter`]
//!   - an optional [`HedgeEngine`]
//!   - a boxed [`RefreshTrigger`]
//!   - a shape-spec enum (one of the 5 `*Spec` types)
//!
//! The composer is routing only — every primitive it returns comes
//! from the existing engine modules. The runner uses these together
//! to drive `run_sim_with_quoter`.

#![cfg(feature = "t4-corpus")]

use std::collections::HashMap;

use crate::hedge::engine::HedgeEngine;
use crate::models::{
    AvellanedaStoikovQuoter, CarteaJaimungalQuoter, FairAnchoredQuoter, GLFTQuoter,
    HoStollQuoter, LadderQuoter, MicropriceSkewQuoter, SymmetricQuoter,
};
use crate::quoter::{
    adverse::{
        AdverseFilter, HybridAdverseFilter, HybridAdverseMode, MicropriceDevFilter,
        OFIFilter, QueueImbalanceFilter, TradeToxicityFilter, VolSurgeFilter,
    },
    inv_penalty::{self, Skew},
    shapes::{DynamicDepthSpec, GeometricSpec, LadderSpec, PairedSpec, SingleSpec},
    triggers::{
        BookEventTrigger, HybridMode, HybridTrigger, InvChangeTrigger,
        MidMoveTrigger, RefreshTrigger, TimeTrigger,
    },
    refprice::{EWMAFairTracker, VWAPTracker},
    Quoter,
};

use crate::ingest::{Book, TradeEvent};

use super::combos::Combo;

/// Default IS-axis values used when a key is missing from `params`.
fn default_param(key: &str) -> f64 {
    match key {
        "gamma" => 0.5,
        "k" => 1.5,
        "horizon_ns" => 3.6e12,
        "inventory_cap" => 0.05,
        "refresh_interval_ns" => 1e9,
        "spread_floor" => 5.0,
        "filter_threshold" => 0.3,
        "quote_size" => 0.001,
        "vol_window_ns" => 1e10,
        "ladder_step" => 2.5,
        "ladder_n_levels" => 3.0,
        "geometric_ratio" => 1.5,
        "ewma_half_life_ns" => 5e9,
        "vwap_window_ns" => 5e9,
        "ho_stoll_alpha" => 10.0,
        "ho_stoll_beta" => 0.05,
        "cj_kappa" => 0.0,
        "glft_A" => 140.0,
        "adverse_window_ns" => 5e9,
        "adverse_vol_threshold_bp" => 50.0,
        "hedge_threshold" => 0.05,
        "trigger_inv_change" => 0.001,
        "trigger_mid_move_bp" => 5.0,
        _ => panic!("unknown default param: {}", key),
    }
}

fn get_param(params: &HashMap<String, f64>, key: &str) -> f64 {
    params.get(key).copied().unwrap_or_else(|| default_param(key))
}

// --------------------------------------------------------------------- //
// Reference-price wrapper
// --------------------------------------------------------------------- //

/// Reference-price evaluator with optional stateful trackers.
///
/// Mirrors Python's `_RefPriceFn` adapter. Pure book-only refs (mid /
/// microprice / weighted_mid) evaluate in-line; stateful trackers
/// (ewma_fair / vwap) hold internal state on the struct.
pub struct RefPriceFn {
    pub name: String,
    ewma: Option<EWMAFairTracker>,
    vwap: Option<VWAPTracker>,
}

impl RefPriceFn {
    pub fn new(name: &str, params: &HashMap<String, f64>) -> Self {
        let ewma = if name == "ewma_fair" {
            Some(EWMAFairTracker::new(get_param(params, "ewma_half_life_ns") as i64))
        } else {
            None
        };
        let vwap = if name == "vwap" {
            Some(VWAPTracker::new(get_param(params, "vwap_window_ns") as i64))
        } else {
            None
        };
        Self { name: name.to_string(), ewma, vwap }
    }

    pub fn eval(&mut self, book: Option<&Book>) -> Option<f64> {
        use crate::quoter::refprice::{microprice, top_mid, weighted_mid};
        match self.name.as_str() {
            "mid" => top_mid(book),
            "microprice" => microprice(book),
            "weighted_mid" => weighted_mid(book),
            "ewma_fair" => {
                if let (Some(b), Some(t)) = (book, self.ewma.as_mut()) {
                    if let Some(m) = b.mid() {
                        t.observe(b.ts_ns, m);
                    }
                }
                self.ewma.as_ref()?.value(book.map(|b| b.ts_ns).unwrap_or(0))
            }
            "vwap" => {
                let ts = book.map(|b| b.ts_ns).unwrap_or(0);
                let v = self.vwap.as_mut()?.value(ts);
                if v.is_none() { top_mid(book) } else { v }
            }
            // model_pred: no ML layer; deterministic mid fallback.
            "model_pred" => top_mid(book),
            other => panic!("unknown reference price: {}", other),
        }
    }

    pub fn observe_trade(&mut self, trade: &TradeEvent) {
        if let Some(t) = self.vwap.as_mut() {
            t.observe(trade);
        }
    }
}

// --------------------------------------------------------------------- //
// Inventory-penalty closure
// --------------------------------------------------------------------- //

/// Boxed inventory-penalty function (closure over IS params).
pub type InvPenaltyFn = Box<dyn FnMut(f64) -> Skew>;

fn make_inv_penalty(name: &str, params: &HashMap<String, f64>) -> InvPenaltyFn {
    let gamma = get_param(params, "gamma");
    let cap = get_param(params, "inventory_cap");
    match name {
        "linear" => Box::new(move |inv| inv_penalty::linear(inv, gamma)),
        "quadratic" => Box::new(move |inv| inv_penalty::quadratic(inv, gamma)),
        "exponential" => {
            let scale = cap.max(1e-9);
            Box::new(move |inv| inv_penalty::exponential(inv, gamma, scale))
        }
        "asymmetric" => Box::new(move |inv| {
            inv_penalty::asymmetric(inv, gamma, gamma / 2.0)
        }),
        "soft_cap" => Box::new(move |inv| inv_penalty::soft_cap(inv, gamma, cap)),
        "hard_cap" => Box::new(move |inv| inv_penalty::hard_cap(inv, cap)),
        other => panic!("unknown inventory penalty: {}", other),
    }
}

// --------------------------------------------------------------------- //
// Adverse-filter factory
// --------------------------------------------------------------------- //

fn make_adverse_filter(
    name: &str,
    params: &HashMap<String, f64>,
) -> Option<Box<dyn AdverseFilter>> {
    if name == "none" {
        return None;
    }
    let window_ns = get_param(params, "adverse_window_ns") as i64;
    let thresh = get_param(params, "filter_threshold");
    let vol_thresh_bp = get_param(params, "adverse_vol_threshold_bp");
    let f: Box<dyn AdverseFilter> = match name {
        "ofi" => Box::new(OFIFilter::new(window_ns, thresh)),
        "toxicity" => {
            let t_mapped = (0.5 + (thresh / 0.9) * 0.45).clamp(0.5, 1.0);
            Box::new(TradeToxicityFilter::new(window_ns, t_mapped))
        }
        "vol_surge" => Box::new(VolSurgeFilter::new(window_ns, vol_thresh_bp)),
        "microprice_dev" => Box::new(MicropriceDevFilter::new(vol_thresh_bp)),
        "queue_imb" => Box::new(QueueImbalanceFilter::new(thresh)),
        "hybrid" => {
            let children: Vec<Box<dyn AdverseFilter>> = vec![
                Box::new(OFIFilter::new(window_ns, thresh)),
                Box::new(QueueImbalanceFilter::new(thresh)),
            ];
            Box::new(HybridAdverseFilter::new(children, HybridAdverseMode::Any))
        }
        other => panic!("unknown adverse filter: {}", other),
    };
    Some(f)
}

// --------------------------------------------------------------------- //
// Hedge-engine factory
// --------------------------------------------------------------------- //

fn make_hedge_engine(name: &str, params: &HashMap<String, f64>) -> Option<HedgeEngine> {
    if name == "none" || name == "options_vega_stub" {
        return None;
    }
    let threshold = get_param(params, "hedge_threshold");
    match name {
        "perp" => Some(HedgeEngine::new(threshold, 1.0, "perp")),
        "basket" => Some(HedgeEngine::new(threshold, 1.0, "basket")),
        other => panic!("unknown hedge mode: {}", other),
    }
}

// --------------------------------------------------------------------- //
// Refresh-trigger factory
// --------------------------------------------------------------------- //

fn make_refresh_trigger(
    name: &str,
    params: &HashMap<String, f64>,
) -> Box<dyn RefreshTrigger> {
    match name {
        "time" => Box::new(TimeTrigger::new(
            get_param(params, "refresh_interval_ns") as i64)),
        "mid_move" => Box::new(MidMoveTrigger::new(
            get_param(params, "trigger_mid_move_bp"))),
        "inv_change" => Box::new(InvChangeTrigger::new(
            get_param(params, "trigger_inv_change"))),
        "book_event" => Box::new(BookEventTrigger::new()),
        "hybrid" => {
            let children: Vec<Box<dyn RefreshTrigger>> = vec![
                Box::new(TimeTrigger::new(
                    get_param(params, "refresh_interval_ns") as i64)),
                Box::new(MidMoveTrigger::new(
                    get_param(params, "trigger_mid_move_bp"))),
            ];
            Box::new(HybridTrigger::new(children, HybridMode::Any))
        }
        other => panic!("unknown refresh trigger: {}", other),
    }
}

// --------------------------------------------------------------------- //
// Quote-shape factory
// --------------------------------------------------------------------- //

/// Enum-wrapped shape spec so the composer's return type is concrete.
#[derive(Debug, Clone)]
pub enum ShapeSpec {
    Single(SingleSpec),
    Paired(PairedSpec),
    Ladder(LadderSpec),
    Geometric(GeometricSpec),
    DynamicDepth(DynamicDepthSpec),
}

fn make_shape_spec(name: &str, params: &HashMap<String, f64>) -> ShapeSpec {
    let size = get_param(params, "quote_size");
    let half_spread = get_param(params, "spread_floor");
    match name {
        "single" => ShapeSpec::Single(SingleSpec { size, half_spread }),
        "paired" => {
            let step = get_param(params, "ladder_step");
            let levels = vec![(half_spread, size), (half_spread + step, size * 0.5)];
            ShapeSpec::Paired(PairedSpec {
                levels_bid: levels.clone(),
                levels_ask: levels,
            })
        }
        "ladder" => ShapeSpec::Ladder(LadderSpec {
            half_spread,
            step: get_param(params, "ladder_step"),
            n_levels: get_param(params, "ladder_n_levels") as usize,
            size_per_level: size,
        }),
        "geometric" => ShapeSpec::Geometric(GeometricSpec {
            half_spread,
            ratio: get_param(params, "geometric_ratio"),
            n_levels: get_param(params, "ladder_n_levels") as usize,
            size_per_level: size,
        }),
        "dynamic_depth" => ShapeSpec::DynamicDepth(DynamicDepthSpec {
            half_spread,
            step: get_param(params, "ladder_step"),
            max_levels: get_param(params, "ladder_n_levels") as usize,
            inv_taper_threshold: get_param(params, "inventory_cap") * 0.5,
            size_per_level: size,
        }),
        other => panic!("unknown quote shape: {}", other),
    }
}

// --------------------------------------------------------------------- //
// Quoting-model factory
// --------------------------------------------------------------------- //

fn make_quoting_model(
    name: &str,
    ref_name: &str,
    params: &HashMap<String, f64>,
) -> Box<dyn Quoter> {
    use crate::models::symmetric::RefStrategy;
    let size = get_param(params, "quote_size");
    let gamma = get_param(params, "gamma");
    let k = get_param(params, "k");
    let horizon_ns = get_param(params, "horizon_ns") as i64;
    let vol_window_ns = get_param(params, "vol_window_ns") as i64;
    let half_spread = get_param(params, "spread_floor");

    // Map the chosen reference_price token onto SymmetricQuoter's
    // RefStrategy enum where supported; fall back to top_mid for
    // ref tokens not natively supported by Symmetric.
    let ref_strategy = match ref_name {
        "microprice" => RefStrategy::Microprice,
        "weighted_mid" => RefStrategy::WeightedMid,
        _ => RefStrategy::TopMid,
    };

    match name {
        "avellaneda_stoikov" => Box::new(AvellanedaStoikovQuoter::new(
            gamma, k, horizon_ns, size, vol_window_ns)),
        "cartea_jaimungal" => Box::new(CarteaJaimungalQuoter::new(
            gamma, k, get_param(params, "cj_kappa"),
            horizon_ns, size, vol_window_ns)),
        "glft" => Box::new(GLFTQuoter::new(
            gamma, k, get_param(params, "glft_A"),
            horizon_ns, size, vol_window_ns)),
        "ho_stoll" => Box::new(HoStollQuoter::new(
            get_param(params, "ho_stoll_alpha"),
            get_param(params, "ho_stoll_beta"),
            size, vol_window_ns)),
        "symmetric" => Box::new(
            SymmetricQuoter::new(half_spread, size).with_ref(ref_strategy)
        ),
        "ladder" => Box::new(LadderQuoter::new(
            half_spread, get_param(params, "ladder_step"),
            get_param(params, "ladder_n_levels") as usize, size,
        )),
        "microprice_skew" => Box::new(MicropriceSkewQuoter::new(half_spread, size)),
        "fair_anchored" => Box::new(FairAnchoredQuoter::new(
            half_spread, size, get_param(params, "ewma_half_life_ns") as i64)),
        other => panic!("unknown quoting model: {}", other),
    }
}

// --------------------------------------------------------------------- //
// Public composer
// --------------------------------------------------------------------- //

pub struct T4StrategyConfig {
    pub combo: Combo,
    pub params: HashMap<String, f64>,
    pub quoter: Box<dyn Quoter>,
    pub ref_price_fn: RefPriceFn,
    pub inv_penalty_fn: InvPenaltyFn,
    pub adverse_filter: Option<Box<dyn AdverseFilter>>,
    pub hedge_engine: Option<HedgeEngine>,
    pub refresh_trigger: Box<dyn RefreshTrigger>,
    pub shape_spec: ShapeSpec,
    pub quote_size: f64,
    pub spread_floor: f64,
}

pub fn build_mm_strategy(combo: Combo, params: HashMap<String, f64>) -> T4StrategyConfig {
    let ref_price_fn = RefPriceFn::new(combo.reference_price, &params);
    let quoter = make_quoting_model(combo.quoting_model, combo.reference_price, &params);
    let inv_penalty_fn = make_inv_penalty(combo.inventory_penalty, &params);
    let adverse_filter = make_adverse_filter(combo.adverse_filter, &params);
    let hedge_engine = make_hedge_engine(combo.hedge_mode, &params);
    let refresh_trigger = make_refresh_trigger(combo.refresh_trigger, &params);
    let shape_spec = make_shape_spec(combo.quote_shape, &params);

    let quote_size = get_param(&params, "quote_size");
    let spread_floor = get_param(&params, "spread_floor");

    T4StrategyConfig {
        combo,
        params,
        quoter,
        ref_price_fn,
        inv_penalty_fn,
        adverse_filter,
        hedge_engine,
        refresh_trigger,
        shape_spec,
        quote_size,
        spread_floor,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::t4_corpus::combos::{QUOTING_MODELS, INVENTORY_PENALTIES, ADVERSE_FILTERS,
        HEDGE_MODES, REFERENCE_PRICES, QUOTE_SHAPES, REFRESH_TRIGGERS};

    fn make_combo(quoting_model: &'static str) -> Combo {
        Combo {
            quoting_model,
            inventory_penalty: "linear",
            adverse_filter: "none",
            hedge_mode: "none",
            reference_price: "mid",
            quote_shape: "single",
            refresh_trigger: "book_event",
        }
    }

    #[test]
    fn every_quoting_model_composes() {
        for qm in QUOTING_MODELS {
            let _ = build_mm_strategy(make_combo(qm), HashMap::new());
        }
    }

    #[test]
    fn every_inventory_penalty_composes() {
        for ip in INVENTORY_PENALTIES {
            let mut c = make_combo("symmetric");
            c.inventory_penalty = ip;
            let _ = build_mm_strategy(c, HashMap::new());
        }
    }

    #[test]
    fn every_adverse_filter_composes() {
        for af in ADVERSE_FILTERS {
            let mut c = make_combo("symmetric");
            c.adverse_filter = af;
            let cfg = build_mm_strategy(c, HashMap::new());
            if af == "none" {
                assert!(cfg.adverse_filter.is_none());
            } else {
                assert!(cfg.adverse_filter.is_some());
            }
        }
    }

    #[test]
    fn every_hedge_mode_composes() {
        for hm in HEDGE_MODES {
            let mut c = make_combo("symmetric");
            c.hedge_mode = hm;
            let cfg = build_mm_strategy(c, HashMap::new());
            if hm == "none" || hm == "options_vega_stub" {
                assert!(cfg.hedge_engine.is_none());
            } else {
                assert!(cfg.hedge_engine.is_some());
            }
        }
    }

    #[test]
    fn every_reference_price_composes() {
        for rp in REFERENCE_PRICES {
            let mut c = make_combo("symmetric");
            c.reference_price = rp;
            let _ = build_mm_strategy(c, HashMap::new());
        }
    }

    #[test]
    fn every_quote_shape_composes() {
        for qs in QUOTE_SHAPES {
            let mut c = make_combo("symmetric");
            c.quote_shape = qs;
            let _ = build_mm_strategy(c, HashMap::new());
        }
    }

    #[test]
    fn every_refresh_trigger_composes() {
        for rt in REFRESH_TRIGGERS {
            let mut c = make_combo("symmetric");
            c.refresh_trigger = rt;
            let _ = build_mm_strategy(c, HashMap::new());
        }
    }

    #[test]
    fn linear_inv_penalty_correct_sign() {
        let mut cfg = build_mm_strategy(make_combo("symmetric"), HashMap::new());
        let s = (cfg.inv_penalty_fn)(1.0);
        assert!(s.price_offset < 0.0);
        let s = (cfg.inv_penalty_fn)(-1.0);
        assert!(s.price_offset > 0.0);
    }

    #[test]
    fn hard_cap_drops_side() {
        let mut c = make_combo("symmetric");
        c.inventory_penalty = "hard_cap";
        let mut params = HashMap::new();
        params.insert("inventory_cap".to_string(), 0.5);
        let mut cfg = build_mm_strategy(c, params);
        let s = (cfg.inv_penalty_fn)(1.0);
        assert_eq!(s.size_scale_bid, 0.0);
        assert_eq!(s.size_scale_ask, 1.0);
    }
}
