//! T4 IS-optimization search space.
//!
//! Mirror of Python's `mmsim.t4_corpus.search_space`. 7 IS-tune axes:
//!   gamma (risk aversion), k (intensity decay), T (horizon ns),
//!   inventory_cap, refresh_interval_ns, spread_floor, filter_threshold.
//!
//! All sampling is deterministic given the seed (`rand_pcg::Pcg64`).
//!
//! # Parity divergence — seeded sample LISTS
//!
//! Python's `np.random.default_rng(seed)` and `rand_pcg::Pcg64::seed_from_u64`
//! have different SeedSequence mixing. The Rust sample is deterministic
//! given the Rust seed, but the LISTS do not match Python bit-for-bit.
//! The parity assertion checks shape invariants (count, key set, range
//! containment, in-Rust determinism) — mirroring the T5/T6 convention.

#![cfg(feature = "t4-corpus")]

use std::collections::HashMap;
use std::sync::OnceLock;

use rand::Rng;
use rand::SeedableRng;
use rand_pcg::Pcg64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AxisKind {
    /// Integer in `[low, high]` inclusive.
    Int,
    /// Float uniform in `[low, high)`.
    Float,
    /// Float exponential in `[low, high]` — uniform in `log10`.
    LogFloat,
    /// Discrete categorical from `choices`.
    Choice,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ParamValue {
    Int(i64),
    Float(f64),
    Str(String),
}

impl ParamValue {
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            ParamValue::Int(i) => Some(*i as f64),
            ParamValue::Float(f) => Some(*f),
            ParamValue::Str(_) => None,
        }
    }
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            ParamValue::Int(i) => Some(*i),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            ParamValue::Str(s) => Some(s.as_str()),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Axis {
    pub name: String,
    pub kind: AxisKind,
    pub low: f64,
    pub high: f64,
    pub choices: Vec<ParamValue>,
}

impl Axis {
    pub fn int(name: impl Into<String>, low: i64, high: i64) -> Self {
        Self {
            name: name.into(),
            kind: AxisKind::Int,
            low: low as f64,
            high: high as f64,
            choices: Vec::new(),
        }
    }
    pub fn float(name: impl Into<String>, low: f64, high: f64) -> Self {
        Self {
            name: name.into(),
            kind: AxisKind::Float,
            low, high, choices: Vec::new(),
        }
    }
    pub fn log_float(name: impl Into<String>, low: f64, high: f64) -> Self {
        Self {
            name: name.into(),
            kind: AxisKind::LogFloat,
            low, high, choices: Vec::new(),
        }
    }
    pub fn choice(name: impl Into<String>, choices: Vec<ParamValue>) -> Self {
        Self {
            name: name.into(),
            kind: AxisKind::Choice,
            low: 0.0, high: 0.0, choices,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SearchSpace {
    pub axes: Vec<Axis>,
}

impl SearchSpace {
    pub fn new(axes: Vec<Axis>) -> Self {
        let mut seen: HashMap<String, ()> = HashMap::new();
        for a in &axes {
            if seen.insert(a.name.clone(), ()).is_some() {
                panic!("duplicate axis name in SearchSpace: {}", a.name);
            }
        }
        Self { axes }
    }

    pub fn names(&self) -> Vec<String> {
        self.axes.iter().map(|a| a.name.clone()).collect()
    }

    pub fn sample(&self, n: usize, seed: u64) -> Vec<HashMap<String, ParamValue>> {
        let mut rng = Pcg64::seed_from_u64(seed);
        let mut out: Vec<HashMap<String, ParamValue>> = Vec::with_capacity(n);
        for _ in 0..n {
            let mut params: HashMap<String, ParamValue> =
                HashMap::with_capacity(self.axes.len());
            for ax in &self.axes {
                params.insert(ax.name.clone(), draw_axis(ax, &mut rng));
            }
            out.push(params);
        }
        out
    }

    pub fn grid(&self, levels_per_axis: usize) -> Vec<HashMap<String, ParamValue>> {
        let n = levels_per_axis.max(1);
        let per_axis_values: Vec<Vec<ParamValue>> = self
            .axes.iter()
            .map(|ax| match ax.kind {
                AxisKind::Choice => ax.choices.clone(),
                AxisKind::Int => linspace(ax.low, ax.high, n)
                    .into_iter()
                    .map(|v| ParamValue::Int(v.round() as i64))
                    .collect(),
                AxisKind::LogFloat => {
                    let lo = ax.low.log10();
                    let hi = ax.high.log10();
                    linspace(lo, hi, n)
                        .into_iter()
                        .map(|v| ParamValue::Float(10f64.powf(v)))
                        .collect()
                }
                AxisKind::Float => linspace(ax.low, ax.high, n)
                    .into_iter()
                    .map(ParamValue::Float)
                    .collect(),
            })
            .collect();
        cartesian_product(&self.axes, &per_axis_values)
    }

    pub fn n_grid_combinations(&self, levels_per_axis: usize) -> usize {
        let mut total: usize = 1;
        for ax in &self.axes {
            match ax.kind {
                AxisKind::Choice => total = total.saturating_mul(ax.choices.len().max(1)),
                _ => total = total.saturating_mul(levels_per_axis.max(1)),
            }
        }
        total
    }
}

fn draw_axis(ax: &Axis, rng: &mut Pcg64) -> ParamValue {
    match ax.kind {
        AxisKind::Int => {
            let lo = ax.low.round() as i64;
            let hi = ax.high.round() as i64;
            let v: i64 = rng.random_range(lo..=hi);
            ParamValue::Int(v)
        }
        AxisKind::Float => {
            let v: f64 = rng.random_range(ax.low..ax.high);
            ParamValue::Float(v)
        }
        AxisKind::LogFloat => {
            let lo = ax.low.log10();
            let hi = ax.high.log10();
            let u: f64 = rng.random_range(lo..hi);
            ParamValue::Float(10f64.powf(u))
        }
        AxisKind::Choice => {
            if ax.choices.is_empty() {
                panic!("axis '{}' has empty choices", ax.name);
            }
            let i: usize = rng.random_range(0..ax.choices.len());
            ax.choices[i].clone()
        }
    }
}

fn linspace(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    if n <= 1 {
        return vec![lo];
    }
    let step = (hi - lo) / ((n - 1) as f64);
    (0..n).map(|i| lo + (i as f64) * step).collect()
}

fn cartesian_product(
    axes: &[Axis],
    per_axis_values: &[Vec<ParamValue>],
) -> Vec<HashMap<String, ParamValue>> {
    let mut out: Vec<HashMap<String, ParamValue>> = Vec::new();
    if per_axis_values.iter().any(|v| v.is_empty()) {
        return out;
    }
    let mut idx = vec![0usize; per_axis_values.len()];
    loop {
        let mut params: HashMap<String, ParamValue> = HashMap::with_capacity(axes.len());
        for (k, ax) in axes.iter().enumerate() {
            params.insert(ax.name.clone(), per_axis_values[k][idx[k]].clone());
        }
        out.push(params);
        // Increment right-most index (matches Python `itertools.product`).
        let mut k = per_axis_values.len();
        loop {
            if k == 0 {
                return out;
            }
            k -= 1;
            idx[k] += 1;
            if idx[k] < per_axis_values[k].len() {
                break;
            }
            idx[k] = 0;
        }
    }
}

/// Return the T4 SearchSpace (singleton). Bounds match the Python sibling.
pub fn search_space_t4() -> &'static SearchSpace {
    static S: OnceLock<SearchSpace> = OnceLock::new();
    S.get_or_init(|| {
        SearchSpace::new(vec![
            Axis::log_float("gamma", 0.05, 5.0),
            Axis::log_float("k", 0.1, 10.0),
            Axis::log_float("horizon_ns", 1e10, 3.6e12),
            Axis::log_float("inventory_cap", 0.001, 1.0),
            Axis::log_float("refresh_interval_ns", 1e5, 1e10),
            Axis::log_float("spread_floor", 0.5, 50.0),
            Axis::float("filter_threshold", 0.01, 0.9),
        ])
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seven_axes() {
        let sp = search_space_t4();
        assert_eq!(sp.axes.len(), 7);
        let names = sp.names();
        for n in ["gamma", "k", "horizon_ns", "inventory_cap",
                   "refresh_interval_ns", "spread_floor", "filter_threshold"] {
            assert!(names.contains(&n.to_string()));
        }
    }

    #[test]
    fn sample_count_matches() {
        let sp = search_space_t4();
        let s = sp.sample(64, 2026);
        assert_eq!(s.len(), 64);
    }

    #[test]
    fn sample_keys_match_axes() {
        let sp = search_space_t4();
        let s = sp.sample(8, 2026);
        let expected: std::collections::HashSet<String> = sp.names().into_iter().collect();
        for sample in &s {
            let got: std::collections::HashSet<String> = sample.keys().cloned().collect();
            assert_eq!(got, expected);
        }
    }

    #[test]
    fn sample_values_in_range() {
        let sp = search_space_t4();
        let s = sp.sample(32, 2026);
        for sample in &s {
            for ax in &sp.axes {
                let v = sample.get(&ax.name).expect("axis present");
                match ax.kind {
                    AxisKind::Int => {
                        let i = v.as_i64().expect("int");
                        assert!((i as f64) >= ax.low && (i as f64) <= ax.high);
                    }
                    AxisKind::Float => {
                        let f = v.as_f64().expect("float");
                        assert!(f >= ax.low && f < ax.high);
                    }
                    AxisKind::LogFloat => {
                        let f = v.as_f64().expect("float");
                        assert!(f >= ax.low && f <= ax.high * 1.0000001);
                    }
                    AxisKind::Choice => assert!(ax.choices.contains(v)),
                }
            }
        }
    }

    #[test]
    fn sample_deterministic_same_seed() {
        let sp = search_space_t4();
        let a = sp.sample(16, 2026);
        let b = sp.sample(16, 2026);
        for (sa, sb) in a.iter().zip(b.iter()) {
            for (k, v) in sa {
                assert_eq!(sb.get(k), Some(v));
            }
        }
    }

    #[test]
    #[should_panic]
    fn duplicate_axis_panics() {
        let _ = SearchSpace::new(vec![
            Axis::float("x", 0.0, 1.0),
            Axis::float("x", 0.0, 1.0),
        ]);
    }
}
