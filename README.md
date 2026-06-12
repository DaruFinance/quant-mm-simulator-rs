# quant-mm-simulator-rs

A high-performance, event-driven **market-making simulator** in Rust.

It replays an L2 limit-order book and trade tape, fills resting quotes with explicit
**queue-position** tracking, carries continuous fractional inventory, drives a library of
microstructure-grounded quoting models, runs a hedge engine, and emits a costed multi-leg trade
ledger with per-fill markout. The engine is built for sweeping large strategy corpora over real
book data; every numeric result is held to bit-level parity against an independent reference
implementation.

## Why this exists
Market-making research lives or dies on two things: a fill model honest enough to trust (queue
position and adverse selection, not just "did price cross my quote"), and the throughput to sweep
a real parameter space over real book data. This engine targets both — a compiled hot path with a
reference oracle proving it computes the same thing as a simple, auditable implementation.

## Quoting-model library
Closed-form and stateful quoters, each behind the `models` feature:

| Model | |
|---|---|
| `avellaneda_stoikov` | Avellaneda–Stoikov inventory-aware optimal quotes |
| `cartea_jaimungal`   | Cartea–Jaimungal market-making with order-flow drift |
| `glft`               | Guéant–Lehalle–Fernandez-Tapia closed-form |
| `ho_stoll`           | Ho–Stoll two-sided dealer |
| `microprice_skew`    | microprice-anchored skew |
| `fair_anchored`      | fair-value-anchored ladder |
| `ladder`             | static multi-level ladder |
| `symmetric`          | symmetric baseline |

## Architecture
```
src/
├── ingest/     L2 book + trade-tape ingestion          (feature: ingest)
├── sim/        event loop, queue position, fills, inventory  (feature: sim)
├── quoter/     Quoter trait + quote shapes, refresh triggers,
│               reference prices, adverse-selection filter, inventory penalty (feature: quoter)
├── models/     quoting-model library (table above)      (feature: models)
├── hedge/      hedge engine                             (feature: hedge)
├── logs/       fill-rate / inventory / queue-position sidecar streams (feature: logs)
└── t4_corpus/  corpus generator — composes every primitive into one
                quoter per structural combo and drives the engine over real fixtures (feature: t4-corpus)
tools/          cross-language parity harness
tests/fixtures/ LOB fixtures for the parity + smoke suites
```
Each subsystem is a cargo feature, so a consumer compiles only what they use. The release profile
is tuned for the sweep workload (`opt-level = 3`, `lto = true`, `codegen-units = 1`).

## Correctness — parity to a reference oracle
Every component has a parity script under `tools/` that runs the Rust path and an independent
reference implementation over the same fixtures and asserts agreement:

- closed-form / deterministic float math — **1e-9 or exact**,
- integer arithmetic — **bit-exact**.

This keeps the fast engine honest: the simple reference is the source of truth, the Rust engine is
the thing that has to match it. Fills emitted by the engine conform to the trade-log v1 schema
(`docs/schemas/`), so a downstream multi-leg aggregation/audit layer consumes engine output
unchanged.

## Build & run
```bash
cargo build --release --features t4-corpus   # full engine + corpus layer
cargo test                                    # unit + parity tests
```

## Related
- A Python sibling, [`quant-mm-simulator`](https://github.com/DaruFinance/quant-mm-simulator),
  provides the reference implementation and the research/orchestration layer.
- The trade-log schema and audit harness come from
  [`quant-research-framework`](https://github.com/DaruFinance/quant-research-framework).

## License
MIT — see `LICENSE`.
