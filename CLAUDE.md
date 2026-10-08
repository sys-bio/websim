# websim

An ODE simulator for biochemical models, written in Rust with egui. It runs as
a desktop app and in the browser (WebAssembly), published at
https://sys-bio.github.io/websim/ from https://github.com/sys-bio/websim.

It is also the model layer for **bifurcata-rs**
(https://github.com/sys-bio/bifurcata-rs), a pure-Rust rewrite of the Delphi
bifurcation library Bifurcata. `docs/bifurcation-rust-plan.md` is the plan for
that work and records its progress; read it before working on anything to do
with steady states, conservation analysis or continuation.

## Layout

- `src/` — the egui app: `main.rs` (UI), `export.rs` (save a file: a Save
  dialog on the desktop, a download in the browser), `bifurcation.rs` (the
  Bifurcation view: one-parameter equilibrium branches from `bifurcata`, grown
  a few steps per frame, stable solid / unstable dashed, LP/BP/H marked).
- `bifurcata` is a git dependency (feature `antimony`). That feature reaches
  `websim-model` through a git dependency on this repository, so `Cargo.toml`
  `[patch]`es it back to `crates/model`: one copy of the model types. To build
  against a local, unpushed bifurcata, add
  `--config 'patch."https://github.com/sys-bio/bifurcata-rs".bifurcata.path="../bifurcata-rs"'`
  (Trunk takes no `--config`: put the same patch in `.cargo/config.toml`
  temporarily, and never commit it).
- `crates/model/` — `websim-model`, the model layer with no UI code:
  - `antimony.rs` — the Antimony-subset front end (reactions, rules, rate
    rules, events, `/* */` comments); also returns the stoichiometry.
  - `model.rs` — the model, expression parser, simulation with events, the
    reduced system (rates, elasticities, reduced Jacobian), examples.
  - `conservation.rs` — conserved moieties, independent species, link matrix.
  - `linalg.rs` — dense matrix, LU with condition estimate, eigenvalues (faer).
  - `newton.rs` — damped Newton, Armijo line search, Levenberg–Marquardt.
  - `steady.rs` — steady states and stability.
  - `ode.rs`, `solvers.rs` — RK4, and BDF/ESDIRK34/Tsit45 via diffsol.
- `crates/model/tests/delphi_models.rs` — checks against the Delphi project's
  bundled models and baselines (see Testing).
- `docs/` — the bifurcation plan.
- `.github/workflows/pages.yml` — builds with Trunk and publishes to Pages on
  every push to `main`.

## Building and running

```
cargo run --release                 desktop app
cargo test --workspace              all tests
cargo check --target wasm32-unknown-unknown    the browser build compiles
trunk serve                         browser version at http://127.0.0.1:8080
trunk build --release --public-url /websim/    what CI publishes
```

Pushing to `main` deploys the site; check the Actions run, then the page.

## Testing

- Unit tests live beside the code in `crates/model/src/*.rs`.
- `delphi_models.rs` reads the Delphi project **in place, read-only** from
  `D:\Documents\Embarcadero\Studio\Projects\Bifurcation_Delphi` (or
  `BIFURCATA_DELPHI_DIR`) and is skipped when it is absent, so it passes on CI
  and other machines without checking anything. It asserts that every bundled
  `.ant` model loads and that every `ant_*.json` baseline's first point — the
  steady state libRoadRunner found — is reproduced.
- The Delphi project is never modified.

## Conventions and pitfalls

- **Browser constraints.** No threads, no file system, no
  `std::time::Instant` (panics in wasm). faer is used with
  `default-features = false`: its default `rayon` feature pulls in
  `spindle` → `atomic-wait`, which does not compile for wasm. Call
  `faer::set_global_parallelism(Par::Seq)` before faer routines.
- **getrandom in the browser** needs both the `wasm_js` feature (wasm-only
  dependency in `crates/model/Cargo.toml`) and the `getrandom_backend` cfg in
  `.cargo/config.toml`.
- **Trunk and `NO_COLOR`.** This machine's shell sets `NO_COLOR=1` and Trunk
  only accepts `true`/`false`: run it with `$env:NO_COLOR = 'true'` first.
- **Delphi baselines start with a UTF-8 BOM**, which serde_json rejects;
  strip `'\u{feff}'` first.
- **Editing files with non-ASCII text** (α, –, ×): use the Edit tool, not
  shell `sed`/Python, which have mangled or truncated files here before.
- **Line endings are LF.** Python on Windows writes CRLF in text mode; a
  Python edit turned a 32-line change to `main.rs` into a 1,556-line diff.
  Write bytes, or check `git diff --stat` before committing.
- **Slider number boxes** reserve a fixed width (`slider_value_room`): a row
  sized from the current value made the side panel grow without end.
- **Finite-difference rate-law derivatives**, not symbolic: symbolic
  derivatives of enzyme rate laws blow up and are slower to evaluate.

### Numerical facts (steady states)

All found while matching the Delphi baselines; the plan, §5.4, has more.

- **Elasticity steps are relative to each species' typical size**, floored at
  1e-3 of the largest species, or 1 if the whole state is zero — never relative
  to zero itself — and one-sided where a backward step would make a
  non-negative species negative.
- **Newton is scaled by the Jacobian** (`scale_from_jacobian`):
  `typical_f_i = Σ_j |J_ij| typical_x_j`. Scaling by `|F(x0)|` (the Delphi
  choice) breaks when a rate starts at exactly zero.
- **Levenberg–Marquardt takes over at rcond 1e-8** in the steady-state solve,
  because a finite-difference Jacobian makes an exactly singular one look
  merely ill-conditioned. Its damping is relative to the largest diagonal of JᵀJ.
- **Newton first keeps species non-negative** (`NonlinearSystem::admissible`),
  so it cannot jump past a rate law's pole; free Newton follows for models
  whose steady state is negative.
- **A stable steady state is preferred**; an unstable one is returned only if
  integrating finds nothing stable (an oscillator).
- **Known issue:** BDF at default tolerances gives up on tyson2001 at
  t ≈ 2250 (diffsol nonlinear-solver failures). The steady-state presimulation
  integrates at rtol 1e-8 to avoid it; a user simulating with the defaults can
  still hit it.

## Progress

See the milestone table in `docs/bifurcation-rust-plan.md` (§8). In short:
**M0 done** (conservation analysis, steady states; October 2026). **M1 done**
in bifurcata-rs (foundations, Stage 0 ported). **M2 done** there (equilibrium
continuation with fold/branch-point/Hopf detection, all baselines matched).
A first **Bifurcation view** is in the app (an early piece of M5: one
parameter, both directions from the steady state). Next is **M3** in
bifurcata-rs: normal forms and branch switching.
