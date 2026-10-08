# Plan: a pure-Rust bifurcation library (stages 0–3)

**Status:** in progress — M0 and M1 done (see §5.4 and §8) · **Date:** October 2026
**Scope of this plan:** up to and including specification stage 3 (equilibria and
codimension-2 curves). Periodic orbits (stages 4–5) and automatic diagrams
(stage 6) are deliberately excluded; a separate plan follows a review at the end
of stage 3.

---

## 1. Goal and ground rules

Build a pure-Rust equivalent of the Delphi library **Bifurcata**
(`D:\Documents\Embarcadero\Studio\Projects\Bifurcation_Delphi`) that runs on
the desktop *and* in the browser (WebAssembly), and use it in the websim app.

Ground rules:

- **The Delphi library is not changed.** It stays in its own repository and is
  used only as a reference and as a source of test targets.
- **The bifurcation library lives in a new, separate repository** (proposed:
  `sys-bio/bifurcata-rs`, crate name `bifurcata`).
- **Model-side work lives in websim** — conservation analysis and the
  steady-state solver are needed by the simulator anyway, and replace what
  libRoadRunner supplies to the Delphi version.
- **Pure Rust, no C or C++ dependencies**, so that everything compiles to
  `wasm32-unknown-unknown`. This rules out LAPACK, libRoadRunner, libantimony
  and SUNDIALS.
- **Stop after stage 3** for review.

## 2. Sources of truth

| Source | Role |
|---|---|
| `BifurcationSpec.md` | **The authority on what the library does.** The Rust version follows it section by section. Where it names Delphi types, the Rust design follows the intent, not the shape. |
| Delphi `CLAUDE.md` | Numerical lessons and conventions that must be carried over deliberately (sign conventions, normalisations, location refinement, duplicate tolerances, testing discipline). Read the relevant part before writing each module. |
| Delphi source (`src\`) | Reference for algorithmic detail the spec leaves open. Read, not transcribed: each Rust module is written fresh against the spec, consulting the matching unit. |
| Delphi tests (`tests\`, ~910 checks) | Expected values, tolerances and test problems. Ported check by check into Rust `#[test]`s. |
| Baselines (`baselines\*.json`, schema `bifurcata/1`) | Cross-implementation oracle. The Rust harness writes the same schema and is compared against them. |
| Independent references | PP2 closed forms, MatCont manual §8.1.5, Kuznetsov's normalisation of the Brusselator `l1 = -0.5`, BifurcationKit values recorded in the Delphi tests. Preferred over baselines wherever available (CLAUDE.md: "baselines lock in current behaviour, including wrong behaviour"). |

## 3. Architecture

### 3.1 Two repositories, one direction of dependency

```
websim (this repo)                         bifurcata-rs (new repo)
───────────────────                        ───────────────────────
crates/model   ◄──────── optional dep ──── bifurcata   (library)
  Antimony parser                            BifurcationProblem trait
  conservation analysis                      linear algebra, Newton, PALC,
  steady-state solver                        equilibria, normal forms,
  reduced rates + Jacobian                   branch switching, codim-2,
  simulation (diffsol)                       serialisation, comparator
                                           + feature "antimony":
app (egui) ──── depends on both ────►        impl BifurcationProblem for
                                             websim's reduced model
                                           + bin "bifurcata" (harness)
```

- `bifurcata` defines the `BifurcationProblem` trait (spec §3.1) and knows
  nothing about Antimony. Its unit tests use hand-written problems, as the
  Delphi tests do (`TBrusselator`, `TFoldProblem`, `TGoodwinProblem`, …).
- websim is reorganised into a Cargo workspace, splitting the model layer
  (`src/antimony.rs`, `model.rs`, `ode.rs`, `solvers.rs`) into a library crate
  `crates/model` with no GUI dependency. The app keeps its current behaviour.
- `bifurcata`'s optional `antimony` feature implements the trait for websim's
  reduced model — the equivalent of the Delphi `TRoadRunnerProblem` adapter
  (spec §3.4). With it, `bifurcata`'s harness can run the `.ant` baselines.
- The websim app enables that feature for its bifurcation view.

This keeps `bifurcata` reusable with other model sources (spec §3.5's "future
native backend" is, in effect, what websim's model crate is).

### 3.2 Delphi dependency → Rust replacement

| Delphi | Used for | Rust replacement |
|---|---|---|
| LAPACK `dgetrf/dgetrs/dgecon` | LU, solves, condition estimate | **faer** (pure Rust; already in the build via diffsol). Condition estimate: faer's or a Hager/Higham estimator (~60 lines). |
| LAPACK `dgeqrf/dorgqr` | null vectors, initial tangent | faer QR |
| LAPACK `dgesvd` | rank decisions, diagnostics | faer SVD |
| LAPACK `dgeev` | spectrum, right **and left** eigenvectors | faer eigendecomposition. Left eigenvectors are only needed *at critical points* (normal forms), so compute them there by a bordered null-vector solve of `(A − μI)ᵀ p = 0` rather than pairing two spectra — robust, and it sidesteps eigenvalue-matching. |
| OpenBLAS | BLAS | none needed (faer); the CLAUDE.md threading problem disappears. |
| `Bifurcata.Complex` | complex arithmetic | `faer::c64` / `num-complex` |
| libRoadRunner + libantimony | model loading, rates, **conservation analysis**, reduced Jacobian, steady state | websim `crates/model` (§4, §5) |
| CVODE | trajectory seeding (stage 4 only) | diffsol (already used) |
| Delphi JSON writer | `bifurcata/1` schema | `serde` + `serde_json` |
| FMX GUI | viewer | the websim egui app |

WebAssembly constraints to respect from the start: no threads, no file system in
the library, and **no `std::time::Instant`** (it panics in the browser) — time
budgets go through the `web-time` crate or an injected clock.

## 4. Conservation analysis (websim, milestone M0)

**Required, not optional:** without the reduction, the Jacobian of a model with a
conserved moiety is singular everywhere, Newton cannot converge and every
equilibrium looks non-hyperbolic (spec §3.2). In the Delphi version
libRoadRunner does this; here it has to be written.

### 4.1 What to compute

From the parsed reactions (floating species only — boundary `$` species are
parameters):

1. **Stoichiometry matrix** `N` (m species × r reactions), from the reaction
   terms the Antimony front end already produces.
2. **Conservation laws**: the left null space of `N`, i.e. rows `γ` with
   `γ N = 0`. Rank-revealing elimination — Gauss–Jordan with partial pivoting
   and a tolerance, or faer's column-pivoted QR on `Nᵀ`. Stoichiometries are
   small integers, so a rational/integer elimination is also an option and
   removes the tolerance question entirely; decide during M0 (Delphi has no
   code for this to compare with — libStructural did it inside libRoadRunner).
3. **Independent species**: a maximal set of linearly independent rows of `N`,
   chosen deterministically (declaration order, greedy).
4. **Reduced stoichiometry** `N_R` (independent rows) and **link matrix**
   `L = [I; L₀]`, with `N = L N_R`, and dependent species
   `x_dep = L₀ x_ind + T`, where the **conserved totals** `T` are fixed from
   the initial values.
5. **Reduced system** `du/dt = N_R v(L u + T)` in independent species only.
6. **Reduced Jacobian** `J = N_R ε L`, with elasticities `ε = ∂v/∂x` by central
   differences per species. Each entry differentiates one rate law with respect
   to one species, so there is no cancellation across unrelated terms — the
   same accuracy argument as libRoadRunner's (spec §3.1), and consistent with
   the finite-difference policy for rate-law derivatives.
7. **Conserved totals as parameters**, named `_CSUM0`, `_CSUM1`, … as
   libRoadRunner does (the `csum_edelstein` baseline continues in `_CSUM0`).
   Setting one recomputes the dependent species.

Species with **rate rules** (`x' = …`) are not reaction-driven: they are always
independent and are appended after the reaction species. (Note: Delphi's
CLAUDE.md says rate rules are unusable *there* only because libRoadRunner's
Jacobian is reaction-derived; here they work naturally.)

### 4.2 Checks (spec §3.4, §11.3; Delphi `Tests.RoadRunner`, 56 checks)

- **edelstein**: 3 floating species, 2 independent, 1 conservation law (E + C).
- **moiety3** closed form: S1 ↔ S2 ↔ S3 has `S2 = 1.75 S1`, `S3 = 7.875 S1`, so
  continuing in the total is a straight line of known slope.
- **State consistency**: setting independent species and reading back
  dependents preserves every total.
- **Spectral consistency**: eigenvalues of the reduced Jacobian equal the
  nonzero eigenvalues of the full `N ε`.
- **Ordering**: the reduced Jacobian agrees with `N_R ε L` assembled
  independently.
- **Hidden conservation** (`FindHiddenConservation`): a conserved combination
  the stoichiometry cannot see leaves the *reduced* Jacobian singular
  everywhere; detect at load and say so. Can follow after M0.

libRoadRunner may choose a different independent set from ours, so baseline
states can be in a different basis. This is not a concern: compare by name
after reconstructing the full state, not by position. The baselines record
`stateNames` and `independentCount`.

### 4.3 Antimony front-end additions needed by the bundled models

The 21 `.ant` models use few constructs beyond what websim already parses:
`/* … */` block comments, the `[bifurcation]` settings block (which sits inside
a block comment), and `species` declarations. To be confirmed by loading every
bundled model in M0; anything missing is added then.

## 5. Steady-state solver (websim, milestone M0)

Needed twice: the app wants a "find steady state" option anyway, and the
continuation needs a starting equilibrium (spec §6.1).

### 5.1 Crates surveyed

| Crate | Notes | Verdict |
|---|---|---|
| `gomez` 0.5 | Pure Rust; Newton-type and trust-region solvers for `F(x) = 0`, plus LM; nalgebra-based | Credible; worth a trial if our own solver struggles |
| `levenberg-marquardt` 0.15 | Pure Rust least squares (rust-cv) | Could serve as the LM fallback only |
| `argmin` 0.11 | Optimisation framework | Not aimed at `F(x) = 0`; would need a least-squares wrapping |
| `roots` | Scalar root finding only | Not applicable |
| `sundials-sys` (KINSOL) | C library | Rules out WebAssembly |
| diffsol's internal Newton | Tied to diffsol's own operator traits | Awkward to use standalone |

### 5.2 Decision: our own solver, following the spec and libRoadRunner

A crate is not the hard part — **globalisation and scaling for badly scaled
biochemical models are**, and the spec already prescribes the method. Write it
ourselves (~400 lines, reference `Bifurcata.Newton.pas`, 572 lines):

1. **Presimulation**: integrate the reduced system with diffsol (BDF) towards an
   attractor — "a trajectory-derived guess beats any globalisation strategy
   from a poor start" (spec §6.1). This is what libRoadRunner does.
2. **Damped Newton** on the reduced system with **Armijo line search** (spec
   §5.3) and **variable and residual scaling** (µM-to-mM ranges).
3. **Levenberg–Marquardt fallback** when the Jacobian is ill-conditioned or the
   line search stalls: `(JᵀJ + µI)δ = −JᵀF`, adapting µ (spec §6.1).
4. Report the steady state, the **eigenvalues of the reduced Jacobian** and
   **stability**; warn on negative concentrations (`HasNegativeSpecies`).

The bifurcation library's own Newton corrector (spec §5, §2.1) stays separate:
it is coupled to step control and tangents. `BifurcationProblem` gets an
optional `steady_state` method so `bifurcata` can ask the model layer for a
starting point, as the Delphi engine asks libRoadRunner.

### 5.3 In the app

A **Steady state** action showing the values, eigenvalues and stability, and
able to use the steady state as initial conditions. Useful on its own, before
any bifurcation work.

### 5.4 M0 outcome (October 2026)

Done. websim is now a workspace with the model layer in `crates/model`
(`websim-model`): conservation analysis (`conservation.rs`), dense linear
algebra (`linalg.rs`), Newton (`newton.rs`) and steady states (`steady.rs`),
plus a **Steady state** section in the app.

Checked against the Delphi baselines (`crates/model/tests/delphi_models.rs`,
reading the Delphi project in place, skipped when it is absent): all 21
bundled models load, and **all 20 `ant_*` baselines' starting steady states
are reproduced, worst relative difference 1.3e-11**, with every model's
independent-species count equal to libRoadRunner's.

What it took, beyond the plan — worth knowing for the continuation code:

- **Elasticity steps must not be relative to zero.** A model starting from
  all-zero concentrations got a finite-difference step of ~1e-18 and a
  meaningless Jacobian. Steps are now relative to each species' typical size
  (floored at 1e-3 of the largest, or 1 if all are zero), one-sided where a
  backward step would make a non-negative species negative.
- **Scaling equations by |F(x0)| breaks when a rate starts at zero** (the
  Delphi scaling; libRoadRunner shielded it from this). That equation gets a
  tiny typical size, dominates the merit function, and the line search can only
  take microscopic steps. Scaling is now `typical_f_i = Σ_j |J_ij| typical_x_j`
  — the residual change for a typical change of state — and the
  Levenberg–Marquardt damping is relative to the largest diagonal of JᵀJ.
- **A finite-difference Jacobian makes an exactly singular one look merely
  ill-conditioned.** hopf2folds' Jacobian is singular at the origin; Newton
  took a step of 3.7e9 along the null direction. With Jacobian errors around
  1e-8, the switch to Levenberg–Marquardt is at rcond 1e-8, not 1e-12.
- **Newton must be kept inside the physical region first** (tyson2001: free
  Newton runs past the pole at TF = −J16 and stalls with Cdh1 = −59). The first
  attempt now rejects trial points with negative species; free Newton follows
  for models whose steady state really is negative (hopf2folds, ODEs).
- **Prefer a stable steady state.** With several equilibria, Newton from the
  initial values can land on an unstable one (Lab2: the saddle at (1, 0)). An
  unstable result is kept only if integrating does not find a stable one — an
  oscillator such as tyson2001 keeps its unstable steady state. With this,
  Lab2 and edelstein reproduce libRoadRunner's choice too.

## 6. The `bifurcata` crate (milestones M1–M4)

### 6.1 Module map

| Rust module | Delphi unit | Spec | Stage |
|---|---|---|---|
| `types` | `Types` | §4 | 0 |
| `linalg` (LU, QR, SVD, eigen, cond. estimate) | `Matrix`, `LinAlg`, `Blas`, `Complex` | §3A | 0 |
| `bordered` | `LinAlg` | §3A.3 | 0 |
| `problem` (trait + FD parameter derivative) | `Problem` | §3.1 | 0 |
| `derivatives` (5-point directional D2/D3, polarisation) | `Derivatives` | §3.3 | 2 |
| `newton` | `Newton` | §5.3, §6.1 | 1 |
| `continuation` (engine as an iterator) | `Continuation` | §4.2, §5 | 1 |
| `equilibrium` (defining system, LP/BP/H tests) | `Equilibrium`, `Bialternate` | §6.1–6.3 | 1 |
| `normal_forms` (fold `a`, Hopf `l1` in both normalisations, BP) | `NormalForms` | §6.4 | 2 |
| `branch_switch` | `BranchSwitch` | §6.5 | 2 |
| `codim2` (fold and Hopf curves; CP, BT, ZH, GH, HH) | `Codim2` | §6.6 | 3 |
| `serialise` (`bifurcata/1`) and `compare` | `Serialise` | §10.3 | 1 |
| `run_spec` (`[bifurcation]` block, options) | `RunSpec` | §10 | 1 |
| `antimony` feature (adapter to websim's model) | `RoadRunner`, `Antimony`, `Models` | §3.4 | 1 |
| `bin/bifurcata`: `run`, `curve`, `compare`, `info`, `models` | `harness\bifurcata.dpr` | §10.1 | 1–3 |

Out of scope here: `Collocation`, `Condensation`, `CycleCodim2`, `Diagram`, and
the `cycles` and `diagram` commands.

### 6.2 Design points to carry over (from the spec and CLAUDE.md)

- **Defining-system abstraction** (§4.1): one engine, written against a
  `DefiningSystem` trait; equilibria, fold curves and Hopf curves are
  implementations.
- **The engine is an iterator** (§4.2): `step()` advances one continuation
  step, so the GUI can draw as the branch grows and stop at any time. Essential
  in the browser, where long computations must yield to the UI.
- **PALC** with tangent bordering by the previous tangent, step control and the
  30° angle criterion (§5.1); Moore–Penrose as an option (§5.2).
- **Bialternate Hopf test** normalised as sign × Σ log|pivot| (§6.3), with
  neutral saddles recorded but not called Hopf (PP2's p1 = 0.4 is the test).
- **`RefineAndLocate`**: re-walk the bracket in real continuation steps before
  bisecting (CLAUDE.md; took Lab 2's branch point from 1.2% to nine figures).
- **Conventions**: fold-coefficient eigenvector sign ("largest-magnitude
  component positive"), `l1` reported in both Kuznetsov's and MatCont's
  normalisation (they differ by a factor of ω), `ActiveParameterIndex` indexes
  the unknown vector, `Lambda` carries every parameter.
- **Comparator semantics**: compare bifurcation kind, location, classification
  and coefficients, plus every 5th point; `CoefficientZeroTolerance` (1e-6)
  suppresses sign checks on definitionally-zero coefficients.

## 7. Validation

### 7.1 Unit tests ported from Delphi

| Delphi test unit | Checks | Lands in |
|---|---|---|
| `Tests.Stage0` | 115 | M1 |
| `Tests.RoadRunner` | 56 | M0 (as model-crate tests: conservation, consistency) |
| `Tests.Newton` | 38 | M2 |
| `Tests.Continuation` | 72 | M2 |
| `Tests.Bialternate` | 28 | M2 |
| `Tests.Scale` | 19 | M2 |
| `Tests.Serialise` | 57 | M2 (equilibrium parts) |
| `Tests.Derivatives` | 20 | M3 |
| `Tests.NormalForms` | 37 | M3 |
| `Tests.BranchSwitch` | 26 | M3 |
| `Tests.Codim2` | 135 | M4 |
| **In scope** | **≈ 600** | |
| Out of scope (stages 4–6): `Collocation` 104, `CycleCodim2` 59, `Diagram` 26 | 189 | — |

The test problems in `Bifurcata.TestProblems.pas` (Brusselator, fold, badly
scaled, Goodwin, transcritical, arctan, singular-consistent, cubic) are ported
first, as they underpin most suites.

### 7.2 Baselines (about 29 of the 41 are in scope)

- **Built-in models** (`run`): `brusselator`, `selkov`, `saddlenode`,
  `transcritical`.
- **Antimony models** (`run`): all 20 `ant_*.json`.
- **Conserved totals**: `moiety3`, `csum_edelstein`.
- **Codim-2 curves** (`curve`): `bistable_fold`, `lorenz84_hopf`; branch
  switching: `lorenz84_hopf_switch`.

Acceptance (spec §11.2): bifurcation parameter values to **6 significant
figures**, normal-form coefficients to **3 significant figures** with
unambiguous sign. Point-by-point agreement is expected to be close but not
exact — the Jacobian's finite differences differ from libRoadRunner's in the
last digits, which can change the step sequence — so points are compared with a
looser tolerance than bifurcations.

### 7.3 Independent targets (preferred over baselines)

- **PP2 closed forms**: BP 0.6, neutral saddle 0.4, BP 0.821904345374,
  LP 0.832929322234, Hopf 0.671593847479 with ω = 0.604782221942.
- **Brusselator** `l1 = −0.5` (Kuznetsov), ω = 1.
- **MatCont manual §8.1.5** catalytic oscillator values.
- **moiety3** linear closed form; **edelstein** fold at `_CSUM0 = 0.5125283838`.

### 7.4 Discipline (from CLAUDE.md)

- **Mutate the code and watch the test fail** before trusting a new test.
- Watch for **vacuous checks** (the Brusselator's `g ≡ 1`, a 1×1 bialternate
  product at n = 2, a symmetric matrix where swapping φ and ψ changes nothing).
- Prefer independently known targets over comparing two discretisations.

## 8. Milestones

Each milestone ends with all ported tests passing and a review point.

| Milestone | Repository | Content | Exit criterion | Status |
|---|---|---|---|---|
| **M0** | websim | Workspace split; stoichiometry and conservation analysis; reduced rates and Jacobian; steady-state solver; Steady-state action in the app; all 21 bundled `.ant` models load | moiety3/edelstein/consistency checks pass; steady states match the baselines' first points | **Done**, October 2026 (§5.4): 20/20 baselines to 1.3e-11 |
| **M1** | bifurcata-rs | Repo, CI (native + wasm build), types, linalg over faer, bordered solver, problem trait, test problems | `Tests.Stage0` ported and passing | **Done**, October 2026: Stage 0 ported, mutation-tested; CI with tests, clippy, wasm |
| **M2** | bifurcata-rs | Newton, PALC engine, equilibrium system, stability, LP/BP/H detection with location; serialisation, comparator, `run` and `compare` commands; `antimony` adapter | Equilibrium baselines (built-in + `ant_*` + conserved totals) agree to 6 s.f.; PP2 closed forms | **Done**, October 2026: Newton, Continuation, Bialternate, Scale and Serialise (equilibrium parts) ported and mutation-tested; 4 built-in + 19 `ant_*` + `csum_edelstein` baselines agree in kinds, counts, locations to 6 s.f. **and** every sampled point to 1e-6 (identical point counts); PP2 closed forms to 1e-8; Lab 2's branch point to 1e-8. Not covered: `moiety3` (SBML input), `ant_pp2_switch` (switching, M3) |
| **M3** | bifurcata-rs | Directional derivatives, normal forms (fold `a`, Hopf `l1`, BP), branch switching (`run --switch`) | Normal-form coefficients to 3 s.f. with correct signs; Brusselator `l1 = −0.5`; PP2 switches | — |
| **M4** | bifurcata-rs | Codim-2 fold and Hopf curves; CP/BT/ZH/GH/HH detection; CP/BT/ZH normal forms; `curve` command | `Tests.Codim2` ported; `bistable_fold`, `lorenz84_hopf` baselines; MatCont §8.1.5 values | — |
| **M5** | websim | Bifurcation view in the app: one-parameter branches growing live with stability colouring and markers; switching at branch points; two-parameter fold/Hopf curves; CSV export; published to GitHub Pages | Usable in the browser on the bundled models | Started, October 2026: a Bifurcation view (one parameter, both directions, live growth, stability styling, LP/BP/H markers and a list) |
| **Review** | — | Decide on stages 4–6 (periodic orbits, cycle bifurcations, automatic diagrams) | — | — |

## 9. Size estimate

The spec estimated stages 0–3 at about 8,700 lines of Delphi. The Rust code
should be of similar size, roughly:

| Part | Lines (approx.) |
|---|---|
| websim: conservation analysis + reduced model | 600 |
| websim: steady-state solver | 400 |
| bifurcata: stages 0–3 library | 6,000–7,000 |
| bifurcata: harness, serialisation, comparator | 1,500 |
| Tests (ported) | 5,000 |

The original effort was dominated by discovering the problems now recorded in
CLAUDE.md; a rewrite that starts from those notes should go considerably
faster, but stage 3 (codim-2, "the single most error-prone derivation in the
library") deserves the most care.

## 10. Risks and open questions

1. **Finite-difference details** differ from libRoadRunner's elasticities, so
   numbers differ in the last digits; acceptance is set at 6 s.f. for that
   reason.
2. **faer is pre-1.0**: pin the version; wrap it behind `linalg` so an API
   change touches one module.
3. **Eigenvector conventions**: signs and normalisation are arbitrary in any
   eigensolver; every quantity that depends on them needs the CLAUDE.md
   conventions, and tests that would catch a missing one.
4. **Hopf test at large n**: the bialternate product is O(n⁶); at n = 30 it is
   fine (spec §6.3). Large models fall back to eigenvalue monitoring.
5. **Browser performance**: release builds only for the published site; the
   engine's iterator design lets the UI stay responsive.
6. **Open**: repository `sys-bio/bifurcata-rs` (created);
   whether `bifurcata` should also write the Delphi harness's exact command-line
   interface, so the two can be driven by the same regression scripts.
