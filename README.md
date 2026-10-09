# websim

A simulator for biochemical models, written in Rust with
[egui](https://github.com/emilk/egui). Type a model, move the sliders, and
watch the time course, the phase plane, the steady state and its bifurcation
diagram update. It runs in the browser (WebAssembly) and as a desktop app.

**Try it:** https://sys-bio.github.io/websim/

- [For users](#for-users)
- [For developers](#for-developers)

---

# For users

## What it does

- **Time courses.** Simulate a model written in a subset of
  [Antimony](https://tellurium.readthedocs.io/en/latest/antimony.html) and
  plot its species and reaction rates over time, with a phase plane beneath.
  Every parameter and initial value gets a slider, and the plots update as
  you drag.
- **Steady states.** Find the steady state and its stability (eigenvalues).
  Conserved moieties are found automatically and the analysis runs on the
  reduced system, so models with conservation laws just work.
- **Bifurcation diagrams.** Follow a steady state as a parameter varies and
  see where it changes character: folds (saddle-nodes), Hopf points (where
  oscillations are born) and branch points. This uses the
  [bifurcata](https://github.com/sys-bio/bifurcata-rs) library.
- **Export.** Save or copy the time course as CSV.

## The window

The **left panel** holds the model editor, an **Examples** menu, a slider for
each parameter and initial value, the simulation settings, and a collapsible
**Steady state** section.

The **central panel** has two tabs:

- **Time course**: tick the species and rates to plot; *Fix y-axis* keeps
  the scale while you move sliders. The phase plane below plots any two
  quantities against each other.
- **Bifurcation**: equilibrium branches against a parameter (see below).

In every plot, drag to pan, scroll to zoom, and double-click to reset the view.

## Writing a model

```
// Comments start with //, or are enclosed in /* */.
J1: $Xo -> 2 S1; k1*Xo       // a reaction: name, stoichiometry, rate law
S1 + S2 -> ; k2*S1*S2        // unnamed; an empty side is a source or sink
v := Vm*S1/(Km + S1)         // a rule, recomputed at every step
x' = -k*x                    // a rate rule (an ODE written directly)
k1 = 0.1; S1 = 0             // parameters and initial values
const Xo                     // a boundary species, held constant (or write $Xo)
E1: at (time > 10 && S1 < 2): k1 = k1/2, S1 = 0    // an event
```

A `model name ... end` wrapper is optional. Not supported yet: event delays
and options, functions, compartments and units, and initial assignments to
species.

**For bifurcation analysis, write real chemistry as reactions.**
`x -> y; k*x` tells the program that what leaves `x` arrives in `y`, which
is how it finds conservation laws. Writing the same thing as two separate
ODEs hides them, and the steady state then becomes impossible to continue.

## Bifurcation diagrams

1. Open the **Bifurcation** tab.
2. Choose the parameter to **vary** (conserved totals such as `_CSUM0` are
   offered too), its range, and the **max step** along the curve.
3. Press **Run**. The branch is started from the steady state at the current
   slider values and followed in both directions, growing as you watch;
   **Stop** halts it.
4. Choose which species to **plot**.

On the diagram, **solid** stretches are stable steady states and **dashed**
ones unstable. Located points are marked and listed with their parameter
values:

| Mark | Meaning |
|---|---|
| **LP** | A fold (limit point, saddle-node): two steady states meet and vanish. A pair of folds bounds a region of bistability. |
| **H** | A Hopf point: a pair of eigenvalues crosses the imaginary axis and oscillations appear or disappear. ω is their angular frequency there. |
| **BP** | A branch point: two branches of steady states cross. |
| NS0 | A neutral saddle: listed for completeness, but not a bifurcation. |

If a branch looks jagged, or a point seems out of place, lower the max step
and run again: located points are only as accurate as the step that found
them.

**Examples → Bifurcation examples** has nineteen models ready to run, among
them the Tyson–Novak cell-cycle models, the Edelstein and bistable switches,
the Gray–Scott isola, and models from MatCont's and AUTO's documentation.
Choosing one sets the parameter, range and step sizes; just press **Run**.

A model can carry these settings itself, in a comment block:

```
/*
[bifurcation]
parameter: m
min: 0
max: 2
ds: 0.001
dsMax: 0.005
maxSteps: 10000
plot: CycBT
*/
```

`ds` is the initial step, `dsMax` the maximum step and `maxSteps` the point
budget; anything left out keeps its default.

Not yet available: classifying Hopf points as super- or subcritical, and
switching onto the other branch at a branch point. Both are next in
bifurcata. Following periodic orbits (limit cycles) and two-parameter
diagrams come later.

---

# For developers

## Layout

```
src/                  the egui app
  main.rs             the window, side panel, time course and phase plane
  bifurcation.rs      the Bifurcation tab (on top of bifurcata)
  export.rs           saving a file: a dialog on the desktop, a download in the browser
crates/model/         websim-model: the model layer, no UI code
  src/antimony.rs     the Antimony-subset parser
  src/model.rs        the model, expressions, simulation with events,
                      the reduced system, the examples
  src/conservation.rs conserved moieties, independent species, link matrix
  src/steady.rs       steady states and stability
  src/newton.rs       damped Newton with Levenberg–Marquardt
  src/ode.rs, solvers.rs   RK4, and BDF/ESDIRK34/Tsit45 via diffsol
  models/             the bifurcation examples, each with a [bifurcation] block
docs/                 the plan for the bifurcation work and its progress
```

The model crate is separate from the app so that
[bifurcata-rs](https://github.com/sys-bio/bifurcata-rs) can build on it:
its `antimony` feature continues websim models directly.

## Building and testing

```
cargo run --release                             desktop app
cargo test --workspace                          all tests
cargo check --target wasm32-unknown-unknown     the browser build compiles
trunk serve                                     browser version at http://127.0.0.1:8080
```

The browser build needs `rustup target add wasm32-unknown-unknown` and
[Trunk](https://trunkrs.dev).

Pushing to `main` builds the web version with Trunk and publishes it to
GitHub Pages (`.github/workflows/pages.yml`).

## The bifurcata dependency

bifurcata is a git dependency with its `antimony` feature. That feature
depends on websim-model through *this* repository, so `Cargo.toml` patches it
back to the local `crates/model`: the app and bifurcata then share one copy
of the model types.

To build against a local, unpushed checkout of bifurcata-rs beside this one:

```
cargo run --config 'patch."https://github.com/sys-bio/bifurcata-rs".bifurcata.path="../bifurcata-rs"'
```

When releasing a change to both: push bifurcata first, then run
`cargo update -p bifurcata` here to pin the new commit in `Cargo.lock`, then
push websim — the site build fetches bifurcata from GitHub.

## Things to know

- **Everything must run in the browser**: pure Rust, no threads, no
  `std::time::Instant`. faer is used without default features, because its
  thread pool does not compile for WebAssembly.
- **Jacobians are finite differences**, never symbolic: symbolic derivatives
  of enzyme rate laws blow up and are slow to evaluate.
- **Steady states are found on the reduced system** (independent species
  only); the full Jacobian of a network with conservation laws is singular.
- `crates/model/tests/delphi_models.rs` checks against the Delphi version of
  Bifurcata when it is present on the machine, and skips otherwise.
- `CLAUDE.md` holds the numerical lessons and pitfalls collected so far.
