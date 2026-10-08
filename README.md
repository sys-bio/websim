# websim

A small ODE simulator for biochemical models, written in Rust with
[egui](https://github.com/emilk/egui). It runs as a desktop app and in the browser
(WebAssembly).

**Try it:** https://sys-bio.github.io/websim/

## Features

- Models written in a basic subset of [Antimony](https://tellurium.readthedocs.io/en/latest/antimony.html):
  reactions with stoichiometry and boundary species (`$`), rate rules (`x' = ...`),
  rules (`:=`), parameters, and events (`at (condition): x = value`)
- Sliders for every parameter and initial value, with live re-simulation
- Solvers: BDF, ESDIRK34 and Tsit45 from [diffsol](https://github.com/martinjrobins/diffsol),
  plus fixed-step RK4
- Steady states with stability (eigenvalues), on the reduced system: conservation
  analysis finds conserved moieties automatically
- Plots of species and reaction rates, and a phase plane; CSV export and copy

## Building

Desktop:

```
cargo run --release
```

Web (needs `rustup target add wasm32-unknown-unknown` and [Trunk](https://trunkrs.dev)):

```
trunk serve
```

Pushing to `main` builds the web version and publishes it to GitHub Pages
(see `.github/workflows/pages.yml`).
