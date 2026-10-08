//! Steady states of a model, on its reduced system (independent species only).
//!
//! The route follows the Bifurcata specification (§6.1) and the Delphi
//! `TRoadRunnerProblem.FindSteadyState`:
//!
//! 1. **Damped Newton from the initial values** — Armijo line search,
//!    Levenberg–Marquardt fallback, residual scaling from the starting rates.
//! 2. If that fails, or lands outside the physical region: **integrate from the
//!    initial values towards an attractor** (t = 3000, BDF) and polish with
//!    Newton. A trajectory cannot cross a pole of a rate law — it would have to
//!    pass through an infinite rate — so it stays in the physical region by
//!    construction, which a Newton step does not. Restarting from the model's
//!    own initial values matters: a failed solve leaves the state wherever it
//!    gave up, possibly beyond a pole.
//! 3. **Physicality.** A negative concentration is not a solution of a reaction
//!    network; it usually means a solver crossed a pole. But for a model whose
//!    states are not concentrations (an ODE written as source reactions),
//!    negative values are simply where the solution lives. So if no route
//!    finds a non-negative steady state, a converged negative one is returned
//!    with a warning rather than refused.
//!
//! Time-dependent rate laws are evaluated at t = 0, and events are ignored:
//! a steady state is a property of the autonomous system.

use crate::linalg::Matrix;
use crate::model::Model;
use crate::newton::{NewtonOptions, NewtonSolver, NonlinearSystem};
use crate::solvers::{Method, SolverSettings};

/// How long to integrate towards an attractor before polishing (route 2).
const PRESIMULATION_TIME: f64 = 3000.0;

/// A steady state and its stability.
#[derive(Clone, Debug)]
pub struct SteadyState {
    /// All species, in the model's order.
    pub state: Vec<f64>,
    /// The independent species only (the reduced system's state).
    pub reduced: Vec<f64>,
    /// The conserved totals that were held fixed.
    pub totals: Vec<f64>,
    /// Eigenvalues of the reduced Jacobian as (re, im), largest real part first.
    pub eigenvalues: Vec<(f64, f64)>,
    pub stability: Stability,
    /// The largest |rate| of the reduced system at the steady state.
    pub max_rate: f64,
    /// How it was found.
    pub route: String,
    /// Set when the result has negative species (see the module notes).
    pub warning: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Stability {
    /// Every eigenvalue has a negative real part.
    Stable,
    /// At least one eigenvalue has a positive real part.
    Unstable,
    /// The largest real part is zero to within rounding: linearisation cannot decide.
    Marginal,
}

/// Find a steady state of `model` with parameters `params` (indexed as
/// [`Model::parameter_values`]), starting from full state `x0`. The conserved
/// totals are those of `x0`.
pub fn find_steady_state(model: &Model, x0: &[f64], params: &[f64]) -> Result<SteadyState, String> {
    let conservation = model.conservation();
    let totals = conservation.totals(x0);
    let u0 = conservation.reduce(x0);
    let free = Reduced { model, totals: &totals, params, nonnegative: false };
    // Keeping the species non-negative only makes sense if they start that way.
    let starts_physical = x0.iter().all(|v| *v >= 0.0);
    let kept_physical = Reduced { nonnegative: true, ..free };

    // A converged point outside the physical region, kept in case nothing better turns up.
    let mut fallback: Option<(Vec<f64>, String, String)> = None;
    // A physical but unstable steady state, kept while we look for a stable one.
    let mut unstable: Option<SteadyState> = None;
    let mut failures = Vec::new();

    // Try one route. Return from the function on a physical, stable steady
    // state; keep a physical unstable one, since a model with several steady
    // states usually means the one it settles into.
    macro_rules! attempt {
        ($system:expr, $start:expr, $label:expr) => {
            match newton($system, $start) {
                Ok((u, detail)) => {
                    let route = format!("{} ({detail})", $label);
                    match negative_species(model, &free.full(&u)) {
                        None => {
                            let found = finish(&free, u, route, None);
                            if found.stability != Stability::Unstable {
                                return Ok(found);
                            }
                            if unstable.is_none() {
                                unstable = Some(found);
                            }
                        }
                        Some(why) => {
                            failures.push(format!("{} reached {why}", $label));
                            if fallback.is_none() {
                                fallback = Some((u, route, why));
                            }
                        }
                    }
                }
                Err(e) => failures.push(format!("{}: {e}", $label)),
            }
        };
    }

    // Route 1: damped Newton from the initial values — first kept inside the
    // physical region, so it cannot jump past a pole of a rate law; then free,
    // for models whose steady state genuinely has negative values.
    if starts_physical {
        attempt!(&kept_physical, &u0, "Newton from the initial values, keeping species non-negative");
    }
    attempt!(&free, &u0, "Newton from the initial values");

    // Route 2: integrate towards an attractor from the initial values, then
    // polish. If the attractor is a limit cycle, its time average usually lies
    // near the unstable steady state inside it, so try from there as well.
    let settings = SolverSettings { method: Method::Bdf, rtol: 1e-8, atol: 1e-10, output_points: 300, ..Default::default() };
    let trajectory = model.simulate_from(x0.to_vec(), params.to_vec(), PRESIMULATION_TIME, &settings);
    if let Some(why) = &trajectory.stopped_early {
        failures.push(format!("integration stopped early: {why}"));
    }
    if let Some(end) = trajectory.y.last() {
        let reached = trajectory.t.last().copied().unwrap_or(0.0);
        let end_u = conservation.reduce(end);
        let half = trajectory.y.len() / 2;
        let tail = &trajectory.y[half..];
        let mean: Vec<f64> = (0..x0.len()).map(|i| tail.iter().map(|y| y[i]).sum::<f64>() / tail.len() as f64).collect();
        let mean_u = conservation.reduce(&mean);
        let end_label = format!("integrated to t = {reached}, then Newton");
        let mean_label = format!("integrated to t = {reached}, then Newton from the trajectory's time average");
        if starts_physical {
            attempt!(&kept_physical, &end_u, end_label.clone());
            attempt!(&kept_physical, &mean_u, mean_label);
        }
        attempt!(&free, &end_u, end_label);
    }

    // No stable steady state: the unstable one is still the answer — this is
    // what an oscillator, whose trajectories end on a limit cycle, has.
    if let Some(found) = unstable {
        return Ok(found);
    }

    // Nothing physical: a converged solution with negative values is still
    // the right answer for a model whose states are not concentrations.
    let system = free;
    if let Some((u, route, why)) = fallback {
        let warning = format!(
            "{why}. No steady state without negative values was found, so this one is used. \
             That is correct for a model whose states are not concentrations; for a reaction \
             network it usually means the solver crossed a pole and the state is not real."
        );
        return Ok(finish(&system, u, route, Some(warning)));
    }
    Err(format!("No steady state found. {}", failures.join("; ")))
}

/// Run Newton on the reduced system, scaled by the rates at the start.
fn newton(system: &Reduced, u0: &[f64]) -> Result<(Vec<f64>, String), String> {
    let mut u = u0.to_vec();
    // The Jacobian comes from finite differences, with a relative error
    // around 1e-8, so a Newton step means nothing once the condition number
    // passes ~1e8: switch to Levenberg–Marquardt's minimum-norm step there
    // (what NLEQ2's rank reduction does in libRoadRunner). A Jacobian that is
    // exactly singular at the start, as hopf2folds' is at the origin, looks
    // merely ill-conditioned through the noise, and a Newton step from it
    // heads off along the null direction.
    let options = NewtonOptions { max_iterations: 200, rcond_threshold: 1e-8, ..Default::default() };
    let mut solver = NewtonSolver::new(options);
    // Scale variables by their size and equations by how strongly they respond
    // to the state (see `scale_from_jacobian`), so a model whose species and
    // rates span orders of magnitude isn't judged by its largest ones alone.
    solver.scale_from_jacobian(system, &u);
    let report = solver.solve(system, &mut u)?;
    if !u.iter().all(|v| v.is_finite()) {
        return Err("the solution is not finite".to_owned());
    }
    let lm = if report.used_levenberg { ", with Levenberg–Marquardt steps" } else { "" };
    Ok((u, format!("{} iterations{lm}", report.iterations)))
}

/// The first species that is negative beyond rounding, described.
fn negative_species(model: &Model, x: &[f64]) -> Option<String> {
    let scale = x.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-300);
    x.iter()
        .zip(&model.species)
        .find(|(v, _)| **v < -1e-10 * scale)
        .map(|(v, s)| format!("{} = {v:.6e}", s.name))
}

fn finish(system: &Reduced, u: Vec<f64>, route: String, warning: Option<String>) -> SteadyState {
    let state = system.full(&u);
    let jacobian = system.jacobian(&u);
    let eigenvalues = jacobian.eigenvalues().unwrap_or_default();
    let mut rates = vec![0.0; u.len()];
    system.residual(&u, &mut rates);
    let max_rate = rates.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    SteadyState {
        state,
        reduced: u,
        totals: system.totals.to_vec(),
        stability: classify(&eigenvalues),
        eigenvalues,
        max_rate,
        route,
        warning,
    }
}

/// Stable, unstable or marginal, from the largest real part. "Zero" is judged
/// relative to the spectrum's size, because the Jacobian's entries come from
/// finite differences and carry a relative error around 1e-8.
fn classify(eigenvalues: &[(f64, f64)]) -> Stability {
    let size = eigenvalues.iter().fold(0.0f64, |m, (re, im)| m.max(re.hypot(*im))).max(1e-300);
    match eigenvalues.first() {
        None => Stability::Stable,
        Some((re, _)) if re.abs() <= 1e-7 * size => Stability::Marginal,
        Some((re, _)) if *re > 0.0 => Stability::Unstable,
        Some(_) => Stability::Stable,
    }
}

/// The reduced system `N_R v(L u + T) = 0` as a nonlinear system for Newton.
#[derive(Clone, Copy)]
struct Reduced<'a> {
    model: &'a Model,
    totals: &'a [f64],
    params: &'a [f64],
    /// Keep every species (independent and dependent) non-negative.
    nonnegative: bool,
}

impl Reduced<'_> {
    fn full(&self, u: &[f64]) -> Vec<f64> {
        self.model.conservation().full_state(u, self.totals)
    }
}

impl NonlinearSystem for Reduced<'_> {
    fn dim(&self) -> usize {
        self.model.conservation().independent.len()
    }

    fn residual(&self, u: &[f64], r: &mut [f64]) {
        r.copy_from_slice(&self.model.reduced_rates(0.0, u, self.totals, self.params));
    }

    fn jacobian(&self, u: &[f64]) -> Matrix {
        self.model.reduced_jacobian(0.0, u, self.totals, self.params)
    }

    fn admissible(&self, u: &[f64]) -> bool {
        if !self.nonnegative {
            return true;
        }
        // Dependent species are recomputed from the totals, so allow rounding below zero.
        let x = self.full(u);
        let scale = x.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        x.iter().all(|v| *v >= -1e-12 * (1.0 + scale))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn steady(text: &str) -> (Model, SteadyState) {
        let model = Model::parse(text).unwrap();
        let ss = find_steady_state(&model, &model.initial_state(), &model.parameter_values()).unwrap();
        (model, ss)
    }

    fn value(model: &Model, ss: &SteadyState, name: &str) -> f64 {
        ss.state[model.species.iter().position(|s| s.name == name).unwrap()]
    }

    const EDELSTEIN: &str = "\
        R0: -> X; kb\n R1: X -> 2X; k1*A*X\n R2: 2X -> X; km1*X^2\n\
        R3: X + E -> C; k2*X*E\n R4: C -> X + E; km2*C\n R5: C -> E; k3*C\n\
        X = 0.2; E = 1; C = 0\n A = 4; k1 = 1; km1 = 0.2; k2 = 50; km2 = 1; k3 = 1; kb = 0.1";

    #[test]
    fn edelstein_reduces_to_two_species() {
        let (model, ss) = steady(EDELSTEIN);
        assert_eq!(model.species_names(), ["X", "E", "C"]);
        let c = model.conservation();
        assert_eq!(c.independent, [0, 1]);
        assert_eq!(c.describe(&model.species_names(), &model.initial_state()), ["C + E = 1"]);

        assert_eq!(ss.reduced.len(), 2, "Newton ran on the reduced system");
        assert!(ss.max_rate < 1e-9, "rates vanish: {}", ss.max_rate);
        assert!((value(&model, &ss, "E") + value(&model, &ss, "C") - 1.0).abs() < 1e-12, "E + C = 1 is kept");
        assert!(ss.warning.is_none());
        // The full Jacobian, by contrast, is singular: the conservation law's zero eigenvalue.
        let full = model.full_jacobian(0.0, &ss.state, &model.parameter_values()).eigenvalues().unwrap();
        let smallest = full.iter().map(|(re, im)| re.hypot(*im)).fold(f64::MAX, f64::min);
        assert!(smallest < 1e-8, "the full Jacobian has a zero eigenvalue: {full:?}");
    }

    /// Spec §11.3: the reduced Jacobian's eigenvalues are the nonzero eigenvalues of N ε.
    #[test]
    fn reduced_and_full_spectra_agree() {
        let (model, ss) = steady(EDELSTEIN);
        let mut full = model.full_jacobian(0.0, &ss.state, &model.parameter_values()).eigenvalues().unwrap();
        full.retain(|(re, im)| re.hypot(*im) > 1e-8);
        assert_eq!(full.len(), ss.eigenvalues.len());
        for ((a, b), (c, d)) in full.iter().zip(&ss.eigenvalues) {
            assert!((a - c).abs() < 1e-6 * a.abs().max(1.0) && (b - d).abs() < 1e-6 * b.abs().max(1.0),
                "full {full:?} vs reduced {:?}", ss.eigenvalues);
        }
    }

    /// S1 <-> S2 <-> S3 at steady state has k1 S1 = k2 S2 and k3 S2 = k4 S3, so
    /// S2 = 1.75 S1 and S3 = 7.875 S1, and the total 15 gives S1 = 15/10.625.
    #[test]
    fn three_species_chain_matches_its_closed_form() {
        let (model, ss) = steady(
            "J1: S1 -> S2; k1*S1 - k2*S2\n J2: S2 -> S3; k3*S2 - k4*S3\n\
             S1 = 10; S2 = 3; S3 = 2\n k1 = 0.7; k2 = 0.4; k3 = 0.9; k4 = 0.2",
        );
        assert_eq!(model.conservation().law_count(), 1);
        let s1 = 15.0 / 10.625;
        for (name, expected) in [("S1", s1), ("S2", 1.75 * s1), ("S3", 7.875 * s1)] {
            let got = value(&model, &ss, name);
            assert!((got - expected).abs() < 1e-9, "{name} = {got}, expected {expected}");
        }
        assert_eq!(ss.stability, Stability::Stable);
    }

    /// The Brusselator: steady state (A, B/A), Jacobian
    /// [[B - 1, A²], [-B, -A²]] there, Hopf at B = 1 + A².
    #[test]
    fn brusselator_steady_state_jacobian_and_stability() {
        let text = |b: f64| format!(
            "J1: -> X; A\n J2: X -> Y; B*X\n J3: 2X + Y -> 3X; X^2*Y\n J4: X -> ; X\n\
             A = 1; B = {b}; X = 1; Y = 1"
        );
        let (model, ss) = steady(&text(1.5));
        assert_eq!(model.conservation().law_count(), 0);
        assert!((value(&model, &ss, "X") - 1.0).abs() < 1e-10 && (value(&model, &ss, "Y") - 1.5).abs() < 1e-10);

        let j = model.reduced_jacobian(0.0, &ss.reduced, &ss.totals, &model.parameter_values());
        let exact = [[0.5, 1.0], [-1.5, -1.0]];
        for r in 0..2 {
            for c in 0..2 {
                assert!((j[(r, c)] - exact[r][c]).abs() < 1e-7, "J[{r}][{c}] = {}", j[(r, c)]);
            }
        }
        assert_eq!(ss.stability, Stability::Stable, "B = 1.5 is below the Hopf at B = 2");
        assert!(ss.eigenvalues[0].1 != 0.0, "a complex pair: {:?}", ss.eigenvalues);

        let (_, ss) = steady(&text(3.0));
        assert_eq!(ss.stability, Stability::Unstable, "B = 3 is past the Hopf");
    }

    /// A species that is only ever a catalyst never changes: its row of N is
    /// zero, so it is a conservation law on its own and leaves the reduced system.
    #[test]
    fn catalyst_only_species_is_constant() {
        let (model, ss) = steady("J1: S + E -> P + E; k*S*E\n J2: P -> S; k2*P\n S = 4; P = 0; E = 2; k = 1; k2 = 0.5");
        let c = model.conservation();
        assert_eq!(c.law_count(), 2, "E is constant, and S + P is conserved");
        assert_eq!(ss.reduced.len(), 1);
        assert_eq!(value(&model, &ss, "E"), 2.0);
        assert!((value(&model, &ss, "S") + value(&model, &ss, "P") - 4.0).abs() < 1e-12);
    }

    /// x' = -x - 2 settles at x = -2: negative, so not physical for a reaction
    /// network, but right for an ODE. It is returned, with a warning.
    #[test]
    fn negative_steady_state_comes_with_a_warning() {
        let (model, ss) = steady("x' = -x - 2\n x = 1");
        assert!((value(&model, &ss, "x") + 2.0).abs() < 1e-10);
        assert!(ss.warning.as_deref().is_some_and(|w| w.contains("x = -2")), "{:?}", ss.warning);
    }
}
