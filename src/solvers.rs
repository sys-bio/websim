//! Solver choice, and the adaptive solvers from the `diffsol` crate.

use std::cell::RefCell;

use diffsol::{
    NalgebraLU, NalgebraMat, OdeBuilder, OdeEquations, OdeSolverMethod, OdeSolverStopReason,
    Vector,
};

use crate::ode::{EventCheck, Rates, SegmentEnd, Solution, rk4_segment};

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Method {
    /// Fixed-step classic Runge–Kutta (our own, in `ode.rs`).
    Rk4,
    /// Adaptive explicit Runge–Kutta (Tsitouras 4/5): good for non-stiff models.
    Tsit45,
    /// Variable-order BDF: for stiff models.
    Bdf,
    /// Adaptive implicit Runge–Kutta (ESDIRK 3/4): for stiff models.
    Esdirk34,
}

impl Method {
    pub const ALL: [Method; 4] = [Method::Bdf, Method::Esdirk34, Method::Tsit45, Method::Rk4];

    pub fn label(self) -> &'static str {
        match self {
            Method::Rk4 => "RK4 (fixed step)",
            Method::Tsit45 => "Tsit45 (adaptive, non-stiff)",
            Method::Bdf => "BDF (stiff)",
            Method::Esdirk34 => "ESDIRK34 (stiff)",
        }
    }

    pub fn is_adaptive(self) -> bool {
        self != Method::Rk4
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SolverSettings {
    pub method: Method,
    /// Relative and absolute error tolerances for the adaptive solvers.
    pub rtol: f64,
    pub atol: f64,
    /// Number of steps for the fixed-step RK4 solver.
    pub rk4_steps: usize,
}

impl Default for SolverSettings {
    fn default() -> Self {
        Self {
            method: Method::Bdf,
            rtol: 1e-6,
            atol: 1e-8,
            rk4_steps: 5000,
        }
    }
}

impl SolverSettings {
    /// Fixed-step RK4 with the given number of steps.
    #[cfg(test)]
    pub fn rk4(steps: usize) -> Self {
        Self {
            method: Method::Rk4,
            rk4_steps: steps,
            ..Default::default()
        }
    }
}

/// How many evenly spaced points the adaptive solvers report over the whole run, for plotting.
const OUTPUT_POINTS: usize = 2000;

/// Give up after this many steps in a run, so a hard problem can't freeze the UI.
const MAX_STEPS: usize = 100_000;

/// Integrate from `(t0, y0)` towards `t_end` with the chosen solver, appending
/// output points to `solution`, until the end time or until `check_events` reports
/// that an event fires.
pub fn solve_segment(
    settings: &SolverSettings,
    rates: &Rates,
    y0: &[f64],
    t0: f64,
    t_end: f64,
    solution: &mut Solution,
    check_events: &mut EventCheck,
) -> SegmentEnd {
    if settings.method == Method::Rk4 {
        let h = t_end / settings.rk4_steps as f64;
        return rk4_segment(rates, y0, t0, t_end, h, solution, check_events);
    }

    let n = y0.len();

    // The implicit solvers need the Jacobian J = ∂rates/∂y, which diffsol asks for as
    // products J·v. We use a directional finite difference, J·v ≈ (f(y + h·v) − f(y)) / h,
    // rather than symbolic differentiation: derivatives of enzyme rate laws grow large
    // and can be slower to evaluate. diffsol builds the full Jacobian from n such
    // products at the same (t, y), so f(y) is cached between them.
    let cached_f: RefCell<Option<(f64, Vec<f64>, Vec<f64>)>> = RefCell::new(None);
    let jacobian_times = |t: f64, y: &[f64], v: &[f64], out: &mut [f64]| {
        let norm = |a: &[f64]| a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let v_norm = norm(v);
        if v_norm == 0.0 {
            out.fill(0.0);
            return;
        }
        let mut cache = cached_f.borrow_mut();
        let is_cached = matches!(&*cache, Some((ct, cy, _)) if *ct == t && cy.as_slice() == y);
        if !is_cached {
            let mut f = vec![0.0; n];
            rates(t, y, &mut f);
            *cache = Some((t, y.to_vec(), f));
        }
        let f = &cache.as_ref().unwrap().2;

        let h = f64::EPSILON.sqrt() * (1.0 + norm(y)) / v_norm;
        let shifted: Vec<f64> = y.iter().zip(v).map(|(yi, vi)| yi + h * vi).collect();
        rates(t, &shifted, out);
        for (o, fi) in out.iter_mut().zip(f) {
            *o = (*o - fi) / h;
        }
    };

    let initial = y0.to_vec();
    let problem = OdeBuilder::<NalgebraMat<f64>>::new()
        .t0(t0)
        .rtol(settings.rtol)
        .atol([settings.atol])
        .rhs_implicit(
            |y, _p, t, out| rates(t, y, out),
            |y, _p, t, v, out| jacobian_times(t, y, v, out),
        )
        .init(move |_p, _t, y| y.copy_from_slice(&initial), n)
        .build();
    let problem = match problem {
        Ok(problem) => problem,
        Err(e) => return SegmentEnd::Failed(format!("Couldn't set up the solver: {e}")),
    };

    let run = Run { t0, t_end, solution, check_events };
    match settings.method {
        Method::Bdf => match problem.bdf::<NalgebraLU<f64>>() {
            Ok(solver) => run.integrate(solver),
            Err(e) => SegmentEnd::Failed(format!("Couldn't start BDF: {e}")),
        },
        Method::Esdirk34 => match problem.esdirk34::<NalgebraLU<f64>>() {
            Ok(solver) => run.integrate(solver),
            Err(e) => SegmentEnd::Failed(format!("Couldn't start ESDIRK34: {e}")),
        },
        Method::Tsit45 => match problem.tsit45() {
            Ok(solver) => run.integrate(solver),
            Err(e) => SegmentEnd::Failed(format!("Couldn't start Tsit45: {e}")),
        },
        Method::Rk4 => unreachable!("handled above"),
    }
}

/// One segment of an adaptive-solver run.
struct Run<'s, 'c> {
    t0: f64,
    t_end: f64,
    solution: &'s mut Solution,
    check_events: &'s mut EventCheck<'c>,
}

impl Run<'_, '_> {
    /// Step the solver from `t0` to `t_end`, recording the solution on an evenly
    /// spaced output grid by interpolation, and checking for events after each step.
    fn integrate<'a, Eqn, S>(self, mut solver: S) -> SegmentEnd
    where
        Eqn: OdeEquations<T = f64> + 'a,
        S: OdeSolverMethod<'a, Eqn>,
    {
        let Run { t0, t_end, solution, check_events } = self;
        if let Err(e) = solver.set_stop_time(t_end) {
            return SegmentEnd::Failed(format!("Couldn't set the end time: {e}"));
        }

        // The output grid is t_end·k/OUTPUT_POINTS; start at the first point after t0.
        let grid = |k: usize| t_end * k as f64 / OUTPUT_POINTS as f64;
        let mut next = ((t0 / t_end) * OUTPUT_POINTS as f64).floor() as usize + 1;

        let mut t_prev = t0;
        loop {
            if solution.steps >= MAX_STEPS {
                return SegmentEnd::Failed(format!(
                    "Gave up after {MAX_STEPS} steps, at t = {t_prev:.4}. \
                     If the model is stiff, try BDF or ESDIRK34."
                ));
            }
            let result = solver.step();
            solution.steps += 1;
            let reached_end = match result {
                Ok(OdeSolverStopReason::TstopReached) => true,
                Ok(_) => false,
                Err(e) => {
                    return SegmentEnd::Failed(format!("The solver stopped at t = {t_prev:.4}: {e}"));
                }
            };
            let t_now = solver.state().t;
            let y_at = |t: f64| match solver.interpolate(t) {
                Ok(y) => y.clone_as_vec(),
                Err(_) => solver.state().y.clone_as_vec(),
            };

            // Record output up to the event, if one fires in this step, else up to t_now.
            let event = check_events(t_prev, t_now, &y_at);
            let record_until = event.unwrap_or(t_now);
            while next <= OUTPUT_POINTS && grid(next) <= record_until {
                let t = grid(next);
                if event.is_some() && t == record_until {
                    break; // the event time itself is recorded by the caller
                }
                solution.t.push(t);
                solution.y.push(y_at(t));
                next += 1;
            }
            if let Some(t_event) = event {
                return SegmentEnd::Event { t: t_event, y: y_at(t_event) };
            }
            if reached_end {
                return SegmentEnd::Done;
            }
            t_prev = t_now;
        }
    }
}
