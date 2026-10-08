//! Shared simulation types, and a fixed-step 4th-order Runge–Kutta solver.
//! The adaptive and stiff solvers from diffsol are in [`crate::solvers`].
//!
//! A simulation with events runs in segments: a solver integrates until an event
//! fires (or the end time is reached), the model applies the event, and a new
//! segment starts from the changed state. See `Model::simulate`.

/// The result of a simulation: `y[i]` is the state vector at time `t[i]`.
/// At an event, the time appears twice: the state just before, then just after.
#[derive(Default)]
pub struct Solution {
    pub t: Vec<f64>,
    pub y: Vec<Vec<f64>>,
    /// Parameter values in force from each point on: `(point index, values)`.
    /// The first entry is for index 0; events add more.
    pub param_changes: Vec<(usize, Vec<f64>)>,
    /// Events that fired: (time, index of the event in the model).
    pub events: Vec<(f64, usize)>,
    /// Why integration stopped before the end time, if it did.
    pub stopped_early: Option<String>,
    /// Number of solver steps taken.
    pub steps: usize,
    /// Number of times the model's rates were evaluated.
    pub rate_evaluations: usize,
}

/// How a solver segment ended.
pub enum SegmentEnd {
    /// Reached the end time.
    Done,
    /// An event's trigger became true at time `t`, where the state was `y`.
    Event { t: f64, y: Vec<f64> },
    /// The solver couldn't continue.
    Failed(String),
}

/// The model's right-hand side: `rates(t, y, out)` writes dy/dt into `out`.
pub type Rates<'a> = dyn Fn(f64, &[f64], &mut [f64]) + 'a;

/// Called after each solver step from `t0` to `t1`, with a function giving the
/// state at any time in that step. Returns the time an event fires, if one does.
pub type EventCheck<'a> = dyn FnMut(f64, f64, &dyn Fn(f64) -> Vec<f64>) -> Option<f64> + 'a;

/// One classic RK4 step of size `h` from `(t, y)`.
fn rk4_step(f: &Rates, t: f64, y: &[f64], h: f64) -> Vec<f64> {
    let n = y.len();
    // y + a·k, element by element.
    let offset = |k: &[f64], a: f64| -> Vec<f64> { y.iter().zip(k).map(|(yi, ki)| yi + a * ki).collect() };
    let (mut k1, mut k2, mut k3, mut k4) = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    f(t, y, &mut k1);
    f(t + h / 2.0, &offset(&k1, h / 2.0), &mut k2);
    f(t + h / 2.0, &offset(&k2, h / 2.0), &mut k3);
    f(t + h, &offset(&k3, h), &mut k4);
    (0..n)
        .map(|i| y[i] + h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]))
        .collect()
}

/// Integrate from `(t0, y0)` towards `t_end` with fixed RK4 steps of size `h`,
/// appending each step to `solution`. Steps stay on the grid 0, h, 2h, …
/// even when a segment starts between grid points (after an event).
pub fn rk4_segment(
    f: &Rates,
    y0: &[f64],
    t0: f64,
    t_end: f64,
    h: f64,
    solution: &mut Solution,
    check_events: &mut EventCheck,
) -> SegmentEnd {
    let mut t = t0;
    let mut y = y0.to_vec();
    while t < t_end {
        let mut next = ((t / h).floor() + 1.0) * h;
        if next - t < 1e-9 * h {
            next += h; // `t` was already on the grid, up to rounding
        }
        let next = next.min(t_end);

        let y_next = rk4_step(f, t, &y, next - t);
        solution.steps += 1;
        if !y_next.iter().all(|v| v.is_finite()) {
            return SegmentEnd::Failed(format!(
                "The solution became infinite or undefined at t = {next:.4}. \
                 If the model is stiff, try the BDF solver."
            ));
        }

        // The state part-way through the step: a shorter RK4 step from its start.
        let y_at = |s: f64| if s >= next { y_next.clone() } else { rk4_step(f, t, &y, s - t) };
        if let Some(t_event) = check_events(t, next, &y_at) {
            return SegmentEnd::Event { t: t_event, y: y_at(t_event) };
        }

        t = next;
        y = y_next;
        solution.t.push(t);
        solution.y.push(y.clone());
    }
    SegmentEnd::Done
}
