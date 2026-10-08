//! Damped Newton with an Armijo line search and a Levenberg–Marquardt
//! fallback, for solving `F(x) = 0`.
//!
//! Follows the Bifurcata specification (§5.3, §6.1) and its Delphi
//! implementation `Bifurcata.Newton`.
//!
//! **Scaling** (Dennis & Schnabel ch. 7): metabolic models span µM to mM, so
//! an unscaled convergence test means nothing across that range. Two vectors of
//! typical values do the scaling:
//!
//! * `typical_x[i]` — the magnitude below which a change in `x[i]` doesn't matter;
//! * `typical_f[i]` — the magnitude below which a residual in equation `i` is negligible.
//!
//! Convergence is `max |F_i| / typical_f_i <= tol_residual`, or a scaled step
//! below `tol_step`. The line search uses the same scaled quantity,
//! `½ Σ (F_i / typical_f_i)²`, so it and the convergence test agree about what
//! progress means. The useful default for a real model is `typical_f` from the
//! rates at the starting point ([`NewtonSolver::typical_f_from_residual`]):
//! an absolute test would call a model whose rates are all ~1e-9 converged
//! before it started.

use crate::linalg::{Lu, Matrix};

/// A square nonlinear system `F(x) = 0`.
pub trait NonlinearSystem {
    fn dim(&self) -> usize;
    fn residual(&self, x: &[f64], r: &mut [f64]);
    fn jacobian(&self, x: &[f64]) -> Matrix;

    /// Whether `x` is allowed at all. The line search backs away from any trial
    /// point that is not, as if its merit were infinite — which is how a solve
    /// is kept inside the physical region of a reaction network, where the
    /// poles of the rate laws lie on the far side of zero.
    fn admissible(&self, _x: &[f64]) -> bool {
        true
    }
}

#[derive(Clone, Debug)]
pub struct NewtonOptions {
    pub max_iterations: usize,
    pub tol_residual: f64,
    pub tol_step: f64,
    pub use_line_search: bool,
    pub max_line_search_steps: usize,
    /// Sufficient-decrease constant of the Armijo condition.
    pub armijo_alpha: f64,
    /// Clamps on the backtracking factor from the quadratic model.
    pub min_backtrack: f64,
    pub max_backtrack: f64,
    pub use_levenberg_fallback: bool,
    /// Below this reciprocal condition number, take a Levenberg–Marquardt step.
    pub rcond_threshold: f64,
    pub lm_initial_mu: f64,
    pub lm_increase: f64,
    pub lm_decrease: f64,
    pub lm_max_mu: f64,
    /// Give up if the merit function grows by this factor.
    pub divergence_factor: f64,
}

impl Default for NewtonOptions {
    fn default() -> Self {
        Self {
            max_iterations: 10,
            tol_residual: 1e-10,
            tol_step: 1e-10,
            use_line_search: true,
            max_line_search_steps: 25,
            armijo_alpha: 1e-4,
            min_backtrack: 0.1,
            max_backtrack: 0.5,
            use_levenberg_fallback: true,
            rcond_threshold: 1e-12,
            lm_initial_mu: 1e-3,
            lm_increase: 10.0,
            lm_decrease: 0.1,
            lm_max_mu: 1e12,
            divergence_factor: 1e8,
        }
    }
}

/// The result of a successful solve.
#[derive(Clone, Debug)]
pub struct NewtonReport {
    pub iterations: usize,
    /// The scaled residual at the solution.
    pub scaled_residual: f64,
    pub used_levenberg: bool,
}

pub struct NewtonSolver {
    pub options: NewtonOptions,
    typical_x: Option<Vec<f64>>,
    typical_f: Option<Vec<f64>>,
}

impl NewtonSolver {
    pub fn new(options: NewtonOptions) -> Self {
        Self { options, typical_x: None, typical_f: None }
    }

    pub fn set_typical_x(&mut self, v: Vec<f64>) {
        self.typical_x = Some(v);
    }

    pub fn set_typical_f(&mut self, v: Vec<f64>) {
        self.typical_f = Some(v);
    }

    /// Set `typical_x` from the magnitudes in `x0` and `typical_f` from the
    /// Jacobian there: `typical_f[i] = Σ_j |J_ij| typical_x[j]`, how much
    /// equation i's residual changes for a typical change in the state. This is
    /// the recommended scaling for a model.
    ///
    /// It makes the convergence test mean what we want — a scaled residual of
    /// 1e-10 is a state error of about 1e-10 of its typical size — and, unlike
    /// scaling by `|F(x0)|`, it doesn't break when a rate happens to be zero at
    /// the starting point (common when a model starts from zero): that would
    /// give the equation a tiny typical size, weight it out of all proportion,
    /// and leave the line search able to take only microscopic steps.
    ///
    /// `typical_x[i]` is `|x0_i|`, floored at a thousandth of the largest
    /// `|x0_j|`, or 1 if the state is all zero.
    pub fn scale_from_jacobian(&mut self, system: &dyn NonlinearSystem, x0: &[f64]) {
        let largest = x0.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let typical_x: Vec<f64> = x0
            .iter()
            .map(|v| {
                let t = v.abs().max(1e-3 * largest);
                if t > 0.0 { t } else { 1.0 }
            })
            .collect();
        let j = system.jacobian(x0);
        let mut typical_f: Vec<f64> = (0..j.rows())
            .map(|i| j.row(i).iter().zip(&typical_x).map(|(a, t)| a.abs() * t).sum::<f64>())
            .map(|v: f64| if v.is_finite() { v } else { 0.0 })
            .collect();
        // An equation with no dependence on the state at x0 still needs a scale.
        let biggest = typical_f.iter().fold(0.0f64, |m, v| m.max(*v));
        let floor = (1e-6 * biggest).max(1e-300);
        for v in &mut typical_f {
            *v = v.max(floor);
        }
        self.typical_x = Some(typical_x);
        self.typical_f = Some(typical_f);
    }

    /// Set `typical_f` from `|F(x0)|` (the Delphi version's scaling).
    ///
    /// Each entry is floored at a millionth of the largest rate (and at
    /// `floor`), so an equation that happens to be exactly zero at the start —
    /// common when a model starts from zero concentrations — doesn't get a
    /// typical size of `floor` and its row scaled up by 1/`floor`, which would
    /// swamp every other equation and make the scaled Jacobian numerically
    /// singular.
    pub fn typical_f_from_residual(&mut self, system: &dyn NonlinearSystem, x0: &[f64], floor: f64) {
        let mut r = vec![0.0; system.dim()];
        system.residual(x0, &mut r);
        let largest = r.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let floor = floor.max(1e-6 * largest);
        self.typical_f = Some(r.iter().map(|v| v.abs().max(floor)).collect());
    }

    /// Solve `F(x) = 0`, starting from and updating `x`.
    pub fn solve(&self, system: &dyn NonlinearSystem, x: &mut [f64]) -> Result<NewtonReport, String> {
        let n = system.dim();
        if n == 0 {
            return Ok(NewtonReport { iterations: 0, scaled_residual: 0.0, used_levenberg: false });
        }
        if x.len() != n {
            return Err(format!("the starting point has {} values but the system has {n}", x.len()));
        }
        let o = &self.options;
        let typical_f = match &self.typical_f {
            Some(v) if v.len() == n => v.clone(),
            Some(v) => return Err(format!("typical_f has {} values, the system has {n}", v.len())),
            None => vec![1.0; n],
        };
        let typical_x = match &self.typical_x {
            Some(v) if v.len() == n => v.clone(),
            Some(v) => return Err(format!("typical_x has {} values, the system has {n}", v.len())),
            None => x.iter().map(|v| v.abs().max(1.0)).collect(),
        };

        let scale = |r: &[f64]| -> Vec<f64> { r.iter().zip(&typical_f).map(|(a, t)| a / t).collect() };
        let merit = |rs: &[f64]| 0.5 * rs.iter().map(|v| v * v).sum::<f64>();
        let max_abs = |v: &[f64]| v.iter().fold(0.0f64, |m, a| m.max(a.abs()));

        let mut r = vec![0.0; n];
        system.residual(x, &mut r);
        let mut rs = scale(&r);
        let mut f = merit(&rs);
        let initial_merit = f;
        if max_abs(&rs) <= o.tol_residual {
            return Ok(NewtonReport { iterations: 0, scaled_residual: max_abs(&rs), used_levenberg: false });
        }

        let mut mu = o.lm_initial_mu;
        let mut used_levenberg = false;
        let mut trial = vec![0.0; n];
        let mut trial_r = vec![0.0; n];

        for iteration in 1..=o.max_iterations {
            // Row i of J is divided by typical_f[i], matching the residual scaling.
            let mut j = system.jacobian(x);
            for row in 0..n {
                for col in 0..n {
                    j[(row, col)] /= typical_f[row];
                }
            }

            // The Newton step, unless J is singular or too ill-conditioned to trust.
            let newton_step = Lu::new(&j).ok().and_then(|lu| {
                let well_conditioned = !o.use_levenberg_fallback || lu.reciprocal_condition() >= o.rcond_threshold;
                well_conditioned.then(|| lu.solve(&rs.iter().map(|v| -v).collect::<Vec<_>>()))
            });
            let mut lm_step_taken = false;
            let mut step = match newton_step {
                Some(step) => step,
                None if o.use_levenberg_fallback => {
                    lm_step_taken = true;
                    used_levenberg = true;
                    levenberg_marquardt_step(&j, &rs, mu)
                        .map_err(|e| format!("Newton: the Levenberg–Marquardt system is singular at iteration {iteration} with mu = {mu:.3e} ({e})"))?
                }
                None => return Err(format!("Newton: the Jacobian is singular at iteration {iteration}")),
            };

            // The merit's slope along the step, from the gradient Jᵀ rs: the
            // identity slope = -2 f holds only for an exact Newton step.
            let gradient = j.mul_transpose_vec(&rs);
            let mut slope: f64 = gradient.iter().zip(&step).map(|(g, s)| g * s).sum();
            if slope >= 0.0 {
                // Not a descent direction (a badly polluted solve): steepest descent.
                step = gradient.iter().map(|g| -g).collect();
                slope = gradient.iter().zip(&step).map(|(g, s)| g * s).sum();
                lm_step_taken = true;
                if slope >= 0.0 {
                    return Err(format!("Newton: no descent direction at iteration {iteration}"));
                }
            }

            // Armijo backtracking with a quadratic model of the merit.
            let mut factor = 1.0;
            let mut accepted = false;
            let mut trial_merit = f;
            let steps = if o.use_line_search { o.max_line_search_steps } else { 1 };
            for _ in 0..steps {
                for k in 0..n {
                    trial[k] = x[k] + factor * step[k];
                }
                if !system.admissible(&trial) {
                    trial_merit = f64::INFINITY;
                    factor *= o.max_backtrack;
                    continue;
                }
                system.residual(&trial, &mut trial_r);
                let trs = scale(&trial_r);
                trial_merit = merit(&trs);
                if trial_merit.is_finite() && (!o.use_line_search || trial_merit <= f + o.armijo_alpha * factor * slope) {
                    accepted = true;
                    rs = trs;
                    r.copy_from_slice(&trial_r);
                    break;
                }
                let denominator = 2.0 * (trial_merit - f - factor * slope);
                let new_factor = if trial_merit.is_finite() && denominator > 0.0 {
                    -slope * factor * factor / denominator
                } else {
                    factor * o.max_backtrack
                };
                factor = new_factor.clamp(o.min_backtrack * factor, o.max_backtrack * factor);
            }

            if !accepted {
                if o.use_levenberg_fallback && mu < o.lm_max_mu {
                    // The line search stalled: lean harder on Levenberg–Marquardt and retry.
                    mu = (mu * o.lm_increase).min(o.lm_max_mu);
                    let lm = levenberg_marquardt_step(&j, &rs, mu);
                    if let Ok(lm) = lm {
                        for k in 0..n {
                            trial[k] = x[k] + lm[k];
                        }
                        if !system.admissible(&trial) {
                            return Err(format!(
                                "Newton: stalled at the edge of the allowed region at iteration {iteration} \
                                 (scaled residual {:.3e})",
                                max_abs(&rs)
                            ));
                        }
                        system.residual(&trial, &mut trial_r);
                        let trs = scale(&trial_r);
                        let m = merit(&trs);
                        if m.is_finite() && m < f {
                            used_levenberg = true;
                            let scaled_step = scaled_step_norm(&lm, x, &typical_x);
                            x.copy_from_slice(&trial);
                            r.copy_from_slice(&trial_r);
                            rs = trs;
                            f = m;
                            if max_abs(&rs) <= o.tol_residual || scaled_step <= o.tol_step {
                                return Ok(NewtonReport { iterations: iteration, scaled_residual: max_abs(&rs), used_levenberg });
                            }
                            continue;
                        }
                    }
                }
                return Err(format!(
                    "Newton: the line search failed at iteration {iteration} (scaled residual {:.3e})",
                    max_abs(&rs)
                ));
            }

            let applied: Vec<f64> = step.iter().map(|s| s * factor).collect();
            let scaled_step = scaled_step_norm(&applied, x, &typical_x);
            x.copy_from_slice(&trial);
            f = trial_merit;
            if lm_step_taken {
                mu = (mu * o.lm_decrease).max(1e-12);
            }

            if f > o.divergence_factor * initial_merit.max(1e-300) {
                return Err(format!("Newton: diverging at iteration {iteration}"));
            }
            let residual = max_abs(&rs);
            if residual <= o.tol_residual || (scaled_step <= o.tol_step && residual <= o.tol_residual.sqrt()) {
                return Ok(NewtonReport { iterations: iteration, scaled_residual: residual, used_levenberg });
            }
        }
        Err(format!(
            "Newton: no convergence in {} iterations (scaled residual {:.3e})",
            o.max_iterations,
            max_abs(&rs)
        ))
    }
}

/// `max_i |dx_i| / max(|x_i|, typical_x_i)`
fn scaled_step_norm(dx: &[f64], x: &[f64], typical_x: &[f64]) -> f64 {
    dx.iter()
        .zip(x)
        .zip(typical_x)
        .map(|((d, xi), t)| d.abs() / xi.abs().max(*t))
        .fold(0.0, f64::max)
}

/// Solve `(Jᵀ J + µ d I) δ = −Jᵀ F` (Dennis & Schnabel ch. 6; Kelley ch. 8),
/// where `d` is the largest diagonal entry of `Jᵀ J`, so that µ is a relative
/// damping that means the same whatever the scale of J. Forming the normal
/// matrix squares the condition number, which is acceptable here because this
/// path is only taken when J is already too ill-conditioned to use directly —
/// the damping is what restores solvability.
fn levenberg_marquardt_step(j: &Matrix, rs: &[f64], mu: f64) -> Result<Vec<f64>, String> {
    let n = j.cols();
    let mut normal = j.transpose().mul(j);
    let largest_diagonal = (0..n).map(|i| normal[(i, i)]).fold(0.0f64, f64::max);
    let damping = mu * if largest_diagonal > 0.0 && largest_diagonal.is_finite() { largest_diagonal } else { 1.0 };
    for i in 0..n {
        normal[(i, i)] += damping;
    }
    let rhs: Vec<f64> = j.mul_transpose_vec(rs).iter().map(|v| -v).collect();
    Ok(Lu::new(&normal)?.solve(&rhs))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A closure-based system for tests.
    struct Closure<F: Fn(&[f64], &mut [f64])> {
        n: usize,
        f: F,
    }

    impl<F: Fn(&[f64], &mut [f64])> NonlinearSystem for Closure<F> {
        fn dim(&self) -> usize {
            self.n
        }
        fn residual(&self, x: &[f64], r: &mut [f64]) {
            (self.f)(x, r)
        }
        fn jacobian(&self, x: &[f64]) -> Matrix {
            let mut j = Matrix::zeros(self.n, self.n);
            let (mut rp, mut rm) = (vec![0.0; self.n], vec![0.0; self.n]);
            for c in 0..self.n {
                let h = 1e-7 * x[c].abs().max(1.0);
                let (mut xp, mut xm) = (x.to_vec(), x.to_vec());
                xp[c] += h;
                xm[c] -= h;
                (self.f)(&xp, &mut rp);
                (self.f)(&xm, &mut rm);
                for r in 0..self.n {
                    j[(r, c)] = (rp[r] - rm[r]) / (2.0 * h);
                }
            }
            j
        }
    }

    #[test]
    fn solves_a_simple_system() {
        // x² + y² = 4, x = y  →  x = y = √2
        let sys = Closure { n: 2, f: |x: &[f64], r: &mut [f64]| {
            r[0] = x[0] * x[0] + x[1] * x[1] - 4.0;
            r[1] = x[0] - x[1];
        } };
        let mut x = vec![1.0, 0.5];
        let report = NewtonSolver::new(NewtonOptions::default()).solve(&sys, &mut x).unwrap();
        assert!((x[0] - 2f64.sqrt()).abs() < 1e-9 && (x[1] - 2f64.sqrt()).abs() < 1e-9);
        assert!(report.iterations <= 8);
    }

    /// arctan(x) = 0 from x0 = 3: a full Newton step overshoots and diverges,
    /// so this needs the line search (the Delphi suite's TArctanProblem).
    #[test]
    fn line_search_rescues_arctan() {
        let sys = Closure { n: 1, f: |x: &[f64], r: &mut [f64]| r[0] = x[0].atan() };
        let mut x = vec![3.0];
        let options = NewtonOptions { max_iterations: 50, ..Default::default() };
        NewtonSolver::new(options.clone()).solve(&sys, &mut x).unwrap();
        assert!(x[0].abs() < 1e-9);

        let mut x = vec![3.0];
        let without = NewtonOptions { use_line_search: false, use_levenberg_fallback: false, ..options };
        assert!(NewtonSolver::new(without).solve(&sys, &mut x).is_err(), "plain Newton diverges from 3");
    }

    /// Residual scaling: rates of order 1e-9 are not "converged" at the start.
    #[test]
    fn scaled_convergence_on_tiny_rates() {
        let sys = Closure { n: 1, f: |x: &[f64], r: &mut [f64]| r[0] = 1e-9 * (2.0 - x[0]) };
        let mut x = vec![1.0];
        let mut solver = NewtonSolver::new(NewtonOptions::default());
        solver.typical_f_from_residual(&sys, &x, 1e-30);
        solver.solve(&sys, &mut x).unwrap();
        assert!((x[0] - 2.0).abs() < 1e-9, "x = {}", x[0]);
    }

    /// A Jacobian whose reciprocal condition number (~1e-14) is below the 1e-12
    /// threshold: Levenberg–Marquardt takes over and still gets the residual down.
    #[test]
    fn levenberg_marquardt_handles_ill_conditioning() {
        let sys = Closure { n: 2, f: |x: &[f64], r: &mut [f64]| {
            r[0] = x[0] + x[1] - 2.0;
            r[1] = x[0] + x[1] - 2.0 + 1e-13 * (x[1] - 1.0);
        } };
        let mut x = vec![5.0, -1.0];
        let report = NewtonSolver::new(NewtonOptions { max_iterations: 100, ..Default::default() })
            .solve(&sys, &mut x)
            .unwrap();
        assert!(report.used_levenberg, "the ill-conditioned Jacobian should send it to LM");
        assert!((x[0] + x[1] - 2.0).abs() < 1e-9, "x = {x:?}");
    }
}
