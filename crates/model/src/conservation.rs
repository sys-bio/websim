//! Conservation analysis: finding the conserved moieties of a reaction network
//! and the reduced system in independent species.
//!
//! For a network with conserved moieties the full Jacobian `N ε` is
//! structurally singular — each conservation law contributes an exact zero
//! eigenvalue — so Newton cannot converge and every equilibrium looks
//! non-hyperbolic. Steady states and continuation therefore work on the
//! reduced system (Bifurcata specification §3.2):
//!
//! ```text
//! du/dt = N_R v(L u + T),        J = N_R ε L
//! ```
//!
//! where `u` holds the independent species, `N_R` is the stoichiometry matrix
//! restricted to their rows, `L = [I; L₀]` is the link matrix with
//! `N = L N_R`, and the dependent species are `x_dep = L₀ u + T`, with the
//! conserved totals `T` fixed by the initial values.
//!
//! The independent species are the first linearly independent rows of `N`,
//! taken in species order, found by Gauss–Jordan elimination on `Nᵀ`.

use crate::linalg::Matrix;

#[derive(Clone, Debug, PartialEq)]
pub struct Conservation {
    /// Indices of the independent species, in species order.
    pub independent: Vec<usize>,
    /// Indices of the dependent species, in species order.
    pub dependent: Vec<usize>,
    /// `L₀` (dependent × independent): `N_dep = L₀ N_ind` and `x_dep = L₀ x_ind + T`.
    pub link: Matrix,
}

impl Conservation {
    /// Analyse a stoichiometry matrix `N` (species × reactions).
    pub fn analyse(n: &Matrix) -> Conservation {
        let species = n.rows();
        let reactions = n.cols();
        // Gauss–Jordan on Nᵀ, column by column in species order: a column with
        // a pivot is an independent species; the reduced echelon form then
        // gives every other column as a combination of the pivot columns.
        let mut a = n.transpose(); // reactions × species
        let largest = (0..reactions)
            .flat_map(|r| (0..species).map(move |s| (r, s)))
            .fold(0.0f64, |m, (r, s)| m.max(a[(r, s)].abs()));
        let tol = 1e-9 * largest.max(1.0);

        let mut pivots: Vec<usize> = Vec::new(); // pivots[k] = species whose column has row k's pivot
        let mut row = 0;
        for col in 0..species {
            if row == reactions {
                break;
            }
            let best = (row..reactions).max_by(|&i, &j| a[(i, col)].abs().total_cmp(&a[(j, col)].abs())).unwrap();
            if a[(best, col)].abs() <= tol {
                continue; // a combination of earlier species: dependent
            }
            for c in 0..species {
                let tmp = a[(row, c)];
                a[(row, c)] = a[(best, c)];
                a[(best, c)] = tmp;
            }
            let pivot = a[(row, col)];
            for c in 0..species {
                a[(row, c)] /= pivot;
            }
            for r in 0..reactions {
                if r != row {
                    let factor = a[(r, col)];
                    if factor != 0.0 {
                        for c in 0..species {
                            a[(r, c)] -= factor * a[(row, c)];
                        }
                    }
                }
            }
            pivots.push(col);
            row += 1;
        }

        let independent = pivots.clone();
        let dependent: Vec<usize> = (0..species).filter(|s| !independent.contains(s)).collect();
        let mut link = Matrix::zeros(dependent.len(), independent.len());
        for (d, &s) in dependent.iter().enumerate() {
            for k in 0..independent.len() {
                link[(d, k)] = tidy(a[(k, s)]);
            }
        }
        Conservation { independent, dependent, link }
    }

    /// The number of conservation laws (= dependent species).
    pub fn law_count(&self) -> usize {
        self.dependent.len()
    }

    /// The independent species of a full state.
    pub fn reduce(&self, x: &[f64]) -> Vec<f64> {
        self.independent.iter().map(|&i| x[i]).collect()
    }

    /// The conserved totals of a full state: `T = x_dep − L₀ x_ind`.
    pub fn totals(&self, x: &[f64]) -> Vec<f64> {
        let u = self.reduce(x);
        let lu = self.link.mul_vec(&u);
        self.dependent.iter().zip(lu).map(|(&d, l)| x[d] - l).collect()
    }

    /// The full state from the independent species and the totals.
    pub fn full_state(&self, u: &[f64], totals: &[f64]) -> Vec<f64> {
        let mut x = vec![0.0; self.independent.len() + self.dependent.len()];
        for (&i, ui) in self.independent.iter().zip(u) {
            x[i] = *ui;
        }
        for ((&d, l), t) in self.dependent.iter().zip(self.link.mul_vec(u)).zip(totals) {
            x[d] = l + t;
        }
        x
    }

    /// The link matrix `L` (species × independent), with `N = L N_R`.
    pub fn link_matrix(&self) -> Matrix {
        let m = self.independent.len() + self.dependent.len();
        let mut l = Matrix::zeros(m, self.independent.len());
        for (k, &i) in self.independent.iter().enumerate() {
            l[(i, k)] = 1.0;
        }
        for (d, &s) in self.dependent.iter().enumerate() {
            for k in 0..self.independent.len() {
                l[(s, k)] = self.link[(d, k)];
            }
        }
        l
    }

    /// Each conservation law as text, e.g. `"C + E = 1"`, using the totals of `x`.
    pub fn describe(&self, names: &[String], x: &[f64]) -> Vec<String> {
        let totals = self.totals(x);
        self.dependent
            .iter()
            .enumerate()
            .map(|(d, &s)| {
                // x_dep − Σ L₀ x_ind = T
                let mut terms = vec![(1.0, names[s].as_str())];
                for (k, &i) in self.independent.iter().enumerate() {
                    if self.link[(d, k)] != 0.0 {
                        terms.push((-self.link[(d, k)], names[i].as_str()));
                    }
                }
                let mut text = String::new();
                for (n, (c, name)) in terms.iter().enumerate() {
                    let sign = if *c < 0.0 { "-" } else { "+" };
                    let magnitude = c.abs();
                    let coefficient = if (magnitude - 1.0).abs() < 1e-12 { String::new() } else { format!("{} ", trim(magnitude)) };
                    if n == 0 {
                        text.push_str(&format!("{}{coefficient}{name}", if *c < 0.0 { "-" } else { "" }));
                    } else {
                        text.push_str(&format!(" {sign} {coefficient}{name}"));
                    }
                }
                format!("{text} = {}", trim(totals[d]))
            })
            .collect()
    }
}

/// Clean up rounding in elimination: snap values very close to an integer,
/// and drop values that are zero in all but rounding.
fn tidy(v: f64) -> f64 {
    let rounded = v.round();
    if (v - rounded).abs() < 1e-10 { rounded } else { v }
}

fn trim(v: f64) -> String {
    let s = format!("{v:.6}");
    let s = s.trim_end_matches('0').trim_end_matches('.');
    if s == "-0" { "0".to_owned() } else { s.to_owned() }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Edelstein's network: species X, E, C; reactions R0..R5.
    ///   R0: -> X   R1: X -> 2X   R2: 2X -> X   R3: X + E -> C   R4: C -> X + E   R5: C -> E
    fn edelstein() -> Matrix {
        Matrix::from_rows(&[
            vec![1.0, 1.0, -1.0, -1.0, 1.0, 0.0],  // X
            vec![0.0, 0.0, 0.0, -1.0, 1.0, 1.0],   // E
            vec![0.0, 0.0, 0.0, 1.0, -1.0, -1.0],  // C
        ])
    }

    #[test]
    fn finds_edelsteins_conserved_cycle() {
        let c = Conservation::analyse(&edelstein());
        assert_eq!(c.independent, [0, 1], "X and E are independent");
        assert_eq!(c.dependent, [2], "C depends on them");
        assert_eq!(c.link.row(0), [0.0, -1.0], "C = -E + T");

        let x = [0.2, 1.0, 0.0];
        assert_eq!(c.totals(&x), [1.0], "E + C = 1");
        let names = ["X".to_owned(), "E".to_owned(), "C".to_owned()];
        assert_eq!(c.describe(&names, &x), ["C + E = 1"]);
    }

    /// The defining property: N = L N_R.
    #[test]
    fn link_matrix_reconstructs_the_stoichiometry() {
        for n in [
            edelstein(),
            // S1 <-> S2 <-> S3 (one law), plus a species no reaction changes.
            Matrix::from_rows(&[vec![-1.0, 0.0], vec![1.0, -1.0], vec![0.0, 1.0], vec![0.0, 0.0]]),
            // Two independent moieties, with a 2:1 stoichiometry.
            Matrix::from_rows(&[vec![-2.0, 0.0], vec![1.0, 0.0], vec![0.0, -1.0], vec![0.0, 1.0]]),
        ] {
            let c = Conservation::analyse(&n);
            let n_r = Matrix::from_rows(&c.independent.iter().map(|&i| n.row(i).to_vec()).collect::<Vec<_>>());
            let rebuilt = c.link_matrix().mul(&n_r);
            assert_eq!(rebuilt, n, "N = L N_R");
            assert_eq!(c.independent.len() + c.law_count(), n.rows());
        }
    }

    #[test]
    fn state_round_trip_preserves_totals() {
        let n = Matrix::from_rows(&[vec![-2.0, 0.0], vec![1.0, 0.0], vec![0.0, -1.0], vec![0.0, 1.0]]);
        let c = Conservation::analyse(&n);
        assert_eq!(c.law_count(), 2);
        let x = [3.0, 0.5, 1.25, 4.0];
        let totals = c.totals(&x);
        assert_eq!(c.full_state(&c.reduce(&x), &totals), x);
        // Move the independent species: the dependents follow and the totals hold.
        let moved = c.full_state(&[1.0, 2.0], &totals);
        assert_eq!(c.totals(&moved), totals);
    }

    #[test]
    fn no_conservation_means_no_reduction() {
        // The Brusselator: X and Y are both independent.
        let n = Matrix::from_rows(&[vec![1.0, -1.0, 1.0, -1.0], vec![0.0, 1.0, -1.0, 0.0]]);
        let c = Conservation::analyse(&n);
        assert_eq!(c.independent, [0, 1]);
        assert_eq!(c.law_count(), 0);
    }
}
