//! Small dense linear algebra for the model layer: a matrix type, LU with
//! partial pivoting and a condition estimate, and eigenvalues (through faer).
//!
//! The matrices here are Jacobians of biochemical models — tens of rows, a few
//! hundred at most — so plain dense code is the right tool.

/// A dense, row-major matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

impl Matrix {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self { rows, cols, data: vec![0.0; rows * cols] }
    }

    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m[(i, i)] = 1.0;
        }
        m
    }

    pub fn from_rows(rows: &[Vec<f64>]) -> Self {
        let r = rows.len();
        let c = rows.first().map_or(0, Vec::len);
        let mut m = Self::zeros(r, c);
        for (i, row) in rows.iter().enumerate() {
            assert_eq!(row.len(), c, "all rows must have the same length");
            m.data[i * c..(i + 1) * c].copy_from_slice(row);
        }
        m
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn row(&self, i: usize) -> &[f64] {
        &self.data[i * self.cols..(i + 1) * self.cols]
    }

    pub fn mul(&self, other: &Matrix) -> Matrix {
        assert_eq!(self.cols, other.rows, "matrix dimensions do not match");
        let mut out = Matrix::zeros(self.rows, other.cols);
        for i in 0..self.rows {
            for k in 0..self.cols {
                let a = self[(i, k)];
                if a != 0.0 {
                    for j in 0..other.cols {
                        out[(i, j)] += a * other[(k, j)];
                    }
                }
            }
        }
        out
    }

    pub fn mul_vec(&self, x: &[f64]) -> Vec<f64> {
        assert_eq!(self.cols, x.len());
        (0..self.rows)
            .map(|i| self.row(i).iter().zip(x).map(|(a, b)| a * b).sum())
            .collect()
    }

    /// `selfᵀ x`
    pub fn mul_transpose_vec(&self, x: &[f64]) -> Vec<f64> {
        assert_eq!(self.rows, x.len());
        let mut out = vec![0.0; self.cols];
        for (i, xi) in x.iter().enumerate() {
            for (o, a) in out.iter_mut().zip(self.row(i)) {
                *o += a * xi;
            }
        }
        out
    }

    pub fn transpose(&self) -> Matrix {
        let mut t = Matrix::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                t[(j, i)] = self[(i, j)];
            }
        }
        t
    }

    /// The largest absolute column sum.
    pub fn norm_1(&self) -> f64 {
        (0..self.cols)
            .map(|j| (0..self.rows).map(|i| self[(i, j)].abs()).sum::<f64>())
            .fold(0.0, f64::max)
    }

    /// Eigenvalues as (real, imaginary) pairs, largest real part first.
    pub fn eigenvalues(&self) -> Result<Vec<(f64, f64)>, String> {
        assert_eq!(self.rows, self.cols, "eigenvalues need a square matrix");
        if self.rows == 0 {
            return Ok(Vec::new());
        }
        // faer would otherwise use a thread pool: no use for matrices this
        // small (the Delphi version found the same with OpenBLAS), and threads
        // are not available in the browser at all.
        faer::set_global_parallelism(faer::Par::Seq);
        let m = faer::Mat::<f64>::from_fn(self.rows, self.cols, |i, j| self[(i, j)]);
        let values = m.eigenvalues().map_err(|e| format!("eigenvalue computation failed: {e:?}"))?;
        let mut pairs: Vec<(f64, f64)> = values.iter().map(|z| (z.re, z.im)).collect();
        pairs.sort_by(|a, b| b.0.total_cmp(&a.0).then(b.1.total_cmp(&a.1)));
        Ok(pairs)
    }
}

impl std::ops::Index<(usize, usize)> for Matrix {
    type Output = f64;
    fn index(&self, (i, j): (usize, usize)) -> &f64 {
        debug_assert!(i < self.rows && j < self.cols);
        &self.data[i * self.cols + j]
    }
}

impl std::ops::IndexMut<(usize, usize)> for Matrix {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut f64 {
        debug_assert!(i < self.rows && j < self.cols);
        &mut self.data[i * self.cols + j]
    }
}

/// An LU factorisation with partial pivoting, `P A = L U`.
pub struct Lu {
    lu: Matrix,
    /// `perm[i]` is the row of A that ended up in row i.
    perm: Vec<usize>,
    norm_1: f64,
}

impl Lu {
    /// Factor a square matrix. Fails if a pivot is exactly zero.
    pub fn new(a: &Matrix) -> Result<Lu, String> {
        assert_eq!(a.rows, a.cols, "LU needs a square matrix");
        let n = a.rows;
        let mut lu = a.clone();
        let mut perm: Vec<usize> = (0..n).collect();
        for k in 0..n {
            let pivot_row = (k..n)
                .max_by(|&i, &j| lu[(i, k)].abs().total_cmp(&lu[(j, k)].abs()))
                .unwrap();
            if lu[(pivot_row, k)] == 0.0 || !lu[(pivot_row, k)].is_finite() {
                return Err(format!("the matrix is singular (column {k} has no usable pivot)"));
            }
            if pivot_row != k {
                for j in 0..n {
                    lu.data.swap(k * n + j, pivot_row * n + j);
                }
                perm.swap(k, pivot_row);
            }
            let pivot = lu[(k, k)];
            for i in k + 1..n {
                let factor = lu[(i, k)] / pivot;
                lu[(i, k)] = factor;
                if factor != 0.0 {
                    for j in k + 1..n {
                        lu[(i, j)] -= factor * lu[(k, j)];
                    }
                }
            }
        }
        Ok(Lu { lu, perm, norm_1: a.norm_1() })
    }

    /// Solve `A x = b`.
    pub fn solve(&self, b: &[f64]) -> Vec<f64> {
        let n = self.lu.rows;
        let mut x: Vec<f64> = self.perm.iter().map(|&p| b[p]).collect();
        for i in 0..n {
            for j in 0..i {
                x[i] -= self.lu[(i, j)] * x[j];
            }
        }
        for i in (0..n).rev() {
            for j in i + 1..n {
                x[i] -= self.lu[(i, j)] * x[j];
            }
            x[i] /= self.lu[(i, i)];
        }
        x
    }

    /// Solve `Aᵀ x = b`.
    pub fn solve_transpose(&self, b: &[f64]) -> Vec<f64> {
        let n = self.lu.rows;
        // Aᵀ = Uᵀ Lᵀ P, so solve Uᵀ z = b, then Lᵀ w = z, then x = Pᵀ w.
        let mut z = b.to_vec();
        for i in 0..n {
            for j in 0..i {
                z[i] -= self.lu[(j, i)] * z[j];
            }
            z[i] /= self.lu[(i, i)];
        }
        for i in (0..n).rev() {
            for j in i + 1..n {
                z[i] -= self.lu[(j, i)] * z[j];
            }
        }
        let mut x = vec![0.0; n];
        for (i, &p) in self.perm.iter().enumerate() {
            x[p] = z[i];
        }
        x
    }

    /// An estimate of the reciprocal condition number in the 1-norm,
    /// `1 / (‖A‖₁ ‖A⁻¹‖₁)`, as LAPACK's `dgecon` gives. ‖A⁻¹‖₁ is estimated by
    /// Hager's method (Higham, *Accuracy and Stability of Numerical
    /// Algorithms*, §15.3), which needs only a few solves.
    pub fn reciprocal_condition(&self) -> f64 {
        let n = self.lu.rows;
        if n == 0 || self.norm_1 == 0.0 {
            return 0.0;
        }
        let mut x = vec![1.0 / n as f64; n];
        let mut estimate = 0.0;
        for _ in 0..5 {
            let y = self.solve(&x);
            let y_norm: f64 = y.iter().map(|v| v.abs()).sum();
            if y_norm <= estimate {
                break;
            }
            estimate = y_norm;
            let sign: Vec<f64> = y.iter().map(|v| if *v >= 0.0 { 1.0 } else { -1.0 }).collect();
            let z = self.solve_transpose(&sign);
            let (j, z_max) = z
                .iter()
                .enumerate()
                .map(|(j, v)| (j, v.abs()))
                .fold((0, 0.0), |best, c| if c.1 > best.1 { c } else { best });
            let zx: f64 = z.iter().zip(&x).map(|(a, b)| a * b).sum();
            if z_max <= zx {
                break;
            }
            x = vec![0.0; n];
            x[j] = 1.0;
        }
        if !estimate.is_finite() || estimate == 0.0 {
            return 0.0;
        }
        1.0 / (self.norm_1 * estimate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lu_solves_both_ways() {
        let a = Matrix::from_rows(&[vec![2.0, 1.0, 1.0], vec![4.0, -6.0, 0.0], vec![-2.0, 7.0, 2.0]]);
        let lu = Lu::new(&a).unwrap();
        let x = vec![1.0, -2.0, 3.0];
        let b = a.mul_vec(&x);
        for (got, want) in lu.solve(&b).iter().zip(&x) {
            assert!((got - want).abs() < 1e-12);
        }
        let bt = a.mul_transpose_vec(&x);
        for (got, want) in lu.solve_transpose(&bt).iter().zip(&x) {
            assert!((got - want).abs() < 1e-12);
        }
    }

    #[test]
    fn singular_matrix_is_rejected() {
        let a = Matrix::from_rows(&[vec![1.0, 2.0], vec![2.0, 4.0]]);
        assert!(Lu::new(&a).is_err());
    }

    /// For a diagonal matrix the condition number is exactly max|d| / min|d|,
    /// and Hager's estimate is exact there.
    #[test]
    fn condition_estimate() {
        let a = Matrix::from_rows(&[vec![1e3, 0.0, 0.0], vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1e-4]]);
        let rcond = Lu::new(&a).unwrap().reciprocal_condition();
        assert!((rcond - 1e-7).abs() < 1e-12, "rcond = {rcond}");
        assert!((Lu::new(&Matrix::identity(4)).unwrap().reciprocal_condition() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn eigenvalues_of_known_matrices() {
        // A rotation-plus-decay block has eigenvalues -1 ± 2i; the other is 3.
        let a = Matrix::from_rows(&[vec![-1.0, -2.0, 0.0], vec![2.0, -1.0, 0.0], vec![0.0, 0.0, 3.0]]);
        let ev = a.eigenvalues().unwrap();
        assert_eq!(ev.len(), 3);
        assert!((ev[0].0 - 3.0).abs() < 1e-12 && ev[0].1.abs() < 1e-12);
        assert!((ev[1].0 + 1.0).abs() < 1e-12 && (ev[1].1.abs() - 2.0).abs() < 1e-12);
        assert!((ev[1].1 + ev[2].1).abs() < 1e-12, "a conjugate pair");
    }
}
