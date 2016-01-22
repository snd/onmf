use std::ops::{Mul};

#[macro_use]
extern crate nalgebra;
use nalgebra::{ApproxEq, DMat, Transpose};

extern crate ndarray;
use ndarray::ArrayBase;
use ndarray::blas::AsBlas;

extern crate rand;
use rand::{StdRng, SeedableRng};

extern crate rblas;
use rblas::{Syrk, Matrix};
use rblas::attribute::{Symmetry};
use rblas::attribute::Transpose as BlasTranspose;

extern crate onmf;
use onmf::helpers::{Array2D, random01};

#[test]
fn test_a_mul_a_transposed_is_symetric() {
    let seed: &[_] = &[1, 2, 3, 4];
    let mut rng: StdRng = SeedableRng::from_seed(seed);

    let rows = 10;
    let cols = 5;

    let mut a: DMat<f32> = DMat::new_zeros(rows, cols);

    for icol in 0..a.ncols() {
        for irow in 0..a.nrows() {
            a[(irow, icol)] = random01(&mut rng);
        }
    }

    println!("{:?}", a);

    // b <- a * a^T
    let b = a.clone().mul(a.transpose());
    println!("{:?}", b);

    // is b symetric
    assert_approx_eq_ulps!(b, b.transpose(), 10);

    // c <- a^T * a
    let c = a.transpose().mul(a);
    println!("{:?}", c);

    // is c symetric
    assert_approx_eq_ulps!(c, c.transpose(), 10);
}

// #[test]
// fn test_syrk() {
//     let n: usize = 10;
//     let k: usize = 4;
//
//     let mut a = ArrayBase::<Vec<f32>, (usize, usize)>::from_elem(
//         (k, n), 2.);
//     let mut c = ArrayBase::<Vec<f32>, (usize, usize)>::from_elem(
//         (k, k), 1.);
//
//     assert!(a.is_standard_layout());
//     assert!(c.is_standard_layout());
//
//     {
//         let ablas = a.blas();
//         let mut cblas = c.blas();
//
//         // a = k x n
//         // c = n x n
//
//         println!("n = {}", ablas.rows());
//         println!("k = {}", ablas.cols());
//         println!("lda = {}", ablas.lead_dim());
//         println!("ldc = {}", cblas.lead_dim());
//
//         // C <- A^T * A
//         Syrk::syrk(
//             Symmetry::Upper,
//             Transpose::NoTrans,
//             // Transpose::Trans,
//             &1.,
//             &ablas,
//             &0.,
//             &mut cblas);
//     }
//
//     println!("c = {:?}", c);
// }
