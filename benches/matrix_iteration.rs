#![feature(test)]
extern crate test;

extern crate nalgebra;
use self::nalgebra::{DMat};

extern crate ndarray;
use ndarray::ArrayBase;

const NSAMPLES: usize = 6 * 1000;
const NHIDDEN: usize = 20;

// column major:
// stores one consecutive column after another.
// row index varies most rapidly as one steps through consecutive memory.
// iterate column in outer loop, row index in inner loop.
//
// row major:
// stores one consecutive row after another.
// column index varies most rapidly as one steps through consecutive memory.
// iterate row in outer loop, column index in inner loop.

// looping through contiguous memory locations
// in the inner loop is much faster due to CPU caching.

// DMat is stored column major.
// ArrayBase is stored row major.

// slow
#[bench]
fn bench_ndarray_col_then_row_iteration(bencher: &mut test::Bencher) {
    let dividend = ArrayBase::<Vec<f64>, (usize, usize)>::from_elem((NSAMPLES, NHIDDEN), 1.);
    let divisor = ArrayBase::<Vec<f64>, (usize, usize)>::from_elem((NSAMPLES, NHIDDEN), 1.);
    let mut result = ArrayBase::<Vec<f64>, (usize, usize)>::from_elem((NSAMPLES, NHIDDEN), 1.);
    bencher.iter(|| {
        for col in 0..result.shape()[1] {
            for row in 0..result.shape()[0] {
                let index = [row, col];
                result[index] = result[index] * dividend[index] / divisor[index];
            }
        }
    });
}

// 3rd fastest
#[bench]
fn bench_ndarray_row_then_col_iteration(bencher: &mut test::Bencher) {
    let dividend = ArrayBase::<Vec<f64>, (usize, usize)>::from_elem((NSAMPLES, NHIDDEN), 1.);
    let divisor = ArrayBase::<Vec<f64>, (usize, usize)>::from_elem((NSAMPLES, NHIDDEN), 1.);
    let mut result = ArrayBase::<Vec<f64>, (usize, usize)>::from_elem((NSAMPLES, NHIDDEN), 1.);
    bencher.iter(|| {
        for row in 0..result.shape()[0] {
            for col in 0..result.shape()[1] {
                let index = [row, col];
                result[index] = result[index] * dividend[index] / divisor[index];
            }
        }
    });
}

// slow
#[bench]
fn bench_ndarray_col_then_row_unchecked_iteration(bencher: &mut test::Bencher) {
    let dividend = ArrayBase::<Vec<f64>, (usize, usize)>::from_elem((NSAMPLES, NHIDDEN), 1.);
    let divisor = ArrayBase::<Vec<f64>, (usize, usize)>::from_elem((NSAMPLES, NHIDDEN), 1.);
    let mut result = ArrayBase::<Vec<f64>, (usize, usize)>::from_elem((NSAMPLES, NHIDDEN), 1.);
    bencher.iter(|| {
        for col in 0..result.shape()[1] {
            for row in 0..result.shape()[0] {
                let index = (row, col);
                unsafe {
                    *result.uget_mut(index) = result.uget(index) *
                        dividend.uget(index) / divisor.uget(index);
                }
            }
        }
    });
}

// FASTEST
#[bench]
fn bench_ndarray_row_then_col_unchecked_iteration(bencher: &mut test::Bencher) {
    let dividend = ArrayBase::<Vec<f64>, (usize, usize)>::from_elem((NSAMPLES, NHIDDEN), 1.);
    let divisor = ArrayBase::<Vec<f64>, (usize, usize)>::from_elem((NSAMPLES, NHIDDEN), 1.);
    let mut result = ArrayBase::<Vec<f64>, (usize, usize)>::from_elem((NSAMPLES, NHIDDEN), 1.);
    assert!(dividend.is_standard_layout());
    assert!(divisor.is_standard_layout());
    assert!(result.is_standard_layout());
    bencher.iter(|| {
        for row in 0..result.shape()[0] {
            for col in 0..result.shape()[1] {
                let index = (row, col);
                unsafe {
                    *result.uget_mut(index) = result.uget(index) *
                        dividend.uget(index) / divisor.uget(index);
                }
            }
        }
    });
}

// 2nd fastest
#[bench]
fn bench_ndarray_2x_zip_mut_with_iteration(bencher: &mut test::Bencher) {
    let dividend = ArrayBase::<Vec<f64>, (usize, usize)>::from_elem((NSAMPLES, NHIDDEN), 1.);
    let divisor = ArrayBase::<Vec<f64>, (usize, usize)>::from_elem((NSAMPLES, NHIDDEN), 1.);
    let mut result = ArrayBase::<Vec<f64>, (usize, usize)>::from_elem((NSAMPLES, NHIDDEN), 1.);
    bencher.iter(|| {
        result.zip_mut_with(&dividend, |dst, src| {
            *dst = dst.clone() * src;
        });
        result.zip_mut_with(&divisor, |dst, src| {
            *dst = dst.clone() / src;
        });
    });
}

#[bench]
fn bench_ndarray_zipped_iteration(bencher: &mut test::Bencher) {
    let dividend = ArrayBase::<Vec<f64>, (usize, usize)>::from_elem((NSAMPLES, NHIDDEN), 1.);
    let divisor = ArrayBase::<Vec<f64>, (usize, usize)>::from_elem((NSAMPLES, NHIDDEN), 1.);
    let mut result = ArrayBase::<Vec<f64>, (usize, usize)>::from_elem((NSAMPLES, NHIDDEN), 1.);
    bencher.iter(|| {
        for ((rresult, rdividend), rdivisor) in result.iter_mut().zip(dividend.iter()).zip(divisor.iter()) {
            *rresult = rresult.clone() * rdividend / rdivisor;
        }
    });
}

// fast
#[bench]
fn bench_dmat_col_then_row_iteration(bencher: &mut test::Bencher) {
    let dividend = DMat::<f64>::new_ones(NSAMPLES, NHIDDEN);
    let divisor = DMat::<f64>::new_ones(NSAMPLES, NHIDDEN);
    let mut result = DMat::<f64>::new_ones(NSAMPLES, NHIDDEN);
    bencher.iter(|| {
        for col in 0..result.ncols() {
            for row in 0..result.nrows() {
                let index = (row, col);
                result[index] = result[index] * dividend[index] / divisor[index];
            }
        }
    });
}

// slow
#[bench]
fn bench_dmat_row_then_col_iteration(bencher: &mut test::Bencher) {
    let dividend = DMat::<f64>::new_ones(NSAMPLES, NHIDDEN);
    let divisor = DMat::<f64>::new_ones(NSAMPLES, NHIDDEN);
    let mut result = DMat::<f64>::new_ones(NSAMPLES, NHIDDEN);
    bencher.iter(|| {
        for row in 0..result.nrows() {
            for col in 0..result.ncols() {
                let index = (row, col);
                result[index] = result[index] * dividend[index] / divisor[index];
            }
        }
    });
}
