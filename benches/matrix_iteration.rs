#![feature(test)]
extern crate test;

extern crate nalgebra;
use self::nalgebra::{DMat};

const NSAMPLES: usize = 6 * 1000;
const NHIDDEN: usize = 20;

#[bench]
fn bench_dmat_col_then_row_iteration(bencher: &mut test::Bencher) {
    let dividend = DMat::<f64>::new_ones(NSAMPLES, NHIDDEN);
    let divisor = DMat::<f64>::new_ones(NSAMPLES, NHIDDEN);
    let mut result = DMat::<f64>::new_ones(NSAMPLES, NHIDDEN);
    bencher.iter(|| {
        // USE THIS. IT'S FASTER.
        // DMat is stored in column major order.
        // it stores one consecutive column after another.
        // in column major order the first (row) index varies
        // faster as one steps through consecutive memory locations.
        // as a result this loops through contiguous memory locations
        // in the inner loop which is faster due to caching.
        for col in 0..result.ncols() {
            for row in 0..result.nrows() {
                let index = (row, col);
                result[index] = result[index] * dividend[index] / divisor[index];
            }
        }
    });
}

#[bench]
fn bench_dmat_row_then_col_iteration(bencher: &mut test::Bencher) {
    let dividend = DMat::<f64>::new_ones(NSAMPLES, NHIDDEN);
    let divisor = DMat::<f64>::new_ones(NSAMPLES, NHIDDEN);
    let mut result = DMat::<f64>::new_ones(NSAMPLES, NHIDDEN);
    bencher.iter(|| {
        // DONT USE THIS. IT'S SLOWER.
        for row in 0..result.nrows() {
            for col in 0..result.ncols() {
                let index = (row, col);
                result[index] = result[index] * dividend[index] / divisor[index];
            }
        }
    });
}
