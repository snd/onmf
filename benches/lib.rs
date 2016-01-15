#![feature(test)]
extern crate test;

use std::ops::{Mul, Add};

extern crate nalgebra;
use self::nalgebra::{DMat, Transpose};

extern crate rblas;
use rblas::matrix::ops::{Gemm};
use rblas::math::mat::Mat;
use rblas::attribute::Transpose as BlasTranspose;

const NSAMPLES: usize = 6 * 1000;
const NOBSERVED: usize = 10 * 10;
const NHIDDEN: usize = 20;

#[bench]
fn bench_ortho_nmf_weights_dividend_nalgebra(bencher: &mut test::Bencher) {
    let samples = DMat::<f64>::new_ones(NSAMPLES, NOBSERVED);
    let hidden = DMat::<f64>::new_ones(NHIDDEN, NOBSERVED);
    bencher.iter(|| {
        samples.clone().mul(hidden.transpose())
    });
}

#[bench]
fn bench_ortho_nmf_weights_dividend_rblas(bencher: &mut test::Bencher) {
    let samples = Mat::<f64>::fill(1., NSAMPLES, NOBSERVED);
    let hidden = Mat::<f64>::fill(1., NHIDDEN, NOBSERVED);
    bencher.iter(|| {
        let mut result = Mat::<f64>::fill(1., NSAMPLES, NHIDDEN);
        Gemm::gemm(
            &1.,
            BlasTranspose::NoTrans, &samples,
            BlasTranspose::Trans, &hidden,
            &0.,
            &mut result);
        result
    });
}

#[bench]
fn bench_ortho_nmf_weights_divisor_nalgebra(bencher: &mut test::Bencher) {
    let weights = DMat::<f64>::new_ones(NSAMPLES, NHIDDEN);
    let hidden = DMat::<f64>::new_ones(NHIDDEN, NOBSERVED);
    bencher.iter(|| {
        weights.clone().mul(&hidden).mul(hidden.transpose())
    });
}

#[bench]
fn bench_ortho_nmf_weights_divisor_rblas(bencher: &mut test::Bencher) {
    let weights = Mat::<f64>::fill(1., NSAMPLES, NHIDDEN);
    let hidden = Mat::<f64>::fill(1., NHIDDEN, NOBSERVED);
    let mut reconstruction = Mat::<f64>::new(NSAMPLES, NOBSERVED);
    bencher.iter(|| {
        let mut result = Mat::<f64>::fill(1., NSAMPLES, NHIDDEN);
        Gemm::gemm(
            &1.,
            BlasTranspose::NoTrans, &weights,
            BlasTranspose::NoTrans, &hidden,
            &0.,
            &mut reconstruction);
        Gemm::gemm(
            &1.,
            BlasTranspose::NoTrans, &reconstruction,
            BlasTranspose::Trans, &hidden,
            &0.,
            &mut result);
        result
    });
}

// TODO why is this so much faster than the other ones
// when it does so much more ?
// is the clone so expensive ?
#[bench]
fn bench_ortho_nmf_hidden_divisor_nalgebra(bencher: &mut test::Bencher) {
    let weights = DMat::<f64>::new_ones(NSAMPLES, NHIDDEN);
    let hidden = DMat::<f64>::new_ones(NHIDDEN, NOBSERVED);
    bencher.iter(|| {
        let gamma = DMat::<f64>::new_ones(NHIDDEN, NHIDDEN);
        weights.transpose().mul(&weights).mul(&hidden)
            .add(gamma.mul(&hidden))
    });
}
