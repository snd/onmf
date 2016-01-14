#![feature(test)]
extern crate test;

use std::ops::{Mul};

extern crate nalgebra;
use self::nalgebra::{DMat, Transpose};

extern crate rblas;
use rblas::matrix::ops::{Gemm, Syrk};
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
    let mut tmp = Mat::<f64>::new(NSAMPLES, NOBSERVED);
    bencher.iter(|| {
        let mut result = Mat::<f64>::fill(1., NSAMPLES, NHIDDEN);
        Gemm::gemm(
            &1.,
            BlasTranspose::NoTrans, &weights,
            BlasTranspose::NoTrans, &hidden,
            &0.,
            &mut tmp);
        Gemm::gemm(
            &1.,
            BlasTranspose::NoTrans, &tmp,
            BlasTranspose::Trans, &hidden,
            &0.,
            &mut result);
        result
    });
}
