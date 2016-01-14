#![feature(test)]
extern crate test;

use std::ops::{Mul};

extern crate nalgebra;
use self::nalgebra::{DMat, Transpose};

extern crate rblas;
use rblas::matrix::ops::Gemm;
use rblas::math::mat::Mat;
use rblas::attribute::Transpose as BlasTranspose;

#[bench]
fn bench_samples_mul_hidden_t_nalgebra(bencher: &mut test::Bencher) {
    let nsamples = 6 * 1000;
    let nobserved = 10 * 10;
    let nhidden = 20;

    bencher.iter(|| {
        let samples = DMat::<f64>::new_ones(nsamples, nobserved);
        let hidden = DMat::<f64>::new_ones(nhidden, nobserved);
        samples.mul(hidden.transpose())
    });
}

#[bench]
fn bench_samples_mul_hidden_t_blas(bencher: &mut test::Bencher) {
    let nsamples = 6 * 1000;
    let nobserved = 10 * 10;
    let nhidden = 20;

    bencher.iter(|| {
        let samples = Mat::<f64>::fill(1., nsamples, nobserved);
        let hidden = Mat::<f64>::fill(1., nhidden, nobserved);
        let mut result = Mat::<f64>::fill(1., nsamples, nhidden);
        Gemm::gemm(
            &1.,
            BlasTranspose::NoTrans, &samples,
            BlasTranspose::Trans, &hidden,
            &0.,
            &mut result);
        result
    });
}
