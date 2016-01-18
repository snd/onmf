#![feature(test)]
extern crate test;

use std::ops::{Mul, Add};

extern crate nalgebra;
use self::nalgebra::{DMat, Transpose};

extern crate rblas;
use rblas::{Gemm};
use rblas::attribute::Transpose as BlasTranspose;

extern crate ndarray;
use ndarray::ArrayBase;
use ndarray::blas::AsBlas;

extern crate rand;
use rand::{StdRng, SeedableRng};

extern crate onmf;

const NSAMPLES: usize = 6 * 1000;
const NOBSERVED: usize = 10 * 10;
const NHIDDEN: usize = 20;

// TODO use the functions in OrthogonalNMFBlas here directly

type FloatT = f32;

#[bench]
fn bench_ortho_nmf_weights_multiplier_nalgebra(bencher: &mut test::Bencher) {
    let samples = DMat::<FloatT>::new_ones(NSAMPLES, NOBSERVED);
    let hidden = DMat::<FloatT>::new_ones(NHIDDEN, NOBSERVED);
    bencher.iter(|| {
        samples.clone().mul(hidden.transpose())
    });
}

#[bench]
fn bench_ortho_nmf_weights_multiplier_rblas(bencher: &mut test::Bencher) {
    let mut samples = ArrayBase::<Vec<FloatT>, (usize, usize)>::from_elem((NSAMPLES, NOBSERVED), 1.);
    let mut hidden = ArrayBase::<Vec<FloatT>, (usize, usize)>::from_elem((NHIDDEN, NOBSERVED), 1.);
    bencher.iter(|| {
        let mut result = ArrayBase::<Vec<FloatT>, (usize, usize)>::from_elem((NSAMPLES, NHIDDEN), 1.);
        Gemm::gemm(
            &1.,
            BlasTranspose::NoTrans, &samples.blas(),
            BlasTranspose::Trans, &hidden.blas(),
            &0.,
            &mut result.blas());
        result
    });
}

#[bench]
fn bench_ortho_nmf_weights_multiplier_rblas2(bencher: &mut test::Bencher) {
    let seed: &[_] = &[1, 2, 3, 4];
    let mut rng: StdRng = SeedableRng::from_seed(seed);
    let mut samples = ArrayBase::<Vec<FloatT>, (usize, usize)>::from_elem((NSAMPLES, NOBSERVED), 1.);
    let mut ortho_nmf = onmf::OrthogonalNMFBlas::new_random01(
        NHIDDEN, NOBSERVED, NSAMPLES, &mut rng);
    bencher.iter(|| {
        ortho_nmf.update_weights_multiplier(&mut samples);
    });
}

#[bench]
fn bench_ortho_nmf_weights_divisor_nalgebra(bencher: &mut test::Bencher) {
    let weights = DMat::<FloatT>::new_ones(NSAMPLES, NHIDDEN);
    let hidden = DMat::<FloatT>::new_ones(NHIDDEN, NOBSERVED);
    bencher.iter(|| {
        weights.clone().mul(&hidden).mul(hidden.transpose())
    });
}

#[bench]
fn bench_ortho_nmf_weights_divisor_rblas(bencher: &mut test::Bencher) {
    let mut weights = ArrayBase::<Vec<FloatT>, (usize, usize)>::from_elem((NSAMPLES, NHIDDEN), 1.);
    let mut hidden = ArrayBase::<Vec<FloatT>, (usize, usize)>::from_elem((NHIDDEN, NOBSERVED), 1.);
    let mut reconstruction = ArrayBase::<Vec<FloatT>, (usize, usize)>::from_elem((NSAMPLES, NOBSERVED), 1.);
    bencher.iter(|| {
        let mut result = ArrayBase::<Vec<FloatT>, (usize, usize)>::from_elem((NSAMPLES, NHIDDEN), 1.);
        Gemm::gemm(
            &1.,
            BlasTranspose::NoTrans, &weights.blas(),
            BlasTranspose::NoTrans, &hidden.blas(),
            &0.,
            &mut reconstruction.blas());
        Gemm::gemm(
            &1.,
            BlasTranspose::NoTrans, &reconstruction.blas(),
            BlasTranspose::Trans, &hidden.blas(),
            &0.,
            &mut result.blas());
        result
    });
}

#[bench]
fn bench_ortho_nmf_hidden_multiplier_nalgebra(bencher: &mut test::Bencher) {
    let weights = DMat::<FloatT>::new_ones(NSAMPLES, NHIDDEN);
    let samples = DMat::<FloatT>::new_ones(NSAMPLES, NOBSERVED);
    bencher.iter(|| {
        weights.transpose().mul(&samples);
    });
}

#[bench]
fn bench_ortho_nmf_hidden_multiplier_rblas(bencher: &mut test::Bencher) {
    let mut weights = ArrayBase::<Vec<FloatT>, (usize, usize)>::from_elem((NSAMPLES, NHIDDEN), 1.);
    let mut samples = ArrayBase::<Vec<FloatT>, (usize, usize)>::from_elem((NSAMPLES, NOBSERVED), 1.);
    bencher.iter(|| {
        let mut result = ArrayBase::<Vec<FloatT>, (usize, usize)>::from_elem((NHIDDEN, NOBSERVED), 1.);
        Gemm::gemm(
            &1.,
            BlasTranspose::Trans, &weights.blas(),
            BlasTranspose::NoTrans, &samples.blas(),
            &0.,
            &mut result.blas());
        result
    });
}

// TODO why is this so much faster than the other ones
// when it does so much more ?
// is the clone so expensive ?
#[bench]
fn bench_ortho_nmf_hidden_divisor_nalgebra(bencher: &mut test::Bencher) {
    let weights = DMat::<FloatT>::new_ones(NSAMPLES, NHIDDEN);
    let hidden = DMat::<FloatT>::new_ones(NHIDDEN, NOBSERVED);
    bencher.iter(|| {
        // TODO this is surprisingly weirdly fast
        let gamma = DMat::<FloatT>::new_ones(NHIDDEN, NHIDDEN);
        weights.transpose().mul(&weights).mul(&hidden)
            .add(gamma.mul(&hidden))
    });
}

#[bench]
fn bench_ortho_nmf_hidden_divisor_rblas(bencher: &mut test::Bencher) {
    let mut weights = ArrayBase::<Vec<FloatT>, (usize, usize)>::from_elem((NSAMPLES, NHIDDEN), 1.);
    let mut weights_copy = ArrayBase::<Vec<FloatT>, (usize, usize)>::from_elem((NSAMPLES, NHIDDEN), 1.);
    let mut hidden = ArrayBase::<Vec<FloatT>, (usize, usize)>::from_elem((NHIDDEN, NOBSERVED), 1.);
    let mut gamma = ArrayBase::<Vec<FloatT>, (usize, usize)>::from_elem((NHIDDEN, NHIDDEN), 1.);
    // this is really small
    let mut tmp = ArrayBase::<Vec<FloatT>, (usize, usize)>::from_elem((NHIDDEN, NHIDDEN), 1.);
    bencher.iter(|| {
        let mut result = ArrayBase::<Vec<FloatT>, (usize, usize)>::from_elem((NHIDDEN, NOBSERVED), 1.);
        // we must make a copy here because we need two mutable
        // references to weights at the same time for .blas()
        // weights_copy.clone_from(&weights);
        let mut weights_copy = weights.clone();
        Gemm::gemm(
            &1.,
            BlasTranspose::Trans, &weights.blas(),
            BlasTranspose::NoTrans, &weights_copy.blas(),
            &0.,
            &mut tmp.blas());
        Gemm::gemm(
            &1.,
            BlasTranspose::NoTrans, &gamma.blas(),
            BlasTranspose::NoTrans, &hidden.blas(),
            &0.,
            &mut result.blas());
        // add to previous result
        Gemm::gemm(
            &1.,
            BlasTranspose::NoTrans, &tmp.blas(),
            BlasTranspose::NoTrans, &hidden.blas(),
            &1.,
            &mut result.blas());
        result
    });
}
