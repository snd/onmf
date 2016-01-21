use std::ops::{Mul, Add};

use rand::{Rand, Rng, Closed01};
use num::{Float, Zero};

use rblas::{Gemm, Matrix};
use rblas::attribute::Transpose;

use ndarray::{ArrayBase, DataOwned, DataMut};
use ndarray::blas::{BlasArrayViewMut, AsBlas};

use helpers::{random01, Dims, Array2D};

pub trait ShapeAsTuple<T> {
    fn shape_as_tuple(&self) -> T;
}

pub type FloatT = f32;

impl<T> ShapeAsTuple<(usize, usize)> for Array2D<T> {
    #[inline]
    fn shape_as_tuple(&self) -> (usize, usize) {
        (self.shape()[0], self.shape()[1])
    }
}

/// gamma is a symetric matrix with diagonal elements equal to zero
/// and other elements equal to 1
pub fn gamma(size: usize) -> Array2D<FloatT> {
    let mut gamma = Array2D::<FloatT>::from_elem((size, size), 1.);
    for x in gamma.diag_mut().iter_mut() {
        *x = 0.
    }
    gamma
}

// TODO better names than divisor and multiplier

/// `weights_multiplier <- samples * hidden.transpose()`
#[inline]
pub fn weights_multiplier(
    samples: &mut Array2D<FloatT>,
    hidden: &mut Array2D<FloatT>,
    weights_multiplier: &mut Array2D<FloatT>,
) {
    assert_eq!(samples.shape()[1], hidden.shape()[1]);
    assert_eq!(samples.shape()[0], weights_multiplier.shape()[0]);
    assert_eq!(hidden.shape()[0], weights_multiplier.shape()[1]);
    Gemm::gemm(
        &1.,
        Transpose::NoTrans, &samples.blas(),
        Transpose::Trans, &hidden.blas(),
        &0.,
        &mut weights_multiplier.blas());
}

// TODO this is the slowest
/// `weights_divisor <- weights * hidden * hidden.transpose()`
#[inline]
pub fn weights_divisor(
    weights: &mut Array2D<FloatT>,
    hidden: &mut Array2D<FloatT>,
    // temporary space to hold `weights * hidden`
    tmp: &mut Array2D<FloatT>,
    weights_divisor: &mut Array2D<FloatT>,
) {
    assert_eq!(weights.shape()[1], hidden.shape()[0]);
    assert_eq!(weights.shape()[0], tmp.shape()[0]);
    assert_eq!(hidden.shape()[1], tmp.shape()[1]);

    // TODO weights * hidden which we compute first is really large
    // we could first compute partial = hidden * hidden.transpose()
    // partial.shape() = (nhidden, nhidden)
    // and then in the second step compute
    // weights * partial.
    // however since we can't alias hidden we need to copy
    // it for this.
    // the question is whether the copying is faster than
    // the larger matrix multiplication. i guess it is.
    // TODO multiplying a matrix by its transpose is symetric
    // can we exploit that ?

    // tmp <- weights * hidden
    Gemm::gemm(
        &1.,
        Transpose::NoTrans, &weights.blas(),
        Transpose::NoTrans, &hidden.blas(),
        &0.,
        &mut tmp.blas());
    // weights_divisor <- tmp * hidden.transpose()
    Gemm::gemm(
        &1.,
        Transpose::NoTrans, &tmp.blas(),
        Transpose::Trans, &hidden.blas(),
        &0.,
        &mut weights_divisor.blas());
}

/// `hidden_multiplier <- weights.transpose() * samples`
#[inline]
pub fn hidden_multiplier(
    weights: &mut Array2D<FloatT>,
    samples:&mut Array2D<FloatT>,
    hidden_multiplier:&mut Array2D<FloatT>,
) {
    assert_eq!(weights.shape()[0], samples.shape()[0]);
    assert_eq!(weights.shape()[1], hidden_multiplier.shape()[0]);
    assert_eq!(samples.shape()[1], hidden_multiplier.shape()[1]);

    Gemm::gemm(
        &1.,
        Transpose::Trans, &weights.blas(),
        Transpose::NoTrans, &samples.blas(),
        &0.,
        &mut hidden_multiplier.blas());
}

// surprisingly this is the fastest
/// `hidden_divisor <- weights.transpose() * weights * hidden
#[inline]
pub fn hidden_divisor(
    weights: &mut Array2D<FloatT>,
    hidden: &mut Array2D<FloatT>,
    tmp: &mut Array2D<FloatT>,
    hidden_divisor: &mut Array2D<FloatT>,
) {
    assert_eq!(weights.shape()[1], hidden.shape()[0]);
    assert_eq!(tmp.shape()[0], hidden.shape()[0]);
    assert_eq!(tmp.shape()[1], hidden.shape()[0]);
    assert_eq!(hidden_divisor.shape_as_tuple(), hidden.shape_as_tuple());

    // TODO benchmark whether using clone_from is faster
    let mut weights_copy = weights.clone();

    Gemm::gemm(
        &1.,
        Transpose::Trans, &weights.blas(),
        Transpose::NoTrans, &weights_copy.blas(),
        &0.,
        &mut tmp.blas());
    Gemm::gemm(
        &1.,
        Transpose::NoTrans, &tmp.blas(),
        Transpose::NoTrans, &hidden.blas(),
        &0.,
        &mut hidden_divisor.blas());
}

/// `input_output <- input_output + alpha * gamma * hidden`
/// #[inline]
pub fn add_orthogonalization(
    alpha: FloatT,
    gamma: &mut Array2D<FloatT>,
    hidden: &mut Array2D<FloatT>,
    input_output: &mut Array2D<FloatT>,
) {
    assert_eq!(gamma.shape()[0], hidden.shape()[0]);
    assert_eq!(gamma.shape()[1], hidden.shape()[0]);
    assert_eq!(hidden.shape_as_tuple(), input_output.shape_as_tuple());

    Gemm::gemm(
        &alpha,
        Transpose::NoTrans, &gamma.blas(),
        Transpose::NoTrans, &hidden.blas(),
        &1.,
        &mut input_output.blas());
}

/// `result(i,j) <- result(i,j) * multiplier(i,j) / divisor(i,j)`
#[inline]
pub fn update_from_multiplier_and_divisor(
    multiplier: &Array2D<FloatT>,
    divisor: &Array2D<FloatT>,
    result: &mut Array2D<FloatT>,
) {
    let shape = result.shape_as_tuple();
    assert_eq!(shape, multiplier.shape_as_tuple());
    assert_eq!(shape, divisor.shape_as_tuple());

    for row in 0..shape.0 {
        for col in 0..shape.1 {
            let index = (row, col);
            let mut div = unsafe { divisor.uget(index).clone() };
            // if we have any 0 in any of the matrixes
            // self.weights or self.hidden then
            // divisor will be 0.
            // we can't divide by 0.
            // so we change them to the min positive value.
            if FloatT::zero() == div {
                div = FloatT::min_positive_value();
            }
            unsafe {
                *result.uget_mut(index) *= multiplier.uget(index) / div;
            }
        }
    }
}

pub struct NMFBlas {
    pub hidden: Array2D<FloatT>,
    pub weights: Array2D<FloatT>,

    // these hold temporary results during an iteration.
    // kept in this struct to prevent unnecessary memory allocations.
    pub weights_multiplier: Array2D<FloatT>,
    pub weights_divisor: Array2D<FloatT>,
    pub weights_divisor_reconstruction: Array2D<FloatT>,

    pub hidden_multiplier: Array2D<FloatT>,
    pub hidden_divisor: Array2D<FloatT>,
    pub hidden_divisor_partial: Array2D<FloatT>,

    pub gamma: Array2D<FloatT>,
}

impl NMFBlas {
    pub fn new_random01<R: Rng>(
        nhidden: usize, nobserved: usize, nsamples: usize, rng: &mut R) -> NMFBlas {
        // TODO potentially add some assertions
        let mut hidden = Array2D::<FloatT>::zeros((nhidden, nobserved));
        for x in hidden.iter_mut() {
            *x = random01(rng);
        }

        let mut weights = Array2D::<FloatT>::zeros((nsamples, nhidden));
        for x in weights.iter_mut() {
            *x = random01(rng);
        }

        Self::new(hidden, weights)
    }

    pub fn new(hidden: Array2D<FloatT>, weights: Array2D<FloatT>) -> NMFBlas {
        let hidden_shape = hidden.shape_as_tuple();
        let nhidden = hidden_shape.0;
        let weights_shape = weights.shape_as_tuple();
        assert!(weights_shape.1 == nhidden, "row count of hidden must be equal to column count of weights");

        let nobserved = hidden_shape.1;
        let nsamples = weights_shape.0;

        let samples_shape = (nsamples, nobserved);

        NMFBlas {
            hidden: hidden,
            weights: weights,

            weights_multiplier: Array2D::<FloatT>::zeros(weights_shape),
            weights_divisor: Array2D::<FloatT>::zeros(weights_shape),
            weights_divisor_reconstruction: Array2D::<FloatT>::zeros(samples_shape),

            hidden_multiplier: Array2D::<FloatT>::zeros(hidden_shape),
            hidden_divisor: Array2D::<FloatT>::zeros(hidden_shape),
            hidden_divisor_partial: Array2D::<FloatT>::zeros((nhidden, nhidden)),

            gamma: gamma(nhidden),
        }
    }

    /// returns the number of observed variables
    #[inline]
    pub fn nobserved(&self) -> usize {
        self.hidden.shape()[1]
    }

    /// returns the number of hidden variables
    #[inline]
    pub fn nhidden(&self) -> usize {
        self.hidden.shape()[0]
    }

    /// returns the shape of the hidden matrix
    pub fn hidden_shape(&self) -> (usize, usize) {
        self.hidden.shape_as_tuple()
    }

    /// returns the shape of the weights matrix
    pub fn weights_shape(&self) -> (usize, usize) {
        self.weights.shape_as_tuple()
    }

    /// returns the number of data points
    #[inline]
    pub fn nsamples(&self) -> usize {
        self.weights.shape()[0]
    }

    pub fn samples_shape(&self) -> (usize, usize) {
        (self.nsamples(), self.nobserved())
    }

    // TODO consider calling this something like iteration_step
    // TODO how many iterations ?
    // TODO compare this to the seoung solution
    /// does one iteration step.
    /// `weights` and `hidden` get better and better with each iteration.
    /// usually around `10000` iterations are required.
    /// `samples` contains one observed per column, one sample per row.
    pub fn iterate(
        &mut self,
        samples: &mut Array2D<FloatT>,
        orthogonal_with_alpha: Option<FloatT>,
    ) {
        assert_eq!(samples.shape_as_tuple(), self.samples_shape());

        // weights_multiplier <- samples * hidden.transpose()
        weights_multiplier(
            samples,
            &mut self.hidden,
            &mut self.weights_multiplier);

        // weights_divisor <- weights * hidden * hidden.transpose()
        weights_divisor(
            &mut self.weights,
            &mut self.hidden,
            &mut self.weights_divisor_reconstruction,
            &mut self.weights_divisor);

        // hidden_multiplier <- weights.transpose() * samples
        hidden_multiplier(
            &mut self.weights,
            samples,
            &mut self.hidden_multiplier);

        // hidden_divisor <- weights.transpose() * weights * hidden
        hidden_divisor(
            &mut self.weights,
            &mut self.hidden,
            &mut self.hidden_divisor_partial,
            &mut self.hidden_divisor);

        if let Some(alpha) = orthogonal_with_alpha {
            // hidden_divisor <- hidden_divisor + alpha * gamma * hidden
            add_orthogonalization(
                alpha,
                &mut self.gamma,
                &mut self.hidden,
                &mut self.hidden_divisor);
        }

        // weights(i,j) <-
        //   weights(i,j) * weights_multiplier(i,j) / weights_divisor(i,j)
        update_from_multiplier_and_divisor(
            &self.weights_multiplier,
            &self.weights_divisor,
            &mut self.weights);

        // hidden(i,j) <-
        //   hidden(i,j) * hidden_multiplier(i,j) / hidden_divisor(i,j)
        update_from_multiplier_and_divisor(
            &self.hidden_multiplier,
            &self.hidden_divisor,
            &mut self.hidden);
    }
}
