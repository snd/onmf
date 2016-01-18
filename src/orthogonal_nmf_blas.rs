use std::ops::{Mul, Add};

use rand::{Rand, Rng, Closed01};
use num::{Float, Zero};

use rblas::{Gemm, Matrix};
use rblas::attribute::Transpose;

use ndarray::{ArrayBase, DataOwned, DataMut};
use ndarray::blas::{BlasArrayViewMut, AsBlas};

use helpers::random01;

pub trait ShapeAsTuple<T> {
    fn shape_as_tuple(&self) -> T;
}

pub type FloatT = f32;

impl<T> ShapeAsTuple<(usize, usize)> for ArrayBase<Vec<T>, (usize, usize)> {
    #[inline]
    fn shape_as_tuple(&self) -> (usize, usize) {
        (self.shape()[0], self.shape()[1])
    }
}

pub type Ix = (usize, usize);
pub type MyMatrix = ArrayBase<Vec<FloatT>, Ix>;

// TODO call this OrthoNMFBlas
pub struct OrthogonalNMFBlas {
    pub hidden: MyMatrix,
    pub weights: MyMatrix,

    // these hold temporary results during an iteration.
    // kept in this struct to prevent unnecessary memory allocations.
    pub weights_dividend: MyMatrix,
    pub weights_divisor: MyMatrix,
    pub weights_divisor_reconstruction: MyMatrix,

    pub hidden_dividend: MyMatrix,
    pub hidden_divisor: MyMatrix,
    pub hidden_divisor_partial: MyMatrix,

    pub gamma: MyMatrix,
}

impl OrthogonalNMFBlas
    // where FloatT: 'a + Float + Mul + Zero + Clone + Gemm,
          // ArrayBase<Vec<FloatT>, (usize, usize)>: 'a + AsBlas<FloatT, ArrayBase<Vec<FloatT>, (usize, usize)>, (usize, usize)> + DataOwned<Elem=FloatT> + DataMut<Elem=FloatT>,
          // BlasArrayViewMut<'a, FloatT, (usize, usize)>: Matrix<FloatT>,
          // Closed01<FloatT>: Rand
{
    pub fn init_random01<R: Rng>(nhidden: usize, nobserved: usize, nsamples: usize, rng: &mut R) -> OrthogonalNMFBlas {
        let mut hidden = MyMatrix::zeros((nhidden, nobserved));
        for x in hidden.iter_mut() {
            *x = random01(rng);
        }

        let mut weights = MyMatrix::zeros((nsamples, nhidden));
        for x in weights.iter_mut() {
            *x = random01(rng);
        }

        Self::init(hidden, weights)
    }

    // TODO initialize with values from last time step t-1 instead of choosing randomly
    pub fn init(hidden: MyMatrix, weights: MyMatrix) -> OrthogonalNMFBlas {
        let weights_shape = weights.shape_as_tuple();
        let hidden_shape = hidden.shape_as_tuple();

        let nhidden = hidden_shape.0;
        let nobserved = hidden_shape.1;
        let nsamples = weights_shape.0;

        let samples_shape = (nsamples, nobserved);

        OrthogonalNMFBlas {
            hidden: hidden,
            weights: weights,

            weights_dividend: MyMatrix::zeros(weights_shape),
            weights_divisor: MyMatrix::zeros(weights_shape),
            weights_divisor_reconstruction: MyMatrix::zeros(samples_shape),

            hidden_dividend: MyMatrix::zeros(hidden_shape),
            hidden_divisor: MyMatrix::zeros(hidden_shape),
            hidden_divisor_partial: MyMatrix::zeros((nhidden, nhidden)),

            gamma: OrthogonalNMFBlas::gamma(nhidden),
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

    /// returns the number of data points
    #[inline]
    pub fn nsamples(&self) -> usize {
        self.weights.shape()[0]
    }

    pub fn samples_shape(&self) -> (usize, usize) {
        (self.nsamples(), self.nobserved())
    }

    /// gamma is a symetric matrix with diagonal elements equal to zero
    /// and other elements equal to 1
    pub fn gamma(size: usize) -> MyMatrix {
        let mut gamma = MyMatrix::from_elem((size, size), 1.);
        for x in gamma.diag_mut().iter_mut() {
            *x = 0.
        }
        gamma
    }

    /// `samples * hidden.transpose()`
    #[inline]
    pub fn iterate_weights_dividend(&mut self, samples: &mut MyMatrix) {
        Gemm::gemm(
            &1.,
            Transpose::NoTrans, &samples.blas(),
            Transpose::Trans, &self.hidden.blas(),
            &0.,
            &mut self.weights_dividend.blas());

    }

    /// `weights * hidden * hidden.transpose()`
    #[inline]
    pub fn iterate_weights_divisor(&mut self) {
        Gemm::gemm(
            &1.,
            Transpose::NoTrans, &self.weights.blas(),
            Transpose::NoTrans, &self.hidden.blas(),
            &0.,
            &mut self.weights_divisor_reconstruction.blas());
        Gemm::gemm(
            &1.,
            Transpose::NoTrans, &self.weights_divisor_reconstruction.blas(),
            Transpose::Trans, &self.hidden.blas(),
            &0.,
            &mut self.weights_divisor.blas());
    }

    /// `weights.transpose() * samples`
    #[inline]
    pub fn iterate_hidden_dividend(&mut self, samples: &mut MyMatrix) {
        Gemm::gemm(
            &1.,
            Transpose::Trans, &self.weights.blas(),
            Transpose::NoTrans, &samples.blas(),
            &0.,
            &mut self.hidden_dividend.blas());
    }

    /// `weights.transpose() * weights * hidden + alpha * gamma * hidden`
    #[inline]
    pub fn iterate_hidden_divisor(&mut self, alpha: FloatT) {
        // we must make a copy here because we need two mutable
        // references to weights at the same time for .blas()
        // weights_copy.clone_from(&weights);
        // TODO this will be solved once
        // https://github.com/bluss/rust-ndarray/pull/40
        // is merged.
        let mut weights_copy = self.weights.clone();
        Gemm::gemm(
            &1.,
            Transpose::Trans, &self.weights.blas(),
            Transpose::NoTrans, &weights_copy.blas(),
            &0.,
            &mut self.hidden_divisor_partial.blas());
        Gemm::gemm(
            &alpha,
            Transpose::NoTrans, &self.gamma.blas(),
            Transpose::NoTrans, &self.hidden.blas(),
            &0.,
            &mut self.hidden_divisor.blas());
        Gemm::gemm(
            &1.,
            Transpose::NoTrans, &self.hidden_divisor_partial.blas(),
            Transpose::NoTrans, &self.hidden.blas(),
            // add to previous contents of self.hidden_divisor
            &1.,
            &mut self.hidden_divisor.blas());
    }

    // TODO better name
    // per element division
    /// `result(i,j) <- result(i,j) * dividend(i,j) / divisor(i,j)`
    #[inline]
    pub fn update(
        dividend: &MyMatrix,
        divisor: &MyMatrix,
        result: &mut MyMatrix,
    ) {
        let shape = result.shape_as_tuple();
        assert_eq!(shape, dividend.shape_as_tuple());
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
                    *result.uget_mut(index) *= dividend.uget(index) / div;
                }
            }
        }
    }

    // TODO consider calling this something like iteration_step
    // TODO how many iterations ?
    // TODO compare this to the seoung solution
    /// it gets better and better with each iteration.
    /// one observed per column.
    /// one sample per row.
    pub fn iterate(&mut self, alpha: FloatT, samples: &mut MyMatrix) {
        assert_eq!(samples.shape_as_tuple(), self.samples_shape());

        // weights_dividend <- samples * hidden.transpose()
        self.iterate_weights_dividend(samples);
        // weights_divisor <- weights * hidden * hidden.transpose()
        self.iterate_weights_divisor();
        // hidden_dividend <- weights.transpose() * samples
        self.iterate_hidden_dividend(samples);
        // hidden_divisor <-
        // weights.transpose() * weights * hidden + alpha * gamma * hidden
        self.iterate_hidden_divisor(alpha);

        // weights(i,j) <-
        // weights(i,j) * weights_dividend(i,j) / weights_divisor(i,j)
        OrthogonalNMFBlas::update(
            &self.weights_dividend,
            &self.weights_divisor,
            &mut self.weights);
        // hidden(i,j) <-
        // hidden(i,j) * hidden_dividend(i,j) / hidden_divisor(i,j)
        OrthogonalNMFBlas::update(
            &self.hidden_dividend,
            &self.hidden_divisor,
            &mut self.hidden);
    }
}
