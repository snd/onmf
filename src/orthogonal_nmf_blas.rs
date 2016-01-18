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

// TODO Dimension type alias
// TODO OwnedArray

// TODO call this OrthoNMFBlas
pub struct OrthogonalNMFBlas {
    pub hidden: ArrayBase<Vec<FloatT>, (usize, usize)>,
    pub weights: ArrayBase<Vec<FloatT>, (usize, usize)>,

    // these hold temporary results during an iteration.
    // kept in struct to prevent unnecessary memory allocations.
    pub weights_dividend: ArrayBase<Vec<FloatT>, (usize, usize)>,
    pub weights_divisor: ArrayBase<Vec<FloatT>, (usize, usize)>,
    pub weights_divisor_reconstruction: ArrayBase<Vec<FloatT>, (usize, usize)>,

    pub hidden_dividend: ArrayBase<Vec<FloatT>, (usize, usize)>,
    pub hidden_divisor: ArrayBase<Vec<FloatT>, (usize, usize)>,
    pub hidden_divisor_partial: ArrayBase<Vec<FloatT>, (usize, usize)>,
}

impl OrthogonalNMFBlas
    // where FloatT: 'a + Float + Mul + Zero + Clone + Gemm,
          // ArrayBase<Vec<FloatT>, (usize, usize)>: 'a + AsBlas<FloatT, ArrayBase<Vec<FloatT>, (usize, usize)>, (usize, usize)> + DataOwned<Elem=FloatT> + DataMut<Elem=FloatT>,
          // BlasArrayViewMut<'a, FloatT, (usize, usize)>: Matrix<FloatT>,
          // Closed01<FloatT>: Rand
{
    // TODO type alias for that ArrayBase stuff

    pub fn init_random01<R: Rng>(nhidden: usize, nobserved: usize, nsamples: usize, rng: &mut R) -> OrthogonalNMFBlas {
        let mut hidden = ArrayBase::zeros((nhidden, nobserved));
        for x in hidden.iter_mut() {
            *x = random01(rng);
        }

        let mut weights = ArrayBase::zeros((nsamples, nhidden));
        for x in weights.iter_mut() {
            *x = random01(rng);
        }

        Self::init(hidden, weights)
    }

    // TODO initialize with values from last time step t-1 instead of choosing randomly
    pub fn init(hidden: ArrayBase<Vec<FloatT>, (usize, usize)>, weights: ArrayBase<Vec<FloatT>, (usize, usize)>) -> OrthogonalNMFBlas {
        let weights_shape = weights.shape_as_tuple();
        let hidden_shape = hidden.shape_as_tuple();

        let nhidden = hidden_shape.0;
        let nobserved = hidden_shape.1;
        let nsamples = weights_shape.0;

        let samples_shape = (nsamples, nobserved);

        OrthogonalNMFBlas {
            hidden: hidden,
            weights: weights,
            weights_dividend: ArrayBase::zeros(weights_shape),
            weights_divisor: ArrayBase::zeros(weights_shape),
            weights_divisor_reconstruction: ArrayBase::zeros(samples_shape),
            hidden_dividend: ArrayBase::zeros(hidden_shape),
            hidden_divisor: ArrayBase::zeros(hidden_shape),
            hidden_divisor_partial: ArrayBase::zeros((nhidden, nhidden)),
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

    /// `samples * hidden.transpose()`
    #[inline]
    pub fn iterate_weights_dividend(&mut self, samples: &mut ArrayBase<Vec<FloatT>, (usize, usize)>) {
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
    pub fn iterate_hidden_dividend(&mut self, samples: &mut ArrayBase<Vec<FloatT>, (usize, usize)>) {
        Gemm::gemm(
            &1.,
            Transpose::Trans, &self.weights.blas(),
            Transpose::NoTrans, &samples.blas(),
            &0.,
            &mut self.hidden_dividend.blas());
    }

    /// gamma is a symetric matrix with diagonal elements equal to zero
    /// and other elements = alpha
    pub fn alpha_to_gamma(alpha: FloatT, nhidden: usize) -> ArrayBase<Vec<FloatT>, (usize, usize)> {
        let mut gamma = ArrayBase::<Vec<FloatT>, (usize, usize)>::from_elem((nhidden, nhidden), alpha);
        for x in gamma.diag_mut().iter_mut() {
            *x = 0.
        }
        gamma
    }

    /// `weights.transpose() * weights * hidden + alpha * gamma * hidden`
    #[inline]
    pub fn iterate_hidden_divisor(&mut self, samples: &mut ArrayBase<Vec<FloatT>, (usize, usize)>, gamma: &mut ArrayBase<Vec<FloatT>, (usize, usize)>) {
        // we must make a copy here because we need two mutable
        // references to weights at the same time for .blas()
        // weights_copy.clone_from(&weights);
        // TODO is there away to prevent this
        // maybe use clone_from here
        let mut weights_copy = self.weights.clone();
        Gemm::gemm(
            &1.,
            Transpose::Trans, &self.weights.blas(),
            Transpose::NoTrans, &weights_copy.blas(),
            &0.,
            &mut self.hidden_divisor_partial.blas());
        Gemm::gemm(
            &1.,
            Transpose::NoTrans, &gamma.blas(),
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
        dividend: &ArrayBase<Vec<FloatT>, (usize, usize)>,
        divisor: &ArrayBase<Vec<FloatT>, (usize, usize)>,
        result: &mut ArrayBase<Vec<FloatT>, (usize, usize)>,
    ) {
        let shape = result.shape_as_tuple();
        assert_eq!(shape, dividend.shape_as_tuple());
        assert_eq!(shape, divisor.shape_as_tuple());

        for row in 0..shape.0 {
            for col in 0..shape.1 {
                let index = (row, col);
                let mut div = unsafe { divisor.uget(index).clone() };
                // if we have any zero in any of the matrizes
                // self.weights or self.hidden then
                // divisor will be zero.
                // we can't divide by zero
                if FloatT::zero() == div {
                    div = FloatT::min_positive_value();
                }
                unsafe {
                    *result.uget_mut(index) *= dividend.uget(index) / div;
                }
                // TODO maybe also zero adjust the final weight
            }
        }
    }

    // TODO consider calling this something like iteration_step
    // TODO how many iterations ?
    // TODO compare this to the seoung solution
    /// it gets better and better with each iteration.
    /// one observed per column.
    /// one sample per row.
    pub fn iterate(&mut self, alpha: FloatT, samples: &mut ArrayBase<Vec<FloatT>, (usize, usize)>) {
        assert_eq!(samples.shape_as_tuple(), self.samples_shape());

        // TODO benchmark these individually
        // TODO add little descriptions like
        // X <- X^T
        // TODO also add them to docstrings
        // TODO maybe pass in all data explicitely
        self.iterate_weights_dividend(samples);
        self.iterate_weights_divisor();
        self.iterate_hidden_dividend(samples);
        let mut gamma = OrthogonalNMFBlas::alpha_to_gamma(
            alpha, self.nhidden());
        self.iterate_hidden_divisor(samples, &mut gamma);

        OrthogonalNMFBlas::update(
            &self.weights_dividend,
            &self.weights_divisor,
            &mut self.weights);
        OrthogonalNMFBlas::update(
            &self.hidden_dividend,
            &self.hidden_divisor,
            &mut self.hidden);
    }
}
