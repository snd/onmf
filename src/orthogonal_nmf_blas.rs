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

pub type FloatT = f64;

impl<T> ShapeAsTuple<(usize, usize)> for ArrayBase<Vec<T>, (usize, usize)> {
    #[inline]
    fn shape_as_tuple(&self) -> (usize, usize) {
        (self.shape()[0], self.shape()[1])
    }
}

// TODO Dimension type alias
// TODO OwnedArray

pub struct OrthogonalNMFBlas {
    pub hidden: ArrayBase<Vec<FloatT>, (usize, usize)>,
    pub weights: ArrayBase<Vec<FloatT>, (usize, usize)>,

    // these hold temporary results during an iteration.
    // kept in struct to prevent unnecessary memory allocations.
    // TODO dont call these tmp
    pub tmp_weights_dividend: ArrayBase<Vec<FloatT>, (usize, usize)>,
    pub tmp_weights_divisor: ArrayBase<Vec<FloatT>, (usize, usize)>,
    pub tmp_weights_divisor_reconstruction: ArrayBase<Vec<FloatT>, (usize, usize)>,

    pub tmp_hidden_dividend: ArrayBase<Vec<FloatT>, (usize, usize)>,
    pub tmp_hidden_divisor: ArrayBase<Vec<FloatT>, (usize, usize)>,
    pub tmp_hidden_divisor_partial: ArrayBase<Vec<FloatT>, (usize, usize)>,
}

impl OrthogonalNMFBlas
    // where FloatT: 'a + Float + Mul + Zero + Clone + Gemm,
          // ArrayBase<Vec<FloatT>, (usize, usize)>: 'a + AsBlas<FloatT, ArrayBase<Vec<FloatT>, (usize, usize)>, (usize, usize)> + DataOwned<Elem=FloatT> + DataMut<Elem=FloatT>,
          // BlasArrayViewMut<'a, FloatT, (usize, usize)>: Matrix<FloatT>,
          // Closed01<FloatT>: Rand
{
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
            tmp_weights_dividend: ArrayBase::zeros(weights_shape),
            tmp_weights_divisor: ArrayBase::zeros(weights_shape),
            tmp_weights_divisor_reconstruction: ArrayBase::zeros(samples_shape),
            tmp_hidden_dividend: ArrayBase::zeros(hidden_shape),
            tmp_hidden_divisor: ArrayBase::zeros(hidden_shape),
            tmp_hidden_divisor_partial: ArrayBase::zeros((nhidden, nhidden)),
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

    #[inline]
    pub fn iterate_weights_dividend(&mut self, samples: &mut ArrayBase<Vec<FloatT>, (usize, usize)>) {
        Gemm::gemm(
            &1.,
            Transpose::NoTrans, &samples.blas(),
            Transpose::Trans, &self.hidden.blas(),
            &0.,
            &mut self.tmp_weights_dividend.blas());

    }

    // TODO docstring
    #[inline]
    pub fn iterate_weights_divisor(&mut self) {
        Gemm::gemm(
            &1.,
            Transpose::NoTrans, &self.weights.blas(),
            Transpose::NoTrans, &self.hidden.blas(),
            &0.,
            &mut self.tmp_weights_divisor_reconstruction.blas());
        Gemm::gemm(
            &1.,
            Transpose::NoTrans, &self.tmp_weights_divisor_reconstruction.blas(),
            Transpose::Trans, &self.hidden.blas(),
            &0.,
            &mut self.tmp_weights_divisor.blas());
    }

    #[inline]
    pub fn iterate_hidden_dividend(&mut self, samples: &mut ArrayBase<Vec<FloatT>, (usize, usize)>) {
        Gemm::gemm(
            &1.,
            Transpose::Trans, &self.weights.blas(),
            Transpose::NoTrans, &samples.blas(),
            &0.,
            &mut self.tmp_hidden_dividend.blas());
    }

    #[inline]
    pub fn iterate_hidden_divisor(&mut self, alpha: FloatT, samples: &mut ArrayBase<Vec<FloatT>, (usize, usize)>) {
        // // gamma is a symetric matrix with diagonal elements equal to zero
        // // and other elements = alpha
        // let gamma_size = self.nhidden();
        // let mut gamma = DMat::from_elem(gamma_size, gamma_size, alpha);
        //
        // // set diagonal to zero
        // for i in 0..gamma_size {
        //     gamma[(i, i)] = FloatT::zero();
        // }
    }

    // TODO better name
    // per element division
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
        self.iterate_hidden_divisor(alpha, samples);

        OrthogonalNMFBlas::update(
            &self.tmp_weights_dividend,
            &self.tmp_weights_divisor,
            &mut self.weights);
        OrthogonalNMFBlas::update(
            &self.tmp_hidden_dividend,
            &self.tmp_hidden_divisor,
            &mut self.hidden);
    }
}
