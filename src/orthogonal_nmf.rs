use std::ops::{Mul, Add};

use nalgebra::{DMat, Transpose};
use rand::{Rand, Rng, Closed01};
use num::{Float, Zero};

use helpers::random01;

pub struct OrthogonalNMF<FloatT> {
    // TODO add docstrings
    pub hidden: DMat<FloatT>,
    pub weights: DMat<FloatT>,
}

impl<FloatT> OrthogonalNMF<FloatT>
    where FloatT: Float + Mul + Zero,
          Closed01<FloatT>: Rand
{
    pub fn init_random01<R: Rng>(nhidden: usize, nobserved: usize, nsamples: usize, rng: &mut R) -> OrthogonalNMF<FloatT> {
        let mut hidden = unsafe { DMat::new_uninitialized(nhidden, nobserved) };
        for x in hidden.as_mut_vec().iter_mut() {
            *x = random01(rng);
        }

        let mut weights = unsafe { DMat::new_uninitialized(nsamples, nhidden) };
        for x in weights.as_mut_vec().iter_mut() {
            *x = random01(rng);
        }

        Self::init(hidden, weights)
    }

    // TODO initialize with values from last time step t-1 instead of choosing randomly
    pub fn init(hidden: DMat<FloatT>, weights: DMat<FloatT>) -> OrthogonalNMF<FloatT> {
        OrthogonalNMF {
            hidden: hidden,
            weights: weights,
        }
    }

    /// returns the number of observed variables
    #[inline]
    pub fn nobserved(&self) -> usize {
        self.hidden.ncols()
    }

    /// returns the number of hidden variables
    #[inline]
    pub fn nhidden(&self) -> usize {
        self.hidden.nrows()
    }

    /// returns the number of data points
    #[inline]
    pub fn nsamples(&self) -> usize {
        self.weights.nrows()
    }

    // TODO how many iterations ?
    // TODO compare this to the seoung solution
    /// it gets better and better with each iteration.
    /// one observed per column.
    /// one sample per row.
    pub fn iterate(&mut self, alpha: FloatT, data: &DMat<FloatT>) {
        assert_eq!(self.nsamples(), data.nrows());
        assert_eq!(self.nobserved(), data.ncols());

        let hidden_transposed = self.hidden.transpose();
        let weights_transposed = self.weights.transpose();

        // has the same shape as weights
        let new_weights_dividend = data.clone().mul(&hidden_transposed);
        // has the same shape as weights
        let new_weights_divisor = self.weights.clone().mul(&self.hidden).mul(&hidden_transposed);

        // has the same shape as hidden
        let new_hidden_dividend = weights_transposed.clone().mul(data);

        // gamma is a symetric matrix with diagonal elements equal to zero
        // and other elements = alpha
        let gamma_size = self.nhidden();
        let mut gamma = DMat::from_elem(gamma_size, gamma_size, alpha);

        // set diagonal to zero
        for i in 0..gamma_size {
            gamma[(i, i)] = FloatT::zero();
        }

        // has the same shape as hidden
        let new_hidden_divisor = weights_transposed.clone()
            .mul(&self.weights)
            .mul(&self.hidden)
            // we add the previous latents
            // multiplied by alpha except for the diag which is set to zero
            .add(gamma.mul(&self.hidden));

        // compute new weights
        // TODO possibly use simd for this
        for col in 0..self.weights.ncols() {
            for row in 0..self.weights.nrows() {
                let index = (row, col);
                // if we have any zero in any of the matrizes
                // self.weights or self.hidden then
                // new_weights_divisor[index] will be zero
                // TODO how to deal with this
                let mut divisor = new_weights_divisor[index];
                if FloatT::zero() == divisor {
                    divisor = FloatT::min_positive_value();
                }
                debug_assert!(FloatT::zero() != divisor);
                self.weights[index] =
                    self.weights[index] *
                    new_weights_dividend[index] /
                    divisor;
                if FloatT::zero() == self.weights[index] {
                    self.weights[index] = FloatT::min_positive_value();
                }
                debug_assert!(FloatT::zero() != self.weights[index]);
            }
        }

        // compute new hidden
        // TODO possibly use simd for this
        for col in 0..self.hidden.ncols() {
            for row in 0..self.hidden.nrows() {
                let index = (row, col);
                let mut divisor = new_hidden_divisor[index];
                if FloatT::zero() == divisor {
                    divisor = FloatT::min_positive_value();
                }
                debug_assert!(FloatT::zero() != divisor);
                self.hidden[index] =
                    self.hidden[index] *
                    new_hidden_dividend[index] /
                    divisor;
                if FloatT::zero() == self.hidden[index] {
                    self.hidden[index] = FloatT::min_positive_value();
                }
                debug_assert!(FloatT::zero() != self.hidden[index]);
            }
        }
    }
}
