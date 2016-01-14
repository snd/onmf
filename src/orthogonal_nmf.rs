use std::ops::{Mul, Add};

use std::fmt::{Display};

extern crate nalgebra;
use self::nalgebra::{DMat, Transpose};

extern crate rand;
use self::rand::{Rand};

extern crate num;
use self::num::{Float, Zero, FromPrimitive};

pub struct OrthogonalNMF<FloatT> {
    // TODO add docstrings
    pub hidden: DMat<FloatT>,
    pub weights: DMat<FloatT>,
    pub iteration: usize,
}

impl<FloatT: Rand + Float + FromPrimitive + Mul + Zero + Display> OrthogonalNMF<FloatT> {
    pub fn init_randomly(nhidden: usize, nobserved: usize, nsamples: usize) -> OrthogonalNMF<FloatT> {
        OrthogonalNMF {
            hidden: DMat::new_random(nhidden, nobserved),
            weights: DMat::new_random(nsamples, nhidden),
            iteration: 0,
        }
    }

    // TODO initialize with values from last time step t-1 instead of choosing randomly
    pub fn init(hidden: DMat<FloatT>, weights: DMat<FloatT>) -> OrthogonalNMF<FloatT> {
        OrthogonalNMF {
            hidden: hidden,
            weights: weights,
            iteration: 0,
        }
    }

    // TODO how many iterations ?
    // TODO compare this to the seoung solution
    /// it gets better and better with each iteration.
    /// one observed per column.
    /// one sample per row.
    pub fn iterate(&mut self, data: &DMat<FloatT>) {
        let nhidden = self.hidden.nrows();
        let nobserved = self.hidden.ncols();
        let nsamples = self.weights.nrows();
        assert_eq!(nsamples, data.nrows());
        assert_eq!(nobserved, data.ncols());

        // alpha gets larger and larger with each iteration
        // 0.1, 0.101, 0.102, 0.103, ...
        // at iteration 232 alpha first goes above 1.0
        // iteration = 232 -> alpha = 1.005
        let alpha: FloatT =
            FloatT::from_f32(0.1).unwrap() *
            FloatT::from_f32(1.01).unwrap().powi(self.iteration as i32);

        println!("iteration = {} alpha = {}", self.iteration, alpha);

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
        let gamma_size = nhidden;
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
        // TODO efficient order ?
        for col in 0..self.weights.ncols() {
            for row in 0..self.weights.nrows() {
                let index = (row, col);
                // if we have any zero in any of the matrizes
                // self.weights or self.hidden then
                // new_weights_divisor[index] will be zero
                // TODO how to deal with this
                debug_assert!(FloatT::zero() != new_weights_divisor[index]);
                self.weights[index] =
                    self.weights[index] *
                    new_weights_dividend[index] /
                    // these get larger and larger
                    new_weights_divisor[index];
            }
        }

        // compute new hidden
        // TODO efficient order ?
        for col in 0..self.hidden.ncols() {
            for row in 0..self.hidden.nrows() {
                let index = (row, col);
                debug_assert!(FloatT::zero() != new_hidden_divisor[index]);
                self.hidden[index] =
                    self.hidden[index] *
                    new_hidden_dividend[index] /
                    new_hidden_divisor[index];
            }
        }

        self.iteration += 1;
    }
}
