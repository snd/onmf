use std::ops::{Mul};

extern crate nalgebra;
use self::nalgebra::{DMat, Transpose};

extern crate rand;
use self::rand::{Rand};

extern crate num;
use self::num::{Float, FromPrimitive};

pub struct OrthogonalNMF<FloatT> {
    // TODO add docstrings
    pub hidden: DMat<FloatT>,
    pub weights: DMat<FloatT>,
    pub iteration: usize,
}

impl<FloatT: Rand + Float + FromPrimitive + Mul> OrthogonalNMF<FloatT> {
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
        let nsamples = self.weights.nrows();
        let nobserved = self.hidden.ncols();
        assert_eq!(nsamples, data.nrows());
        assert_eq!(nobserved, data.ncols());
        let alpha: FloatT =
            FloatT::from_f32(0.1).unwrap() *
            FloatT::from_f32(1.01).unwrap().powi(self.iteration as i32);

        let hidden_transposed = self.hidden.transpose();
        let weights_transposed = self.weights.transpose();

        let new_weights_dividend = data.clone().mul(&hidden_transposed);
        let new_weights_divsor = self.weights.clone().mul(&self.hidden).mul(&hidden_transposed);

        let new_hidden_dividend = weights_transposed.clone().mul(data);

        // let size = ?
        // let gamma = DMat::from_elem(size, size, alpha);
        // let zeroes = DVec::new_zeros(size);
        // gamma.set_diag(zeroes);
        // let latents_bar = weights_transposed
        //     .mul(self.weights)
        //     .mul(self.latents)
        //     // we add the previous latents
        //     // multiplied by alpha except for the diag which is set to zero
        //     .add(gamma.mul(self.latents));

        // // iterate weights
        // // TODO correct order ?
        // for col in 0..self.weights.ncols() {
        //     for row in 0..self.weights.nrows() {
        //         let index = (row, col);
        //         let foo_value = unsafe { weights_foo.unsafe_at(index) }
        //         // these get larger and larger
        //         let bar_value = unsafe { weights_bar.unsafe_at(index) }
        //         let old_value = unsafe { self.weights.unsafe_at(index) }
        //         let new_value = old_value * foo_value / bar_value;
        //         unsafe { self.weights.unsafe_set(index, new_value) }
        //     }
        // }
        //
        // // iterate latents
        // // TODO correct order ?
        // for col in 0..self.latents.ncols() {
        //     for row in 0..self.latents.nrows() {
        //         let index = (row, col);
        //         let foo_value = latents_foo[index];
        //         let bar_value = latents_bar[index]
        //         let old_value = self.latents[index];
        //         let new_value = old_value * foo_value / bar_value;
        //         self.latents[index] = new_value;
        //     }
        // }
    }
}
