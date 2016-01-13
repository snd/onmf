extern crate nalgebra;
use self::nalgebra::{DMat};

extern crate rand;
use self::rand::{Rand};

pub struct OrthogonalNMF<FloatT> {
    // TODO add docstrings
    pub hidden: DMat<FloatT>,
    pub weights: DMat<FloatT>,
}

impl<FloatT: Rand> OrthogonalNMF<FloatT> {
    /// one observed per column.
    /// one time per row.
    pub fn from_data(data: &DMat<FloatT>, nhidden: usize, niterations: usize) -> OrthogonalNMF<FloatT> {
        let nobserved = data.ncols();
        let ntimes = data.nrows();
        let mut nmf = OrthogonalNMF {
            hidden: DMat::new_random(nhidden, nobserved),
            weights: DMat::new_random(ntimes, nhidden),
        };
        // TODO how many iterations ?
        // it gets better and better with each iteration
        // for i in 0..niterations {
        //     // TODO alpha goes to infinity
        //     // detail factor
        //     // tunable
        //     let alpha = 0.1 * 1.01.powi(i);
        //     nmf.iterate(data, alpha);
        // }
        nmf
    }

    // TODO maybe a function to reuse an existing OrthogonalNMF
    // in order to avoid unnecessary memory allocations

    // TODO initialize with values from last time step t-1 instead of choosing randomly
    // init_manually

    // TODO compare this to the seoung solution
    // #[inline]
    // fn iterate(&mut self, data: &DMat<f64>, alpha: f64) {
    //     let latents_transposed = self.latents.transpose();
    //     let weights_transposed = self.weights.transpose();
    //
    //     // TODO names
    //     let weights_foo = data.mul(latents_transposed);
    //     let weights_bar = self.weights.mul(self.latents).mul(latents_transposed);
    //
    //     let latents_foo = weights_transposed.mul(data);
    //     let size = ?
    //     let gamma = DMat::from_elem(size, size, alpha);
    //     let zeroes = DVec::new_zeros(size);
    //     gamma.set_diag(zeroes);
    //     let latents_bar = weights_transposed
    //         .mul(self.weights)
    //         .mul(self.latents)
    //         // we add the previous latents
    //         // multiplied by alpha except for the diag which is set to zero
    //         .add(gamma.mul(self.latents));
    //
    //     // iterate weights
    //     // TODO correct order ?
    //     for col in 0..self.weights.ncols() {
    //         for row in 0..self.weights.nrows() {
    //             let index = (row, col);
    //             let foo_value = unsafe { weights_foo.unsafe_at(index) }
    //             // these get larger and larger
    //             let bar_value = unsafe { weights_bar.unsafe_at(index) }
    //             let old_value = unsafe { self.weights.unsafe_at(index) }
    //             let new_value = old_value * foo_value / bar_value;
    //             unsafe { self.weights.unsafe_set(index, new_value) }
    //         }
    //     }
    //
    //     // iterate latents
    //     // TODO correct order ?
    //     for col in 0..self.latents.ncols() {
    //         for row in 0..self.latents.nrows() {
    //             let index = (row, col);
    //             let foo_value = latents_foo[index];
    //             let bar_value = latents_bar[index]
    //             let old_value = self.latents[index];
    //             let new_value = old_value * foo_value / bar_value;
    //             self.latents[index] = new_value;
    //         }
    //     }
    // }
}
