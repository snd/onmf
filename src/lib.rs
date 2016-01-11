extern crate nalgebra;
use nalgebra::{DMat};

extern crate num;
use num::traits::{Zero};

pub struct OnlineNMF<FloatT> {
    /// maps times (= rows) to hidden variables (= cols).
    /// will grow in size.
    pub weights: DMat<FloatT>,
    /// maps hidden variables (= rows) to observed variables (= cols).
    /// stays constant in size.
    pub hidden: DMat<FloatT>,
}

impl<FloatT: Zero + Clone + Copy> OnlineNMF<FloatT> {
    pub fn new(nobserved: usize, nhidden: usize) -> OnlineNMF<FloatT> {
        OnlineNMF {
            hidden: DMat::new_zeros(nhidden, nobserved),
            weights: DMat::new_zeros(0, nhidden)
        }
    }

    #[inline]
    pub fn nobserved(&self) -> usize {
        self.hidden.ncols()
    }

    #[inline]
    pub fn nhidden(&self) -> usize {
        self.hidden.nrows()
    }

    pub fn update(&mut self, new_observed_as_columns: &DMat<FloatT>) {
        assert_eq!(self.nobserved(), new_observed_as_columns.nrows());

        // do nothing for now
    }
}
