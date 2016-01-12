extern crate nalgebra;
use self::nalgebra::{DMat};

pub struct OrthogonalNMF<FloatT> {
    /// maps hidden variables (= rows) to observed variables (= cols).
    /// changes on every `update`.
    /// stays constant in size.
    pub hidden: DMat<FloatT>,
    /// maps times (= rows) to hidden variables (= cols).
    /// changes on every `update`.
    /// inspect this.
    /// will grow in size.
    pub weights: DMat<FloatT>,
}
