use std::ops::Div;

use helpers::PartialMaxIteratorExt;

extern crate nalgebra;
use self::nalgebra::DMat;

extern crate num;
use self::num::{Zero};

pub trait Normalize {
    /// divide all values in `self` by max value in `self`
    fn normalize(self) -> Self;
}

impl<T> Normalize for DMat<T>
    where T: PartialOrd + Clone + Copy + Zero,
          DMat<T>: Div<T, Output=DMat<T>>
{
    fn normalize(self) -> Self {
        assert!(0 < self.ncols());
        assert!(0 < self.nrows());
        let max = self.as_vec().iter().cloned().partial_max().unwrap();
        if max == T::zero() {
            self
        } else {
            self.div(max)
        }
    }
}

#[test]
fn test_normalize_zeros() {
    let zeros = DMat::<f64>::new_zeros(10, 10);
    assert_eq!(zeros.clone(), zeros.normalize());
}

#[test]
fn test_normalize_ones() {
    let ones = DMat::<f64>::new_ones(10, 10);
    assert_eq!(ones.clone(), ones.normalize());
}
