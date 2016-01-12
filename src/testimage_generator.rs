/*!
generates images used to test NMF

reproducably (seeded) generate test data.

a series of pictures.
each picture is generated from a linear combination of factors.
the coefficients of the linear combination are chosen randomly.

two evolving factors change over time.
the two evolving factors change 6 times.
we generate 1000 pictures for each change of the evolving factors
during which the evolving factors stay constant.

the series of pictures is fed into ONMF one at a time.
a picture is fed as a 1D array (series of values, slice) into the ONMF
and not as a 2D array.
*/

use std::ops::Range;
use std::iter::{Map, Rev, once, Once, Chain, Zip};

extern crate nalgebra;
use self::nalgebra::{DMat};

extern crate num;
use self::num::traits::{Zero, One};

pub type HorizontalReturnT<T> = Map<Range<usize>, fn(usize) -> T>;
pub type VerticalReturnT<T> = Map<Rev<Range<usize>>, fn(usize) -> T>;
pub type StaticReturnT<T> = Chain<Chain<Chain<Once<T>, VerticalReturnT<T>>, Once<T>>, HorizontalReturnT<T>>;
pub type TestReturnT<T> = Map<Zip<HorizontalReturnT<T>, VerticalReturnT<T>>, fn((T, T)) -> T>;

/// returns a matrix that has `1` in `row` for all `cols` and `0` everywhere else
///
/// ```
/// # use onmf::testimage_generator::horizontal_line;
/// horizontal_line::<f64, _>(9, 0..5);
/// ```
pub fn horizontal_line<T, I>(row: usize, cols: I) -> DMat<T>
    where T: Clone + Copy + Zero + One,
          I: Iterator<Item = usize>
{
    let mut factor: DMat<T> = DMat::new_zeros(10, 10);
    for col in cols {
        factor[(row, col)] = T::one();
    }
    factor
}

pub fn horizontal_evolving_factors<T>() -> HorizontalReturnT<DMat<T>>
    where T: Clone + Copy + Zero + One
{
    fn helper<U: Clone + Copy + Zero + One>(start_col: usize) -> DMat<U> {
        horizontal_line(9, start_col..(start_col + 5))
    }
    (0..6).map(helper)
}

/// returns a matrix that has `1` in `col` for all `rows` and `0` everywhere else
///
/// ```
/// # use onmf::testimage_generator::vertical_line;
/// vertical_line::<f64, _>(5..10, 0);
/// ```
pub fn vertical_line<T, I>(rows: I, col: usize) -> DMat<T>
    where T: Clone + Copy + Zero + One,
          I: Iterator<Item = usize>
{
    let mut factor: DMat<T> = DMat::new_zeros(10, 10);
    for row in rows {
        factor[(row, col)] = T::one();
    }
    factor
}

pub fn vertical_evolving_factors<T>() -> VerticalReturnT<DMat<T>>
    where T: Clone + Copy + Zero + One
{
    fn helper<U: Clone + Copy + Zero + One>(start_row: usize) -> DMat<U> {
        vertical_line(start_row..(start_row + 5), 0)
    }
    (0..6).rev().map(helper)
}

pub fn static_factors<T>() -> StaticReturnT<DMat<T>>
    where T: Clone + Copy + Zero + One
{
    fn helper_horizontal<U: Clone + Copy + Zero + One>(row: usize) -> DMat<U> {
        horizontal_line(row, 0..10)
    }
    fn helper_vertical<U: Clone + Copy + Zero + One>(col: usize) -> DMat<U> {
        vertical_line(0..10, col)
    }
    once(horizontal_line(9, 0..5))
        .chain((0..9).rev().map(helper_horizontal as fn(usize) -> DMat<T>))
        .chain(once(vertical_line(5..10, 0)))
        .chain((1..10).map(helper_vertical as fn(usize) -> DMat<T>))
}
