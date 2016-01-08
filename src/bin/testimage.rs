use std::ops::IndexMut;

extern crate nalgebra;
use nalgebra::{DMat};

type FloatType = f64;
type Mat = DMat<FloatType>;

fn horizontal_line<I: Iterator<Item = usize>>(row: usize, cols: I) -> Mat {
    let mut factor: Mat = DMat::new_zeros(10, 10);
    for col in cols {
        *factor.index_mut((row, col)) = 1.;
    }
    factor
}

fn vertical_line<I: Iterator<Item = usize>>(rows: I, col: usize) -> Mat {
    let mut factor: Mat = DMat::new_zeros(10, 10);
    for row in rows {
        *factor.index_mut((row, col)) = 1.;
    }
    factor
}

fn main() {
    let mut factors: Vec<DMat<f64>> = Vec::new();

    factors.push(horizontal_line(9, 0..5));
    for row in (0..9).rev() {
        factors.push(horizontal_line(row, 0..10));
    }

    factors.push(vertical_line(5..10, 0));
    for col in 1..10 {
        factors.push(vertical_line(0..10, col));
    }

    for factor in factors {
        println!("{:?}", factor);
    }
}

