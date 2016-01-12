/// generates images used to test NMF

use std::ops::{Index, IndexMut, Mul, Add, Div};

use std;

use std::cmp::PartialOrd;

use std::ops::Range;
use std::iter::{Map, Rev, once, Once, Chain, Zip};

extern crate num;
use num::traits::{Zero, One, FromPrimitive, ToPrimitive, Float};

extern crate nalgebra;
use nalgebra::{DMat, DVec, Norm, BaseFloat};

extern crate image;
use self::image::{ImageBuffer, Luma, DynamicImage};

use partialmax::PartialMaxIteratorExt;

quick_error! {
    #[derive(Debug)]
    pub enum ImageSaveError {
        /// an error has occured when opening the file
        Io(err: std::io::Error) {}
        /// an error has occured when writing the image into the file
        Image(err: image::ImageError) {}
    }
}

pub type HorizontalReturnT<T> = Map<Range<usize>, fn(usize) -> T>;
pub type VerticalReturnT<T> = Map<Rev<Range<usize>>, fn(usize) -> T>;
pub type StaticReturnT<T> = Chain<Chain<Chain<Once<T>, VerticalReturnT<T>>, Once<T>>, HorizontalReturnT<T>>;
pub type TestReturnT<T> = Map<Zip<HorizontalReturnT<T>, VerticalReturnT<T>>, fn((T, T)) -> T>;

// TODO move the type constraints back up here
// TODO split into multiple traits
// normalize
// images
// creation
pub trait Testimage<T> {
    type Item;

    /// returns a testimage that has `1` in `row` for all `cols` and `0` everywhere else
    ///
    /// ```
    /// Testimage::new_horizontal_line(9, 0..5);
    /// ```
    fn new_horizontal_line<I: Iterator<Item = usize>>(row: usize, cols: I) -> Self where Self::Item: Clone + Copy + Zero + One;
    /// returns a testimage that has `1` in `col` for all `rows` and `0` everywhere else
    ///
    /// ```
    /// Testimage::new_vertical_line(5..10, 0);
    /// ```
    fn new_vertical_line<I: Iterator<Item = usize>>(rows: I, col: usize) -> Self where Self::Item: Clone + Copy + Zero + One;
    /// # Panics
    /// panics unless all the values in `self` are between
    /// `0` (inclusive) and `1` (inclusive)
    fn luma01_to_image(&self) -> DynamicImage where Self::Item: Zero + One + FromPrimitive + ToPrimitive + PartialOrd + Float;
    /// # Panics
    /// panics unless all the values in `self` are between
    /// `0` (inclusive) and `1` (inclusive)
    fn save_luma01_to_png(&self, filename: &str) -> Result<(), ImageSaveError> where Self::Item: Zero + One + FromPrimitive + ToPrimitive + PartialOrd + Float;
    fn static_factors() -> StaticReturnT<Self>
        where Self: Sized,
              Self::Item: Clone + Copy + Zero + One;
    // TODO it's probably a much better idea to just use plain functions here
    fn horizontal_evolving_factors() -> HorizontalReturnT<Self> where Self::Item: Clone + Copy + Zero + One;
    fn vertical_evolving_factors() -> VerticalReturnT<Self> where Self::Item: Clone + Copy + Zero + One;
    // TODO trait
    fn linear_combination<C>(factors: &[Self], coefficients: &[C]) -> Self
        where C: Clone,
              Self: Sized + Mul<C, Output = Self>,
              Self::Item: Clone + Copy + Zero;
    fn normalize_mut(self) -> Self
        where Self::Item: Clone + Copy + BaseFloat,
              Self: Div<T, Output = Self>,
              Self: Sized;
    fn testimages() -> TestReturnT<Self> where Self::Item: Clone + Copy + Zero + One + BaseFloat;
}

impl<T> Testimage<T> for DMat<T> {
    // here's where we associate Self::Item and T
    type Item = T;

    fn new_horizontal_line<I: Iterator<Item = usize>>(row: usize, cols: I) -> Self
        where Self::Item: Clone + Copy + Zero + One
    {
        let mut factor: DMat<Self::Item> = DMat::new_zeros(10, 10);
        for col in cols {
            *factor.index_mut((row, col)) = Self::Item::one();
        }
        factor
    }

    fn new_vertical_line<I: Iterator<Item = usize>>(rows: I, col: usize) -> Self
        where Self::Item: Clone + Copy + Zero + One
    {
        let mut factor: DMat<Self::Item> = DMat::new_zeros(10, 10);
        for row in rows {
            *factor.index_mut((row, col)) = Self::Item::one();
        }
        factor
    }

    fn luma01_to_image(&self) -> DynamicImage
        where Self::Item: Zero + One + FromPrimitive + ToPrimitive + PartialOrd + Float
    {
        let mut image_buffer = ImageBuffer::new(self.ncols() as u32, self.nrows() as u32);
        for (x, y, pixel) in image_buffer.enumerate_pixels_mut() {
            let value = self.index((y as usize, x as usize));
            assert!(Self::Item::zero() <= *value);
            assert!(*value <= Self::Item::one());
            let max = Self::Item::from_u8(std::u8::MAX).unwrap();
            let byte = (*value * max).round().to_u8().unwrap();
            *pixel = Luma([byte]);
        }
        DynamicImage::ImageLuma8(image_buffer)
    }

    fn save_luma01_to_png(&self, filename: &str) -> Result<(), ImageSaveError>
        where Self::Item: Zero + One + FromPrimitive + ToPrimitive + PartialOrd + Float
    {
        let image = self.luma01_to_image();
        let path = std::path::Path::new(filename);
        let mut file = try!(std::fs::File::create(&path).map_err(ImageSaveError::Io));
        image.save(&mut file, image::PNG).map_err(ImageSaveError::Image)
    }

    fn static_factors() -> StaticReturnT<Self>
        where Self::Item: Clone + Copy + Zero + One
    {
        fn helper_horizontal<U: Clone + Copy + Zero + One>(row: usize) -> DMat<U> {
            Testimage::new_horizontal_line(row, 0..10)
        }
        fn helper_vertical<U: Clone + Copy + Zero + One>(col: usize) -> DMat<U> {
            Testimage::new_vertical_line(0..10, col)
        }
        once(Testimage::new_horizontal_line(9, 0..5))
            .chain((0..9).rev().map(helper_horizontal as fn(usize) -> Self))
            .chain(once(Testimage::new_vertical_line(5..10, 0)))
            .chain((1..10).map(helper_vertical as fn(usize) -> Self))
    }

    fn horizontal_evolving_factors() -> HorizontalReturnT<Self>
        where Self::Item: Clone + Copy + Zero + One
    {
        fn helper<U: Clone + Copy + Zero + One>(start_col: usize) -> DMat<U> {
            Testimage::new_horizontal_line(9, start_col..(start_col + 5))
        }
        (0..6).map(helper)
    }

    fn vertical_evolving_factors() -> VerticalReturnT<Self>
        where Self::Item: Clone + Copy + Zero + One
    {
        fn helper<U: Clone + Copy + Zero + One>(start_row: usize) -> DMat<U> {
            Testimage::new_vertical_line(start_row..(start_row + 5), 0)
        }
        (0..6).rev().map(helper)
    }

    fn linear_combination<C>(factors: &[Self], coefficients: &[C]) -> Self
        where C: Clone,
              Self: Mul<C, Output = Self>,
              Self::Item: Clone + Copy + Zero
    {
        assert!(0 < factors.len());
        assert_eq!(factors.len(), coefficients.len());
        let nrows = factors[0].nrows();
        let ncols = factors[0].ncols();
        let mut result = DMat::new_zeros(nrows, ncols);
        for (factor, coefficient) in factors.iter().zip(coefficients.iter()) {
            assert_eq!(factor.nrows(), nrows);
            assert_eq!(factor.ncols(), ncols);
            result = result.add(factor.clone().mul(coefficient.clone()));
        }
        result
    }

    /// divide all values by max value
    // TODO put this into its own trait
    fn normalize_mut(self) -> Self
        where Self::Item: Clone + Copy + PartialOrd,
              Self: Div<T, Output = Self>,
              Self: Sized
    {
        let max = self.as_vec().iter().cloned().partial_max().unwrap();
        self.div(max)
    }

    fn testimages() -> TestReturnT<Self>
        where Self::Item: Clone + Copy + Zero + One + BaseFloat
    {
        fn helper<U, V, W, X>((a, b): (U, V)) -> W
            where U: Add<V, Output=W>,
                  W: Testimage<X>,
                  W::Item: Clone + BaseFloat,
                  W: Div<X, Output=W>
        {
            a.add(b).normalize_mut()
        }
        Testimage::horizontal_evolving_factors()
            .zip(Testimage::vertical_evolving_factors())
            .map(helper)

        //     horizontal
        //     .zip(vertical)
        //     .enumerate()
        //     .map(|(step, (horizontal, vertical))| {
        //         // let h: Self = horizontal;
        //         // // let cloned: Self = (*horizontal).clone::<Self>();
        //         // let foo = h.add(vertical);
        //         // let result: Self = foo.normalize::<Self>();
        //         // result
        //         horizontal.clone()
        //     }))
    }
}

// fn random_coefficient<R: Rng>(rng: &mut R) -> FloatType {
//     let Closed01(random) = rng.gen::<Closed01<FloatType>>();
//     random
// }
//
// fn random_coefficients<R: Rng>(rng: &mut R, count: usize) -> DVec<FloatType> {
//     let mut coefficients: DVec<FloatType> = DVec::new_zeros(count);
//     for i in 0..count {
//         coefficients[i] = random_coefficient(rng);
//     }
//     coefficients
// }
