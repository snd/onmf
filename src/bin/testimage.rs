use std::ops::{Index, IndexMut, Mul, Add};

#[macro_use]
extern crate quick_error;

extern crate nalgebra;
use nalgebra::{DMat};

extern crate image;
use image::{ImageBuffer, Luma, DynamicImage};

extern crate rand;
use rand::{StdRng, Rng, SeedableRng, Closed01};

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

fn mat_to_image(mat: &Mat) -> DynamicImage {
    let mut image_buffer
        = ImageBuffer::new(mat.ncols() as u32, mat.nrows() as u32);
    for (x, y, pixel) in image_buffer.enumerate_pixels_mut() {
        let value = mat.index((y as usize, x as usize));
        let byte = (*value * 255.).round() as u8;
        *pixel = Luma([byte]);
    }
    DynamicImage::ImageLuma8(image_buffer)
}

quick_error! {
    #[derive(Debug)]
    pub enum ImageSaveError {
        Io(err: std::io::Error) {}
        Image(err: image::ImageError) {}
    }
}

fn save_as_png(mat: &Mat, filename: &str) -> Result<(), ImageSaveError> {
    let image = mat_to_image(&mat);
    let path = std::path::Path::new(filename);
    let mut file = try!(std::fs::File::create(&path)
                        .map_err(ImageSaveError::Io));
    image.save(&mut file, image::PNG).map_err(ImageSaveError::Image)
}

fn linear_combination(factors: &[Mat], coefficients: &[FloatType]) -> Mat {
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
    // result.div(factors.len() as FloatType)
    result
}

fn random_linear_combination<R: Rng>(factors: &[Mat], rng: &mut R) -> Mat {
    let mut coefficients: Vec<f64> = Vec::new();
    for _ in 0..factors.len() {
        let Closed01(random) = rng.gen::<Closed01<f64>>();
        coefficients.push(random / 2.);
    }
    linear_combination(factors, &coefficients[..])
}

fn main() {
    let mut factors: Vec<Mat> = Vec::new();

    factors.push(horizontal_line(9, 0..5));
    for row in (0..9).rev() {
        factors.push(horizontal_line(row, 0..10));
    }

    factors.push(vertical_line(5..10, 0));
    for col in 1..10 {
        factors.push(vertical_line(0..10, col));
    }

    println!("factors:\n");
    for (i, factor) in factors.iter().enumerate() {
        save_as_png(&factor, &format!("factor-{}.png", i)[..]).unwrap();
        println!("{:?}", factor);
    }

    let mut horizontal_evolving: Vec<Mat> = Vec::new();
    for start_col in 0..6 {
        horizontal_evolving.push(horizontal_line(9, start_col..(start_col + 5)));
    }

    println!("short horizontal bar that moves from left to right:\n");
    for (i, factor) in horizontal_evolving.iter().enumerate() {
        save_as_png(&factor, &format!("horizontal-{}.png", i)[..]).unwrap();
        println!("{:?}", factor);
    }

    let mut vertical_evolving: Vec<Mat> = Vec::new();
    for start_row in (0..6).rev() {
        vertical_evolving.push(vertical_line(start_row..(start_row + 5), 0));
    }

    println!("short vertical bar that moves from bottom to top:\n");
    for (i, factor) in vertical_evolving.iter().enumerate() {
        save_as_png(&factor, &format!("vertical-{}.png", i)[..]).unwrap();
        println!("{:?}", factor);
    }

    let seed: &[_] = &[1, 2, 3, 4];
    let mut rng: StdRng = SeedableRng::from_seed(seed);

    for i in 0..10 {
        let combination = random_linear_combination(&factors[..], &mut rng);
        save_as_png(&combination, &format!("combination-{}.png", i)[..]).unwrap();
    }
}
