use std;

use std::cmp::PartialOrd;

extern crate num;
use self::num::traits::{Zero, One, FromPrimitive, ToPrimitive, Float};

extern crate nalgebra;
use self::nalgebra::{DMat};

extern crate image;
use self::image::{ImageBuffer, Luma, DynamicImage};

quick_error! {
    #[derive(Debug)]
    pub enum ImageSaveError {
        /// an error has occured when opening the file
        Io(err: std::io::Error) {}
        /// an error has occured when writing the image into the file
        Image(err: image::ImageError) {}
    }
}

/// make images from things containing brightness information
pub trait ToImage {
    /// # Panics
    /// panics unless all the values in `self` are between
    /// `0` (inclusive) and `1` (inclusive)
    fn to_image(&self) -> DynamicImage;
    /// # Panics
    /// panics unless all the values in `self` are between
    /// `0` (inclusive) and `1` (inclusive)
    fn save_to_png(&self, filename: &str) -> Result<(), ImageSaveError>;
}

impl<T: Zero + One + FromPrimitive + ToPrimitive + PartialOrd + Float> ToImage for DMat<T> {
    fn to_image(&self) -> DynamicImage {
        let mut image_buffer = ImageBuffer::new(self.ncols() as u32, self.nrows() as u32);
        for (x, y, pixel) in image_buffer.enumerate_pixels_mut() {
            let value = self[(y as usize, x as usize)];
            assert!(value.is_nan());
            assert!(T::zero() <= value);
            assert!(value <= T::one());
            let max = T::from_u8(std::u8::MAX).unwrap();
            let byte = (value * max).round().to_u8().unwrap();
            *pixel = Luma([byte]);
        }
        DynamicImage::ImageLuma8(image_buffer)
    }

    fn save_to_png(&self, filename: &str) -> Result<(), ImageSaveError> {
        let image = self.to_image();
        let path = std::path::Path::new(filename);
        let mut file = try!(std::fs::File::create(&path).map_err(ImageSaveError::Io));
        image.save(&mut file, image::PNG).map_err(ImageSaveError::Image)
    }
}
