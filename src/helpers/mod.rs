/*!
helper functions used by this crate
whose utility surpasses the confines of this crate.

generic things that are also useful in other crates.
*/

use ndarray::ArrayBase;

pub type Dims = (usize, usize);
pub type Array2D<T> = ArrayBase<Vec<T>, Dims>;

mod repeatedly;
pub use self::repeatedly::{Repeatedly, repeatedly};

// this needs to be public otherwise the impl is not exported
pub mod normalize;
pub use self::normalize::Normalize;

mod partial_max;
pub use self::partial_max::PartialMaxIteratorExt;

mod random;
pub use self::random::random01;

pub mod to_image;
pub use self::to_image::ToImage;

mod magnify;
pub use self::magnify::magnify;
