#[macro_use]
extern crate quick_error;
extern crate num;
extern crate nalgebra;
extern crate rand;
extern crate image;
extern crate ndarray;

pub mod testimage_generator;

pub mod helpers;

mod online_nmf;
pub use online_nmf::OnlineNMF;

mod orthogonal_nmf;
pub use orthogonal_nmf::OrthogonalNMF;
