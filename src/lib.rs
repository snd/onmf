#[macro_use]
extern crate quick_error;
extern crate num;
extern crate nalgebra;
extern crate rand;
extern crate image;
extern crate ndarray;
extern crate rblas;

pub mod testimage_generator;

pub mod helpers;

mod online_nmf;
pub use online_nmf::OnlineNMF;

mod orthogonal_nmf;
pub use orthogonal_nmf::OrthogonalNMF;

mod orthogonal_nmf_blas;
pub use orthogonal_nmf_blas::OrthogonalNMFBlas;
