#[macro_use]
extern crate quick_error;

pub mod testimage_generator;

pub mod helpers;

mod online_nmf;
pub use online_nmf::OnlineNMF;

mod orthogonal_nmf;
pub use orthogonal_nmf::OrthogonalNMF;
