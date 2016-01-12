/*!
helper functions used by this crate
whose utility surpasses the confines of this crate.

generic things that are also useful in other crates.
*/

mod repeatedly;
pub use self::repeatedly::{Repeatedly, repeatedly};

mod normalize;
pub use self::normalize::Normalize;

mod partialmax;
pub use self::partialmax::PartialMaxIteratorExt;

mod random;
pub use self::random::random01;
