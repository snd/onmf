extern crate onmf;
use onmf::helpers::{ToImage, magnify};
use onmf::testimage_generator;

extern crate rand;
use rand::{StdRng, SeedableRng};

fn main() {
    let mag_factor = 10;
    for (i, factor) in testimage_generator::static_factors::<f64>().enumerate() {
        magnify(factor, mag_factor).save_to_png(&format!("factor-{}.png", i)[..]).unwrap();
    }

    for (i, factor) in testimage_generator::horizontal_evolving_factors::<f64>().enumerate() {
        magnify(factor, mag_factor).save_to_png(&format!("horizontal-{}.png", i)[..]).unwrap();
    }

    for (i, factor) in testimage_generator::vertical_evolving_factors::<f64>().enumerate() {
        magnify(factor, mag_factor).save_to_png(&format!("vertical-{}.png", i)[..]).unwrap();
    }

    let seed: &[_] = &[1, 2, 3, 4];
    let mut rng: StdRng = SeedableRng::from_seed(seed);

    let per_step = 10;
    for (horizontal, vertical, i, factor) in testimage_generator::testimages::<f64, _>(per_step, &mut rng) {
        magnify(factor, mag_factor).save_to_png(&format!("test-{}-{}-{}.png", horizontal, vertical, i)[..]).unwrap();
    }
}
