extern crate onmf;
use onmf::testimage::Testimage;

extern crate nalgebra;
use nalgebra::{DMat};

extern crate rand;
use rand::{StdRng, Rng, SeedableRng, Closed01};

fn main() {
    let static_factors: onmf::testimage::StaticReturnT<DMat<f64>> =
        Testimage::static_factors();
    for (i, factor) in static_factors.enumerate() {
        factor.save_luma01_to_png(&format!("factor-{}.png", i)[..]).unwrap();
    }

    let horizontal_evolving_factors: onmf::testimage::HorizontalReturnT<DMat<f64>> =
        Testimage::horizontal_evolving_factors();
    for (i, factor) in horizontal_evolving_factors.enumerate() {
        factor.save_luma01_to_png(&format!("horizontal-{}.png", i)[..]).unwrap();
    }

    let vertical_evolving_factors: onmf::testimage::VerticalReturnT<DMat<f64>> =
        Testimage::vertical_evolving_factors();
    for (i, factor) in vertical_evolving_factors.enumerate() {
        factor.save_luma01_to_png(&format!("vertical-{}.png", i)[..]).unwrap();
    }

    let testimages: onmf::testimage::TestReturnT<DMat<f64>> =
        Testimage::testimages();
    for (i, factor) in testimages.enumerate() {
        factor.save_luma01_to_png(&format!("test-{}.png", i)[..]).unwrap();
    }

    // let seed: &[_] = &[1, 2, 3, 4];
    // let mut rng: StdRng = SeedableRng::from_seed(seed);
    //
    // for i in 0..10 {
    //     let combination = random_linear_combination(&factors[..], &mut rng);
    //     save_as_png(&combination, &format!("combination-{}.png", i)[..]).unwrap();
    // }

//     let iter = horizontal_evolving.iter()
//         .zip(vertical_evolving.iter()).enumerate();
//     for (step, (horizontal, vertical)) in iter {
//         for i in 0..10 {
//             // static factors
//             let random_coefficients = random_coefficients(&mut rng, factors.len()).div(4.);
//             let result = linear_combination(&factors[..], random_coefficients.as_slice());
//
//             // evolving factors
//             let horizontal_coefficient = random_coefficient(&mut rng) / 4.;
//             let result = result.add(
//                 horizontal.clone().mul(horizontal_coefficient));
//
//             let vertical_coefficient = random_coefficient(&mut rng) / 4.;
//             let result = result.add(
//                 vertical.clone().mul(vertical_coefficient));
//
//             save_as_png(&result, &format!("test-{}-{}.png", step, i)[..]).unwrap();
//         }
//     }
}
