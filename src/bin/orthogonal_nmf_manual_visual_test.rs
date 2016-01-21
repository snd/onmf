extern crate onmf;
use onmf::helpers::{ToImage, Normalize, magnify};
use onmf::testimage_generator;

extern crate nalgebra;
use self::nalgebra::{DMat};

extern crate rand;
use rand::{StdRng, SeedableRng, ChaChaRng};

extern crate num;
use num::{Float};

extern crate ndarray;
use ndarray::ArrayBase;

type FloatT = f32;

fn main() {
    let mag_factor = 10;

    let seed: &[_] = &[1, 2, 3, 4];
    // let mut rng: ChaChaRng = ChaChaRng::from_seed(seed);
    let mut rng: StdRng = SeedableRng::from_seed(seed);

    let steps = 12;
    let per_step = 1000;
    let nsamples = steps * per_step;
    let nobserved = 10 * 10;
    // let nhidden = 20 + steps;
    let nhidden = 40;

    let mut data: ArrayBase<Vec<FloatT>, (usize, usize)> =
        ArrayBase::zeros((nsamples, nobserved));

    // write one testimage into each row of data
    for (horizontal, vertical, i, factor) in testimage_generator::testimages::<FloatT, _>(per_step, &mut rng) {
        let row = (horizontal + vertical) * i;
        for (col, val) in factor.as_vec().iter().enumerate() {
            data[(row, col)] = val.clone();
        }
    }

    let mut nmf = onmf::NMFBlas::new_random01(
        nhidden, nobserved, nsamples, &mut rng);

    let mut iteration: i32 = 0;
    loop {
        // alpha gets larger and larger with each iteration
        // 0.1, 0.101, 0.102, 0.103, ...
        // at iteration 232 alpha first goes above 1.0
        // iteration = 232 -> alpha = 1.005
        let alpha = 0.1 * 1.01.powi(iteration);

        // this does not converge:
        // let alpha = 0.1 * 1.0001.powi(iteration);

        nmf.iterate(&mut data, Some(alpha));

        if iteration % 10 == 0 {
            println!("iteration = {} alpha = {}", iteration, alpha);

            // read testimage out of each row of nmf.hidden
            for irow in 0..nhidden {
                let mut column = Vec::<FloatT>::new();
                for icol in 0..nobserved {
                    column.push(nmf.hidden[(irow, icol)]);
                }
                let image = DMat::<FloatT>::from_col_vec(10, 10, &column[..]);
                // println!("image {}", irow);
                // println!("{:?}", image);
                magnify(image.normalize(), mag_factor).save_to_png(&format!("orthogonal-nmf-hidden-{}.png", irow)[..]).unwrap();
            }
        }

        iteration += 1;
    }
}
