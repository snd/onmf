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

    let mut data: ArrayBase<Vec<f64>, (usize, usize)> =
        ArrayBase::zeros((nsamples, nobserved));

    // write one testimage into each row of data
    for (horizontal, vertical, i, factor) in testimage_generator::testimages::<f64, _>(per_step, &mut rng) {
        let row = (horizontal + vertical) * i;
        for (col, val) in factor.as_vec().iter().enumerate() {
            data[(row, col)] = val.clone();
        }
    }

    // let mut ortho_nmf = onmf::OrthogonalNMF::<f64>::init_random01(
    let mut ortho_nmf = onmf::OrthogonalNMFBlas::init_random01(
        nhidden, nobserved, nsamples, &mut rng);

    let mut iteration: i32 = 0;
    loop {
        // alpha gets larger and larger with each iteration
        // 0.1, 0.101, 0.102, 0.103, ...
        // at iteration 232 alpha first goes above 1.0
        // iteration = 232 -> alpha = 1.005
        let alpha = 0.1 * 1.01.powi(iteration);
        // TODO this could converge faster
        // let alpha = 0.1 * 1.0001.powi(iteration);

        // let alpha = 0.1;

        // TODO maybe try an even smaller alpha
        // let alpha = 0.01;


        ortho_nmf.iterate(alpha, &mut data);

        if iteration % 10 == 0 {
            println!("iteration = {} alpha = {}", iteration, alpha);

            // read testimage out of each row of nmf.hidden
            for irow in 0..nhidden {
                let mut column = Vec::<f64>::new();
                for icol in 0..nobserved {
                    column.push(ortho_nmf.hidden[(irow, icol)]);
                }
                let image = DMat::<f64>::from_col_vec(10, 10, &column[..]);
                // println!("image {}", irow);
                // println!("{:?}", image);
                magnify(image.normalize(), mag_factor).save_to_png(&format!("orthogonal-nmf-hidden-{}.png", irow)[..]).unwrap();
            }
        }

        iteration += 1;
    }
}
