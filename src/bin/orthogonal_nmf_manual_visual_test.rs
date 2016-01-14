extern crate onmf;
use onmf::helpers::{ToImage, Normalize};
use onmf::testimage_generator;
use onmf::testimage_generator::{magnify};

extern crate nalgebra;
use self::nalgebra::{DMat};

extern crate rand;
use rand::{StdRng, SeedableRng};

fn main() {
    let mag_factor = 10;

    let seed: &[_] = &[1, 2, 3, 4];
    let mut rng: StdRng = SeedableRng::from_seed(seed);

    let steps = 6;
    let per_step = 10;
    let nsamples = steps * per_step;
    let nobserved = 10 * 10;
    let nhidden = 10;
    let niterations = 2;

    let mut data = DMat::<f64>::new_zeros(nsamples, nobserved);

    // write on testimage into each row of data
    for (step, i, factor) in testimage_generator::testimages::<f64, _>(per_step, &mut rng) {
        let row = step * i;
        for (col, val) in factor.as_vec().iter().enumerate() {
            data[(row, col)] = val.clone();
        }
    }

    let mut nmf = onmf::OrthogonalNMF::<f64>::init_randomly(
        nhidden, nobserved, nsamples);

    for _ in 0..niterations { nmf.iterate(&data) }

    // read testimage out of each row of nmf.hidden
    for irow in 0..nhidden {
        let mut column = Vec::<f64>::new();
        for icol in 0..nobserved {
            column.push(nmf.hidden[(irow, icol)]);
        }
        let image = DMat::<f64>::from_col_vec(10, 10, &column[..]);
        // println!("{:?}", image);
        magnify(image.normalize(), mag_factor).save_to_png(&format!("orthogonal-nmf-hidden-{}.png", irow)[..]).unwrap();

    }
}
