#![feature(test)]
extern crate test;

extern crate onmf;

extern crate nalgebra;
use self::nalgebra::{DMat};

extern crate rand;
use rand::{StdRng, SeedableRng};

extern crate num;
use num::{Float};

macro_rules! bench_ortho_nmf {
    ($bencher:expr, $float:ty, $nhidden:expr, $nobserved:expr, $nsamples:expr) => {{
        let seed: &[_] = &[1, 2, 3, 4];
        let mut rng: StdRng = SeedableRng::from_seed(seed);

        let mut ortho_nmf = onmf::OrthogonalNMF::<$float>::init_random01(
            $nhidden, $nobserved, $nsamples, &mut rng);

        let samples = DMat::<$float>::new_ones($nsamples, $nobserved);

        let alpha: $float = 0.1 * 1.01.powi(0);
        $bencher.iter(|| {
            ortho_nmf.iterate(alpha, &samples);
        });
    }}
}

#[bench]
fn bench_ortho_nmf_f64_10_256_11(bencher: &mut test::Bencher) {
    bench_ortho_nmf!(bencher, f64, 10, 256, 11);
}

#[bench]
fn bench_ortho_nmf_f32_10_256_11(bencher: &mut test::Bencher) {
    bench_ortho_nmf!(bencher, f32, 10, 256, 11);
}

#[bench]
fn bench_ortho_nmf_f64_10_512_11(bencher: &mut test::Bencher) {
    bench_ortho_nmf!(bencher, f64, 10, 512, 11);
}

#[bench]
fn bench_ortho_nmf_f32_10_512_11(bencher: &mut test::Bencher) {
    bench_ortho_nmf!(bencher, f32, 10, 512, 11);
}

#[bench]
fn bench_ortho_nmf_f64_20_512_30(bencher: &mut test::Bencher) {
    bench_ortho_nmf!(bencher, f64, 20, 512, 30);
}

#[bench]
fn bench_ortho_nmf_f32_20_512_30(bencher: &mut test::Bencher) {
    bench_ortho_nmf!(bencher, f32, 20, 512, 30);
}
