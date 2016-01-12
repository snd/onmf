extern crate rand;
use self::rand::{Rng, Closed01, Rand};

/// returns a random number between `0.` (inclusive) and `1.` (inclusive)
pub fn random01<T, R>(rng: &mut R) -> T
    where Closed01<T>: Rand,
          R: Rng
{
    let Closed01(random) = rng.gen::<Closed01<T>>();
    random
}

#[test]
fn test_random01() {
    use self::rand::{StdRng, SeedableRng};

    let seed: &[_] = &[1, 2, 3, 4];
    let mut rng: StdRng = SeedableRng::from_seed(seed);
    let r: f64 = random01(&mut rng);
    assert_eq!(r, 0.5162139860908154);
}
