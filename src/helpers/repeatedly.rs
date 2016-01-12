extern crate rand;

pub struct Repeatedly<F> {
    pub f: F,
}

impl<T, F: FnMut() -> T> Iterator for Repeatedly<F> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        let mut f = &mut self.f;
        Some(f())
    }
}


/// takes a function `f` of no args, presumably with side effects, and
/// returns an iterator that endlessly calls `f` and yields what it returns
pub fn repeatedly<T, F: FnMut() -> T>(f: F) -> Repeatedly<F> {
    Repeatedly { f: f }
}

#[test]
fn test_repeatedly() {
    use self::rand::{StdRng, SeedableRng};
    use helpers::random01;

    assert_eq!(vec![3, 3, 3, 3],
               repeatedly(|| 3).take(4).collect::<Vec<usize>>());

    let mut i = 0;
    assert_eq!(vec![1, 2, 3, 4],
               repeatedly(|| {
                   i += 1;
                   i
               })
                   .take(4)
                   .collect::<Vec<usize>>());
    assert_eq!(i, 4);

    let seed: &[_] = &[1, 2, 3, 4];
    let mut rng: StdRng = SeedableRng::from_seed(seed);
    assert_eq!(vec![0.5162139860908154, 0.13628294371987984, 0.21635575241586105, 0.10006169673911681],
               repeatedly(|| random01(&mut rng))
                   .take(4)
                   .collect::<Vec<f64>>());
}
