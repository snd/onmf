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
/// returns an iterator that endlessly calls `f` and yields the return value
pub fn repeatedly<T, F: FnMut() -> T>(f: F) -> Repeatedly<F> {
    Repeatedly { f: f }
}

#[test]
fn test_repeatedly() {
    assert_eq!(vec![3, 3, 3, 3],
               repeatedly(|| 3).take(4).collect::<Vec<usize>>());

    let mut i = 0;
    assert_eq!(vec![1, 2, 3, 4],
               repeatedly(move || {
                   i += 1;
                   i
               })
                   .take(4)
                   .collect::<Vec<usize>>());
}
