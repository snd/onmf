fn select_fold1<I,B, FProj, FCmp>(mut it: I,
                                  mut f_proj: FProj,
                                  mut f_cmp: FCmp) -> Option<(B, I::Item)>
    where I: Iterator,
          FProj: FnMut(&I::Item) -> B,
          FCmp: FnMut(&B, &I::Item, &B, &I::Item) -> bool
{
    // start with the first element as our selection. This avoids
    // having to use `Option`s inside the loop, translating to a
    // sizeable performance gain (6x in one case).
    it.next().map(|mut sel| {
        let mut sel_p = f_proj(&sel);

        for x in it {
            let x_p = f_proj(&x);
            if f_cmp(&sel_p,  &sel, &x_p, &x) {
                sel = x;
                sel_p = x_p;
            }
        }
        (sel_p, sel)
    })
}

/// the `:` here says that it's an extension trait
pub trait PartialMaxIteratorExt<T>: Iterator<Item=T> {
    fn partial_max(self) -> Option<T>;
}

/// blanket implementation for all iterators over partially ordered items
impl<
    T: PartialOrd,
    I: Iterator<Item=T> + Sized
> PartialMaxIteratorExt<T> for I {
    // TODO make this faster by using your own iterator instead of maps
    // and folds
    fn partial_max(self) -> Option<Self::Item>
        // where Self: Sized, Self::Item: PartialOrd
    {
        select_fold1(self,
                     |_| (),
                     // switch to y even if it is only equal, to preserve
                     // stability.
                     |_, x, _, y| x.lt(y))
            .map(|(_, x)| x)
    }
}
