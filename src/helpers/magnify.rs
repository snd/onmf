use nalgebra::DMat;

pub fn magnify<T>(matrix: DMat<T>, amount: usize) -> DMat<T>
    where T: Clone
{
    let mut result = unsafe {
        DMat::<T>::new_uninitialized(
            matrix.nrows() * amount, matrix.ncols() * amount)
    };
    for row in 0..result.nrows() {
        // DMat is stored in colum major order.
        // in column major order consecutive elements of the columns
        // are contiguous.
        // loop through contiguous slices in the inner loop which
        // is faster due to caching.
        for col in 0..result.ncols() {
            result[(row, col)] = matrix[(row / amount, col / amount)].clone();
        }
    }
    result
}
