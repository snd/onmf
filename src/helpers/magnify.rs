use nalgebra::DMat;

pub fn magnify<T>(matrix: DMat<T>, amount: usize) -> DMat<T>
    where T: Clone
{
    let mut result = unsafe {
        DMat::<T>::new_uninitialized(
            matrix.nrows() * amount, matrix.ncols() * amount)
    };
    for col in 0..result.ncols() {
        for row in 0..result.nrows() {
            result[(row, col)] = matrix[(row / amount, col / amount)].clone();
        }
    }
    result
}
