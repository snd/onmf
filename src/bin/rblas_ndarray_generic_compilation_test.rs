extern crate num;
use num::{One, Zero};

extern crate rblas;
use rblas::attribute::Transpose;
use rblas::{Gemm};
// the matrix trait is in scope
use rblas::matrix::Matrix;

extern crate ndarray;
// this implements Matrix for BlasArrayViewMut
use ndarray::blas::{AsBlas, BlasArrayViewMut};
use ndarray::blas;
use ndarray::{ArrayBase, DataOwned, DataMut};

fn multiply_f64(
    a: &mut ArrayBase<Vec<f64>, (usize, usize)>,
    b: &mut ArrayBase<Vec<f64>, (usize, usize)>,
    c: &mut ArrayBase<Vec<f64>, (usize, usize)>,
)
{
    Gemm::gemm(
        &1.,
        Transpose::NoTrans, &a.blas(),
        Transpose::NoTrans, &b.blas(),
        &0.,
        &mut c.blas())
}

// fn multiply<FloatT, ContainerT, StorageT, Ix>(
//     a: &mut ContainerT,
//     b: &mut ContainerT,
//     c: &mut ContainerT
// )
//     where FloatT: One + Zero + Clone + Gemm,
//           ContainerT: AsBlas<FloatT, StorageT, (Ix, Ix)>,
//           StorageT: DataOwned<Elem=FloatT> + DataMut<Elem=FloatT>,
// {
//     let ablas: BlasArrayViewMut<FloatT, (Ix, Ix)> = a.blas();
//     let bblas: BlasArrayViewMut<FloatT, (Ix, Ix)> = b.blas();
//     let mut cblas: BlasArrayViewMut<FloatT, (Ix, Ix)> = c.blas();
//     Gemm::gemm(
//         &FloatT::one(),
//         Transpose::NoTrans, &ablas,
//         Transpose::NoTrans, &bblas,
//         &FloatT::zero(),
//         &mut cblas())
// }

fn main() {
    let mut a: ArrayBase<Vec<f64>, (usize, usize)> =
        ArrayBase::from_elem((10, 10), 2.);
    let mut b: ArrayBase<Vec<f64>, (usize, usize)> =
        ArrayBase::from_elem((10, 10), 2.);
    let mut c: ArrayBase<Vec<f64>, (usize, usize)> =
        ArrayBase::from_elem((10, 10), 2.);
    multiply_f64(&mut a, &mut b, &mut c);
    println!("{:?}", c);
}
