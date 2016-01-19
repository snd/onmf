# understanding

## matrix

a matrix is like a spreadsheet.

`m x n` matrix has `m` rows and `n` columns.

when indexing or writing the form of a matrix **row index is written first, column index second**

rows are displayed vertically.
columns are displayed horizontally.
as you would expect from a spreadsheet.

`W` is a 5 x 3` matrix:
```
W = [[ 0  1 10]
     [11  5  0]
     [ 8  5  7]
     [11  1 20]
     [ 3  5  0]]
```

`H` is a `3 x 4` matrix:
```
H = [[0 1 1 6]
     [2 3 0 0]]
     [9 0 4 5]]
```

## matrix multiplication

```
W * H
```

**the dimensions that touch the `*` must match up**

`W.cols()` must be equal to `H.rows()` for matrix multiplication to be possible.
that is the shared dimension. it also gets eliminated.

**the output matrix has the form of the outside of `W * H`**

the output matrix has the form `W.rows() * H.cols()`


