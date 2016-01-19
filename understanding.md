# intuitive understanding

## matrix

a matrix is like a spreadsheet.

`(m, n)` is the shape of a matrix that has `m` rows and `n` columns.

**row index is written first, column index second**

rows are displayed vertically.
columns are displayed horizontally.
as you would expect from a spreadsheet.

`W` is a `(5, 3)` matrix:
```
W = [[ 0  1 10]
     [11  5  0]
     [ 8  5  7]
     [11  1 20]
     [ 3  5  0]]
```

`H` is a `(3, 4)` matrix:
```
H = [[0 1 1 6]
     [2 3 0 0]]
     [9 0 4 5]]
```

the columns and rows of a matrix are vectors

## matrix multiplication

```
R = W * H
```

**the dimensions that touch the `*` must match up**

**the first matrix must be as wide as the second is high**

the first matrix must have as many columns as the second matrix has rows

**this ensures that a row in the first matrix is as long as a column in the second matrix**

`W.cols()` must be equal to `H.rows()` for matrix multiplication to be possible.
that is the shared dimension. it also gets eliminated.

**the output matrix has the shape of the outside of the shapes of `W` and `H`**

the output matrix has the shape `(W.rows(), H.cols())`

`R` is therefore a `(5, 4)` matrix

**R is build by taking the dot products of the rows of W and the columns of H (which have the same length)**

cell `R(i, j)` has the value of taking the dot product of row `i` of `W` and column `j` of `H`.

### dot product

the dot product of two vectors is a number

dot product = scalar product = inner product

**sum of the results of pairwise multiplication**

## intuition

a vector describes a direction in n-dimensional space

the rows of the 

when is the dot product the highest ?

when is the dot product the lowest ?


`R = W * H`:

**each row vector of `R` is the linear combination of the row vectors of `H` with the coefficients in the columns of `W`**

columns of `W` store coefficients for combining the rows of `H`
