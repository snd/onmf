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

### a matrix is a linear function

## matrix multiplication

don't be afraid of matrix multiplication. it's so immensely useful.

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

**the dot product is the sum of the results of pairwise multiplication**

the dot product is also the product of the lengths of both vectors and the cosine of the angle between them

**`cos(0) = 1` which means that the the dot product is greatest when the angle between the vectors is `0`,
when they point in the same direction.**

[its a form of directional multiplication](http://betterexplained.com/articles/vector-calculus-understanding-the-dot-product/)

- zero: we don’t have any growth in the original direction
- positive number: we have some growth in the original direction
- negative number: we have negative (reverse) growth in the original direction

[i like the mario-kart speed boost analogy here ; )](http://betterexplained.com/articles/vector-calculus-understanding-the-dot-product/)

- if you come in going 0, you’ll get nothing [if you are just dropped onto the pad, there’s no boost]
- if you cross the pad perpendicularly, you’ll get 0 [just like the banana obliteration, it will give you 0x boost in the perpendicular direction]

“along the path”. how much energy is actually going in our original direction?

- it just boosts the existing velocity
- only the part of the velocity going into the same direction as the other

**only multiplies the parts that go into the same direction**

if the vectors point in opposite directions then the dot product is `0`

unit vector ?

when is the dot product the highest ?

when is the dot product the lowest ?

what role does the length play ?

the dot product also increases if the length of the vectors increases

higher if the numbers are high

### back to the matrix multiplication

a vector describes a direction in n-dimensional space

a higher value in cell `R(i,j)` means that row `i` of `W` goes into the same direction as col `j` of `H`

### matrix multiplication can also be thought of as function composition

## intuition

the rows of the 



`R = W * H`:

**each row vector of `R` is the linear combination of the row vectors of `H` with the coefficients in the columns of `W`**

columns of `W` store coefficients for combining the rows of `H`


## applied to the parts

### `weights_multiplier <- samples * hidden.transposed()`

`samples.shape() = (nsamples, nobserved)`

`hidden.shape() = (nhidden, nobserved)`

`hidden.transposed().shape() = (nobserved, nhidden)`

`weights_multiplier.shape() = weigths.shape() = (nsamples, nhidden)

**the transpose is only here so we can actually multiply `samples` with `hidden`**

otherwise their shapes are incompatible.
in fact this is the only way to make them compatible !

so we take the dot products of the rows of `samples`
and the cols of `hidden.transposed()` (meaning the rows of `hidden`)

the rows of `samples` encode the observed vectors.

the rows of `hidden` encode the 

**THIS: we can think of `weights_multiplier` as reinforcing those weights where the `hidden` variable
vectors are correctly aligned with the observed data.
weights are reinforced depending on how correctly the `hidden` variable vectors model the observed data.**
