# ONMF

*very early work in progress. API in flux.*

nothing to see here. move along.

## how to test this ?

reproducably (seeded) generate test data.

a series of pictures.
each picture is generated from a linear combination of factors.
the coefficients of the linear combination are chosen randomly.

two evolving factors change over time.
the two evolving factors change 6 times.
we generate 1000 pictures for each change of the evolving factors
during which the evolving factors stay constant.

the series of pictures is fed into ONMF one at a time.
a picture is fed as a 1D array (series of values, slice) into the ONMF
and not as a 2D array.


the reconstructed factors evolve. what does that mean ?
