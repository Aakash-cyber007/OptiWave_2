# OptiWave

## Overview :
This library use optimized SVD method to perform image compression.

## Features :
- Block SVD : Performs SVD using the Block Power Method.
- BPSO : Algorithm used to find optimal singular value for image compression.
- K SVD : An algorithm used to reduce the noise in an image.

## Upcoming features :
- A model which works with RGB images as well.
- A model which saves image to its accurate size more efficiently.

## Documentation :
For detailed information about OptiWave library, check out [Documentation]().

## Installation :
To install the library, use the following command on terminal:

```bash
pip install OptiWave
```

Please make sure, that OptiWave and image are in the same directory.

## Usage :
1. Block SVD :

```python
OptiWave.Block_SVD(Matrix,s)
```

- Block SVD function computes SVD of the input matrix 'Matrix'.
- 's' is an optional parameter, if the user wishes to compute the s'th approximation of the input matrix then 's' must be specified.

2. BPSO :

```python
OptiWave.svd_bpso(image,p_n,n_iterations)
```

- BPSO function computes the 'optimal-singular value' which is used to compress image.
- image: is a matrix which a numpy array obtained from the original image.
- p_n: It is an optional parameter, p_n need not be specified.
- n_iterations: It is an optional parameter, the user usually may not need to use it. Greater number of 'n_iterations' imply a more assured way of getting the ideal singular value which is used for image compression. Usually 3 to 4 iterations are enough to find the optimal singular value.

3. K SVD :

```python
OptiWave.ksvd(IMAGE : str, patch_size, stride, sparsity, max_iter)
```

- IMAGE: Image path.
- patch_size: Size of the square patch (pxp). It is an optional parameter.
- stride: Step size for moving the patch window. It is an optional parameter.
- sparsity: Desired sparsity level. It is an optional parameter.
- max_iter: Maxximum number of iterations. It is an optional parameter.

## Contributors :

- Here is the list of the Project Members, [Contributors]()
