# 0x05. Advanced Linear Algebra

## Resources

**Read or watch**:

*   [The determinant | Essence of linear algebra](https://www.youtube.com/watch?v=Ip3X9LOh2dk&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab&index=7&t=0s&ab_channel=3Blue1Brown)
*   [Determinant of a Matrix](https://www.mathsisfun.com/algebra/matrix-determinant.html)
*   [Determinant](https://mathworld.wolfram.com/Determinant.html)
*   [Determinant of an empty matrix](https://www.quora.com/What-is-the-determinant-of-an-empty-matrix-such-as-a-0x0-matrix)
*   [Inverse matrices, column space and null space](https://www.youtube.com/watch?v=uQhTuRlWMxw&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab&index=8&t=0s&ab_channel=3Blue1Brown)
*   [Inverse of a Matrix using Minors, Cofactors and Adjugate](https://www.mathsisfun.com/algebra/matrix-inverse-minors-cofactors-adjugate.html)
*   [Minor](https://mathworld.wolfram.com/Minor.html)
*   [Cofactor](https://mathworld.wolfram.com/Cofactor.html)
*   [Adjugate matrix](https://en.wikipedia.org/wiki/Adjugate_matrix)
*   [Singular Matrix](https://mathworld.wolfram.com/SingularMatrix.html)
*   [Elementary Matrix Operations](https://stattrek.com/matrix-algebra/elementary-operations.aspx)
*   [Gaussian Elimination](https://mathworld.wolfram.com/GaussianElimination.html)
*   [Gauss-Jordan Elimination](https://mathworld.wolfram.com/Gauss-JordanElimination.html)
*   [Matrix Inverse](https://mathworld.wolfram.com/MatrixInverse.html)
*   [Eigenvectors and eigenvalues | Essence of linear algebra](https://www.youtube.com/watch?v=PFDu9oVAE-g&ab_channel=3Blue1Brown)
*   [Eigenvalues and eigenvectors](https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors)
*   [Eigenvalues and Eigenvectors](https://math.mit.edu/~gs/linearalgebra/ila0601.pdf)
*   [Definiteness of a matrix](https://en.wikipedia.org/wiki/Definite_symmetric_matrix) **Up to Eigenvalues**
*   [Definite, Semi-Definite and Indefinite Matrices](http://mathonline.wikidot.com/definite-semi-definite-and-indefinite-matrices) **Ignore Hessian Matrices**
*   [Tests for Positive Definiteness of a Matrix](https://www.gaussianwaves.com/2013/04/tests-for-positive-definiteness-of-a-matrix/)
*   [Positive Definite Matrices and Minima](https://www.youtube.com/watch?v=tccVVUnLdbc&ab_channel=MITOpenCourseWare)
*   [Positive Definite Matrices](https://www.math.utah.edu/~zwick/Classes/Fall2012_2270/Lectures/Lecture33_with_Examples.pdf)

**As references**:

*   [numpy.linalg.eig](https://numpy.org/doc/stable/reference/generated/numpy.linalg.eig.html)

## Learning Objectives


### General

*   What is a determinant? How would you calculate it?
*   What is a minor, cofactor, adjugate? How would calculate them?
*   What is an inverse? How would you calculate it?
*   What are eigenvalues and eigenvectors? How would you calculate them?
*   What is definiteness of a matrix? How would you determine a matrix’s definiteness?

## Requirements

### General

*   Allowed editors: `vi`, `vim`, `emacs`
*   All your files will be interpreted/compiled on Ubuntu 16.04 LTS using `python3` (version 3.5)
*   Your files will be executed with `numpy` (version 1.15)
*   All your files should end with a new line
*   The first line of all your files should be exactly `#!/usr/bin/env python3`
*   A `README.md` file, at the root of the folder of the project, is mandatory
*   Your code should use the `pycodestyle` style (version 2.5)
*   All your modules should have documentation (`python3 -c 'print(__import__("my_module").__doc__)'`)
*   All your classes should have documentation (`python3 -c 'print(__import__("my_module").MyClass.__doc__)'`)
*   All your functions (inside and outside a class) should have documentation (`python3 -c 'print(__import__("my_module").my_function.__doc__)'` and `python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)'`)
*   Unless otherwise noted, you are not allowed to import any module
*   All your files must be executable
*   The length of your files will be tested using `wc`

* * *

## Tasks
<details>
<summary>View Contents</summary>

#### 0\. Determinant <span class="alert alert-warning mandatory-optional">mandatory</span>

Write a function `def determinant(matrix):` that calculates the determinant of a matrix:

*   `matrix` is a list of lists whose determinant should be calculated
*   If `matrix` is not a list of lists, raise a `TypeError` with the message `matrix must be a list of lists`
*   If `matrix` is not square, raise a `ValueError` with the message `matrix must be a square matrix`
*   The list `[[]]` represents a `0x0` matrix
*   Returns: the determinant of `matrix`

    
```
    alexa@ubuntu-xenial:0x05-advanced_linear_algebra$ ./0-main.py 
    1
    5
    -2
    0
    192
    matrix must be a list of lists
    matrix must be a square matrix
    alexa@ubuntu-xenial:0x05-advanced_linear_algebra$
```
**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `math/0x05-advanced_linear_algebra`
*   File: [`0-determinant.py`](./0-determinant.py)

#### 1\. Minor <span class="alert alert-warning mandatory-optional">mandatory</span>

Write a function `def minor(matrix):` that calculates the minor matrix of a matrix:

*   `matrix` is a list of lists whose minor matrix should be calculated
*   If `matrix` is not a list of lists, raise a `TypeError` with the message `matrix must be a list of lists`
*   If `matrix` is not square or is empty, raise a `ValueError` with the message `matrix must be a non-empty square matrix`
*   Returns: the minor matrix of `matrix`
    
```
    alexa@ubuntu-xenial:0x05-advanced_linear_algebra$ ./1-main.py 
    [[1]]
    [[4, 3], [2, 1]]
    [[1, 1], [1, 1]]
    [[-12, -36, 0], [10, -34, -32], [47, 13, -16]]
    matrix must be a list of lists
    matrix must be a non-empty square matrix
    alexa@ubuntu-xenial:0x05-advanced_linear_algebra$
```
**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `math/0x05-advanced_linear_algebra`
*   File: [`1-minor.py`](./1-minor.py)

#### 2\. Cofactor <span class="alert alert-warning mandatory-optional">mandatory</span>

Write a function `def cofactor(matrix):` that calculates the cofactor matrix of a matrix:

*   `matrix` is a list of lists whose cofactor matrix should be calculated
*   If `matrix` is not a list of lists, raise a `TypeError` with the message `matrix must be a list of lists`
*   If `matrix` is not square or is empty, raise a `ValueError` with the message `matrix must be a non-empty square matrix`
*   Returns: the cofactor matrix of `matrix`

```
    alexa@ubuntu-xenial:0x05-advanced_linear_algebra$ ./2-main.py 
    [[1]]
    [[4, -3], [-2, 1]]
    [[1, -1], [-1, 1]]
    [[-12, 36, 0], [-10, -34, 32], [47, -13, -16]]
    matrix must be a list of lists
    matrix must be a non-empty square matrix
    alexa@ubuntu-xenial:0x05-advanced_linear_algebra$
```
**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `math/0x05-advanced_linear_algebra`
*   File: [`2-cofactor.py`](./2-cofactor.py)

#### 3\. Adjugate <span class="alert alert-warning mandatory-optional">mandatory</span>

Write a function `def adjugate(matrix):` that calculates the adjugate matrix of a matrix:

*   `matrix` is a list of lists whose adjugate matrix should be calculated
*   If `matrix` is not a list of lists, raise a `TypeError` with the message `matrix must be a list of lists`
*   If `matrix` is not square or is empty, raise a `ValueError` with the message `matrix must be a non-empty square matrix`
*   Returns: the adjugate matrix of `matrix`

```   
    alexa@ubuntu-xenial:0x05-advanced_linear_algebra$ ./3-main.py 
    [[1]]
    [[4, -2], [-3, 1]]
    [[1, -1], [-1, 1]]
    [[-12, -10, 47], [36, -34, -13], [0, 32, -16]]
    matrix must be a list of lists
    matrix must be a non-empty square matrix
    alexa@ubuntu-xenial:0x05-advanced_linear_algebra$
```
**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `math/0x05-advanced_linear_algebra`
*   File: [`3-adjugate.py`](./3-adjugate.py)

#### 4\. Inverse <span class="alert alert-warning mandatory-optional">mandatory</span>

Write a function `def inverse(matrix):` that calculates the inverse of a matrix:

*   `matrix` is a list of lists whose inverse should be calculated
*   If `matrix` is not a list of lists, raise a `TypeError` with the message `matrix must be a list of lists`
*   If `matrix` is not square or is empty, raise a `ValueError` with the message `matrix must be a non-empty square matrix`
*   Returns: the inverse of `matrix`, or `None` if `matrix` is singular

``` 
    alexa@ubuntu-xenial:0x05-advanced_linear_algebra$ ./4-main.py 
    [[0.2]]
    [[-2.0, 1.0], [1.5, -0.5]]
    None
    [[-0.0625, -0.052083333333333336, 0.24479166666666666], [0.1875, -0.17708333333333334, -0.06770833333333333], [0.0, 0.16666666666666666, -0.08333333333333333]]
    matrix must be a list of lists
    matrix must be a non-empty square matrix
    alexa@ubuntu-xenial:0x05-advanced_linear_algebra$
```
**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `math/0x05-advanced_linear_algebra`
*   File: [`4-inverse.py`](./4-inverse.py)


#### 5\. Definiteness <span class="alert alert-warning mandatory-optional">mandatory</span>

Write a function `def definiteness(matrix):` that calculates the definiteness of a matrix:

*   `matrix` is a `numpy.ndarray` of shape `(n, n)` whose definiteness should be calculated
*   If `matrix` is not a `numpy.ndarray`, raise a `TypeError` with the message `matrix must be a numpy.ndarray`
*   If `matrix` is not a valid matrix, return `None`
*   Return: the string `Positive definite`, `Positive semi-definite`, `Negative semi-definite`, `Negative definite`, or `Indefinite` if the matrix is positive definite, positive semi-definite, negative semi-definite, negative definite of indefinite, respectively
*   If `matrix` does not fit any of the above categories, return `None`
*   You may `import numpy as np`
```
    alexa@ubuntu-xenial:0x05-advanced_linear_algebra$ ./5-main.py 
    Positive definite
    Positive semi-definite
    Negative semi-definite
    Negative definite
    Indefinite
    None
    None
    matrix must be a numpy.ndarray
    alexa@ubuntu-xenial:0x05-advanced_linear_algebra$
```
**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `math/0x05-advanced_linear_algebra`
*   File: [`5-definiteness.py`](./5-definiteness.py)

</details>
