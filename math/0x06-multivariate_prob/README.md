# 0x06. Multivariate Probability

## Learning Objectives

### General

-   Who is Carl Friedrich Gauss?
-   What is a joint/multivariate distribution?
-   What is a covariance?
-   What is a correlation coefficient?
-   What is a covariance matrix?
-   What is a multivariate gaussian distribution?

## Requirements

### General

-   Allowed editors: `vi`, `vim`, `emacs`
-   All your files will be interpreted/compiled on Ubuntu 16.04 LTS using `python3` (version 3.5)
-   Your files will be executed with `numpy` (version 1.15)
-   All your files should end with a new line
-   The first line of all your files should be exactly `#!/usr/bin/env python3`
-   A `README.md` file, at the root of the folder of the project, is mandatory
-   Your code should use the `pycodestyle` style (version 2.5)
-   All your modules should have documentation (`python3 -c 'print(__import__("my_module").__doc__)'`)
-   All your classes should have documentation (`python3 -c 'print(__import__("my_module").MyClass.__doc__)'`)
-   All your functions (inside and outside a class) should have documentation (`python3 -c 'print(__import__("my_module").my_function.__doc__)'` and `python3 -c 'print(__import__("my_module").MyClass.my_function.__doc__)'`)
-   Unless otherwise noted, you are not allowed to import any module except `import numpy as np`
-   All your files must be executable
-   The length of your files will be tested using `wc`

## Resources

**Read or watch**:

- [Joint Probability Distributions](http://homepage.stat.uiowa.edu/~rdecook/stat2020/notes/ch5_pt1.pdf)
- [Multivariate Gaussian distributions](https://www.youtube.com/watch?v=eho8xH3E6mE&ab_channel=AlexanderIhler)
- [The Multivariate Gaussian Distribution](http://cs229.stanford.edu/section/gaussians.pdf)
- [An Introduction to Variance, Covariance & Correlation](https://www.alchemer.com/resources/blog/variance-covariance-correlation/)
- [Variance-covariance matrix using matrix notation of factor analysis](https://www.youtube.com/watch?v=G16c2ZODcg8&ab_channel=BenLambert)

**Definitions to skim:**

- [Carl Friedrich Gauss](https://en.wikipedia.org/wiki/Carl_Friedrich_Gauss)
- [Joint probability distribution](https://en.wikipedia.org/wiki/Joint_probability_distribution)
- [Covariance](https://en.wikipedia.org/wiki/Covariance)
- [Covariance matrix](https://en.wikipedia.org/wiki/Covariance_matrix)

**As references:**

- [numpy.cov](https://numpy.org/doc/stable/reference/generated/numpy.cov.html)
- [numpy.corrcoef](https://numpy.org/doc/stable/reference/generated/numpy.corrcoef.html)
- [numpy.linalg.det](https://numpy.org/doc/stable/reference/generated/numpy.linalg.det.html)
- [numpy.linalg.inv](https://numpy.org/doc/stable/reference/generated/numpy.linalg.inv.html)
- [numpy.random.multivariate_normal](https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.multivariate_normal.html)

---

## Quiz questions

<details>
<summary>Show</summary>

### Question #0

p<sub>x, y</sub>(x, y) =

- [ ] P(X = x)P(Y = y)

- [ ] P(X = x | Y = y)

- [x] P(X = x | Y = y)P(Y = y)

- [ ] P(Y = y | X = x)

- [x] P(Y = y | X = x)P(X = x)

- [x] P(X = x ??? Y = y)

- [ ] P(X = x ??? Y = y)

### Question #1

The `i,j`<sup>th</sup> entry in the covariance matrix is

- [ ] the variance of variable `i` plus the variance of variable `j`

- [x] the variance of `i` if `i == j`

- [x] the same as the `j,i`<sup>th</sup> entry

- [x] the variance of variable `i` and variable `j`


### Question #2

The correlation coefficient of the variables X and Y is defined as:

- [ ] ?? = ??<sub>XY</sub><sup>2</sup>

- [ ] ?? = ??<sub>XY</sub>

- [x] ?? = ??<sub>XY</sub> / ( ??<sub>X</sub> ??<sub>Y</sub> )

- [ ] ?? = ??<sub>XY</sub> / ( ??<sub>XX</sub> ??<sub>YY</sub> )

</details>

---

## Tasks

<details>
<summary>View Contents</summary>

#### 0\. Mean and Covariance <span class="alert alert-warning mandatory-optional">mandatory</span>

Write a function `def mean_cov(X):` that calculates the mean and covariance of a data set:

*   `X` is a `numpy.ndarray` of shape `(n, d)` containing the data set:
    *   `n` is the number of data points
    *   `d` is the number of dimensions in each data point
    *   If `X` is not a 2D `numpy.ndarray`, raise a `TypeError` with the message `X must be a 2D numpy.ndarray`
    *   If `n` is less than 2, raise a `ValueError` with the message `X must contain multiple data points`
*   Returns: `mean`, `cov`:
    *   `mean` is a `numpy.ndarray` of shape `(1, d)` containing the mean of the data set
    *   `cov` is a `numpy.ndarray` of shape `(d, d)` containing the covariance matrix of the data set
*   You are not allowed to use the function `numpy.cov`
``` 
    alexa@ubuntu-xenial:0x06-multivariate_prob$ ./0-main.py 
    [[12.04341828 29.92870885 10.00515808]]
    [[ 36.2007391  -29.79405239  15.37992641]
     [-29.79405239  97.77730626 -20.67970134]
     [ 15.37992641 -20.67970134  24.93956823]]
    alexa@ubuntu-xenial:0x06-multivariate_prob$
```
**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `math/0x06-multivariate_prob`
*   File: [`0-mean_cov.py`](./0-mean_cov.py)


#### 1\. Correlation <span class="alert alert-warning mandatory-optional">mandatory</span>

Write a function `def correlation(C):` that calculates a correlation matrix:

*   `C` is a `numpy.ndarray` of shape `(d, d)` containing a covariance matrix
    *   `d` is the number of dimensions
    *   If `C` is not a `numpy.ndarray`, raise a `TypeError` with the message `C must be a numpy.ndarray`
    *   If `C` does not have shape `(d, d)`, raise a `ValueError` with the message `C must be a 2D square matrix`
*   Returns a `numpy.ndarray` of shape `(d, d)` containing the correlation matrix

```  
    alexa@ubuntu-xenial:0x06-multivariate_prob$ ./1-main.py 
    [[ 36 -30  15]
     [-30 100 -20]
     [ 15 -20  25]]
    [[ 1\.  -0.5  0.5]
     [-0.5  1\.  -0.4]
     [ 0.5 -0.4  1\. ]]
    alexa@ubuntu-xenial:0x06-multivariate_prob$
```
**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `math/0x06-multivariate_prob`
*   File: [`1-correlation.py`](./1-correlation.py)


#### 2\. Initialize 

Create the class `MultiNormal` that represents a Multivariate Normal distribution:

*   class constructor `def __init__(self, data):`
    *   `data` is a `numpy.ndarray` of shape `(d, n)` containing the data set:
    *   `n` is the number of data points
    *   `d` is the number of dimensions in each data point
    *   If `data` is not a 2D `numpy.ndarray`, raise a `TypeError` with the message `data must be a 2D numpy.ndarray`
    *   If `n` is less than 2, raise a `ValueError` with the message `data must contain multiple data points`
*   Set the public instance variables:
    *   `mean` - a `numpy.ndarray` of shape `(d, 1)` containing the mean of `data`
    *   `cov` - a `numpy.ndarray` of shape `(d, d)` containing the covariance matrix `data`
*   You are not allowed to use the function `numpy.cov`

```    
    alexa@ubuntu-xenial:0x06-multivariate_prob$ ./2-main.py 
    [[12.04341828]
     [29.92870885]
     [10.00515808]]
    [[ 36.2007391  -29.79405239  15.37992641]
     [-29.79405239  97.77730626 -20.67970134]
     [ 15.37992641 -20.67970134  24.93956823]]
    alexa@ubuntu-xenial:0x06-multivariate_prob$
```
**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `math/0x06-multivariate_prob`
*   File: [`multinormal.py`](./multinormal.py)

#### 3\. PDF 

Update the class `MultiNormal`:

*   public instance method `def pdf(self, x):` that calculates the PDF at a data point:
    *   `x` is a `numpy.ndarray` of shape `(d, 1)` containing the data point whose PDF should be calculated
        *   `d` is the number of dimensions of the `Multinomial` instance
    *   If `x` is not a `numpy.ndarray`, raise a `TypeError` with the message `x must by a numpy.ndarray`
    *   If `x` is not of shape `(d, 1)`, raise a `ValueError` with the message `x mush have the shape ({d}, 1)`
    *   Returns the value of the PDF
    *   You are not allowed to use the function `numpy.cov`

```   
    alexa@ubuntu-xenial:0x06-multivariate_prob$ ./3-main.py 
    [[ 8.20311936]
     [32.84231319]
     [ 9.67254478]]
    0.00022930236202143824
    alexa@ubuntu-xenial:0x06-multivariate_prob$ 
``` 
**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `math/0x06-multivariate_prob`
*   File: [`multinormal.py`](./multinormal.py)

</details>
