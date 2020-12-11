# 0x07. Bayesian Probability

## Learning Objectives

-   What is Bayesian Probability?
-   What is Bayes’ rule and how do you use it?
-   What is a base rate?
-   What is a prior?
-   What is a posterior?
-   What is a likelihood?

## Resources

**Read or watch**:
*   [Bayesian probability](https://en.wikipedia.org/wiki/Bayesian_probability)
*   [Bayesian statistics](https://en.wikipedia.org/wiki/Bayesian_statistics)
*   [Bayes’ Theorem - The Simplest Case](https://www.youtube.com/watch?v=XQoLVl31ZfQ "Bayes' Theorem - The Simplest Case")
*   [A visual guide to Bayesian thinking](https://www.youtube.com/watch?v=BrK7X_XlGB8 "A visual guide to Bayesian thinking")
*   [Base Rates](http://onlinestatbook.com/2/probability/base_rates.html "Base Rates")
*   [Bayesian statistics: a comprehensive course](https://www.youtube.com/playlist?list=PLFDbGp5YzjqXQ4oE4w9GVWdiokWB9gEpm "Bayesian statistics: a comprehensive course")
    - [Bayes’ rule - an intuitive explanation](https://www.youtube.com/watch?v=EbyUsf_jUjk&list=PLFDbGp5YzjqXQ4oE4w9GVWdiokWB9gEpm&index=14&ab_channel=Oxeduc)
    - [Bayes’ rule in statistics](https://www.youtube.com/watch?v=i567qvWejJA&list=PLFDbGp5YzjqXQ4oE4w9GVWdiokWB9gEpm&index=15&ab_channel=Oxeduc)
    - [Bayes’ rule in inference - likelihood](https://www.youtube.com/watch?v=c69a_viMRQU&list=PLFDbGp5YzjqXQ4oE4w9GVWdiokWB9gEpm&index=16&ab_channel=Oxeduc)
    - [Bayes’ rule in inference - the prior and denominator](https://www.youtube.com/watch?v=a5QDDZLGSXY&list=PLFDbGp5YzjqXQ4oE4w9GVWdiokWB9gEpm&index=17&ab_channel=Oxeduc)
    - [Bayes’ rule denominator: discrete and continuous](https://www.youtube.com/watch?v=QEzeLh6L9Tg&list=PLFDbGp5YzjqXQ4oE4w9GVWdiokWB9gEpm&index=24&ab_channel=Oxeduc)
    - [Bayes’ rule: why likelihood is not a probability](https://www.youtube.com/watch?v=sm60vapz2jQ&list=PLFDbGp5YzjqXQ4oE4w9GVWdiokWB9gEpm&index=25&ab_channel=Oxeduc)


## Quiz questions

<details>
<summary>Show</summary>

### Question #0

Bayes’ rule states that `P(A | B) = P(B | A) * P(A) / P(B)`

What is `P(A | B)`?

* [ ] Likelihood

* [ ] Marginal probability

* [x] Posterior probability

* [ ] Prior probability

### Question #1

Bayes’ rule states that `P(A | B) = P(B | A) * P(A) / P(B)`

What is `P(B | A)`?

* [x] Likelihood

* [ ] Marginal probability

* [ ] Posterior probability

* [ ] Prior probability

### Question #2

Bayes’ rule states that `P(A | B) = P(B | A) * P(A) / P(B)`

What is `P(A)`?

* [ ] Likelihood

* [ ] Marginal probability

* [ ] Posterior probability

* [x] Prior probability

### Question #3

Bayes’ rule states that `P(A | B) = P(B | A) * P(A) / P(B)`

What is `P(B)`?

* [ ] Likelihood

* [x] Marginal probability

* [ ] Posterior probability

* [ ] Prior probability

</details>

## Tasks

<details>
<summary>Show Tasks</summary>

### 0\. Likelihood 

You are conducting a study on a revolutionary cancer drug and are looking to find the probability that a patient who takes this drug will develop severe side effects. During your trials, `n` patients take the drug and `x` patients develop severe side effects. You can assume that `x` follows a binomial distribution.

Write a function `def likelihood(x, n, P):` that calculates the likelihood of obtaining this data given various hypothetical probabilities of developing severe side effects:

*   `x` is the number of patients that develop severe side effects
*   `n` is the total number of patients observed
*   `P` is a 1D `numpy.ndarray` containing the various hypothetical probabilities of developing severe side effects
*   If `n` is not a positive integer, raise a `ValueError` with the message `n must be a positive integer`
*   If `x` is not an integer that is greater than or equal to `0`, raise a `ValueError` with the message `x must be an integer that is greater than or equal to 0`
*   If `x` is greater than `n`, raise a `ValueError` with the message `x cannot be greater than n`
*   If `P` is not a 1D `numpy.ndarray`, raise a `TypeError` with the message `P must be a 1D numpy.ndarray`
*   If any value in `P` is not in the range `[0, 1]`, raise a `ValueError` with the message `All values in P must be in the range [0, 1]`
*   Returns: a 1D `numpy.ndarray` containing the likelihood of obtaining the data, `x` and `n`, for each probability in `P`, respectively

```   
    alexa@ubuntu-xenial:0x07-bayesian_prob$ ./0-main.py 
    [0.00000000e+00 2.71330957e-04 8.71800070e-02 3.07345706e-03
     5.93701546e-07 1.14387595e-12 1.09257177e-20 6.10151799e-32
     9.54415702e-49 1.00596671e-78 0.00000000e+00]
    alexa@ubuntu-xenial:0x07-bayesian_prob$
```
**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `math/0x07-bayesian_prob`
*   File: [`0-likelihood.py`](./0-likelihood.py)

### 1\. Intersection 

Based on `0-likelihood.py`, write a function `def intersection(x, n, P, Pr):` that calculates the intersection of obtaining this data with the various hypothetical probabilities:

*   `x` is the number of patients that develop severe side effects
*   `n` is the total number of patients observed
*   `P` is a 1D `numpy.ndarray` containing the various hypothetical probabilities of developing severe side effects
*   `Pr` is a 1D `numpy.ndarray` containing the prior beliefs of `P`
*   If `n` is not a positive integer, raise a `ValueError` with the message `n must be a positive integer`
*   If `x` is not an integer that is greater than or equal to `0`, raise a `ValueError` with the message `x must be an integer that is greater than or equal to 0`
*   If `x` is greater than `n`, raise a `ValueError` with the message `x cannot be greater than n`
*   If `P` is not a 1D `numpy.ndarray`, raise a `TypeError` with the message `P must be a 1D numpy.ndarray`
*   If `Pr` is not a `numpy.ndarray` with the same shape as `P`, raise a `TypeError` with the message `Pr must be a numpy.ndarray with the same shape as P`
*   If any value in `P` or `Pr` is not in the range `[0, 1]`, raise a `ValueError` with the message `All values in {P} must be in the range [0, 1]` where `{P}` is the incorrect variable
*   If `Pr` does not sum to `1`, raise a `ValueError` with the message `Pr must sum to 1` **Hint: use [numpy.isclose](/rltoken/7pptg2vy0_-c0qQ9MnZu1w "numpy.isclose")**
*   All exceptions should be raised in the above order
*   Returns: a 1D `numpy.ndarray` containing the intersection of obtaining `x` and `n` with each probability in `P`, respectively

```
    alexa@ubuntu-xenial:0x07-bayesian_prob$ ./1-main.py 
    [0.00000000e+00 2.46664506e-05 7.92545518e-03 2.79405187e-04
     5.39728678e-08 1.03988723e-13 9.93247059e-22 5.54683454e-33
     8.67650639e-50 9.14515194e-80 0.00000000e+00]
    alexa@ubuntu-xenial:0x07-bayesian_prob$
```
**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `math/0x07-bayesian_prob`
*   File: [`1-intersection.py`](./1-intersection.py)

### 2\. Marginal Probability 

Based on `1-intersection.py`, write a function `def marginal(x, n, P, Pr):` that calculates the marginal probability of obtaining the data:

*   `x` is the number of patients that develop severe side effects
*   `n` is the total number of patients observed
*   `P` is a 1D `numpy.ndarray` containing the various hypothetical probabilities of patients developing severe side effects
*   `Pr` is a 1D `numpy.ndarray` containing the prior beliefs about `P`
*   If `n` is not a positive integer, raise a `ValueError` with the message `n must be a positive integer`
*   If `x` is not an integer that is greater than or equal to `0`, raise a `ValueError` with the message `x must be an integer that is greater than or equal to 0`
*   If `x` is greater than `n`, raise a `ValueError` with the message `x cannot be greater than n`
*   If `P` is not a 1D `numpy.ndarray`, raise a `TypeError` with the message `P must be a 1D numpy.ndarray`
*   If `Pr` is not a `numpy.ndarray` with the same shape as `P`, raise a `TypeError` with the message `Pr must be a numpy.ndarray with the same shape as P`
*   If any value in `P` or `Pr` is not in the range `[0, 1]`, raise a `ValueError` with the message `All values in {P} must be in the range [0, 1]` where `{P}` is the incorrect variable
*   If `Pr` does not sum to `1`, raise a `ValueError` with the message `Pr must sum to 1`
*   All exceptions should be raised in the above order
*   Returns: the marginal probability of obtaining `x` and `n`

```
    alexa@ubuntu-xenial:0x07-bayesian_prob$ ./2-main.py 
    0.008229580791426582
    alexa@ubuntu-xenial:0x07-bayesian_prob$
```

**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `math/0x07-bayesian_prob`
*   File: [`2-marginal.py`](./2-marginal.py)

### 3\. Posterior 

Based on `2-marginal.py`, write a function `def posterior(x, n, P, Pr):` that calculates the posterior probability for the various hypothetical probabilities of developing severe side effects given the data:

*   `x` is the number of patients that develop severe side effects
*   `n` is the total number of patients observed
*   `P` is a 1D `numpy.ndarray` containing the various hypothetical probabilities of developing severe side effects
*   `Pr` is a 1D `numpy.ndarray` containing the prior beliefs of `P`
*   If `n` is not a positive integer, raise a `ValueError` with the message `n must be a positive integer`
*   If `x` is not an integer that is greater than or equal to `0`, raise a `ValueError` with the message `x must be an integer that is greater than or equal to 0`
*   If `x` is greater than `n`, raise a `ValueError` with the message `x cannot be greater than n`
*   If `P` is not a 1D `numpy.ndarray`, raise a `TypeError` with the message `P must be a 1D numpy.ndarray`
*   If `Pr` is not a `numpy.ndarray` with the same shape as `P`, raise a `TypeError` with the message `Pr must be a numpy.ndarray with the same shape as P`
*   If any value in `P` or `Pr` is not in the range `[0, 1]`, raise a `ValueError` with the message `All values in {P} must be in the range [0, 1]` where `{P}` is the incorrect variable
*   If `Pr` does not sum to `1`, raise a `ValueError` with the message `Pr must sum to 1`
*   All exceptions should be raised in the above order
*   Returns: the posterior probability of each probability in `P` given `x` and `n`, respectively

```    
    alexa@ubuntu-xenial:0x07-bayesian_prob$ ./3-main.py 
    [0.00000000e+00 2.99729127e-03 9.63044824e-01 3.39513268e-02
     6.55839819e-06 1.26359684e-11 1.20692303e-19 6.74011797e-31
     1.05430721e-47 1.11125368e-77 0.00000000e+00]
    alexa@ubuntu-xenial:0x07-bayesian_prob$
```
**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `math/0x07-bayesian_prob`
*   File: [`3-posterior.py`](./3-posterior.py)

### 4\. Continuous Posterior 

Based on `3-posterior.py`, write a function `def posterior(x, n, p1, p2):` that calculates the posterior probability that the probability of developing severe side effects falls within a specific range given the data:

*   `x` is the number of patients that develop severe side effects
*   `n` is the total number of patients observed
*   `p1` is the lower bound on the range
*   `p2` is the upper bound on the range
*   You can assume the prior beliefs of `p` follow a uniform distribution
*   If `n` is not a positive integer, raise a `ValueError` with the message `n must be a positive integer`
*   If `x` is not an integer that is greater than or equal to `0`, raise a `ValueError` with the message `x must be an integer that is greater than or equal to 0`
*   If `x` is greater than `n`, raise a `ValueError` with the message `x cannot be greater than n`
*   If `p1` or `p2` are not floats within the range `[0, 1]`, raise a`ValueError` with the message `{p} must be a float in the range [0, 1]` where `{p}` is the corresponding variable
*   if `p2` <= `p1`, raise a `ValueError` with the message `p2 must be greater than p1`
*   The only import you are allowed to use is `from scipy import math, special`
*   Returns: the posterior probability that `p` is within the range `[p1, p2]` given `x` and `n`

```
    alexa@ubuntu-xenial:0x07-bayesian_prob$ ./100-main.py 
    0.6098093274896035
    alexa@ubuntu-xenial:0x07-bayesian_prob$
```
**Repo:**

*   GitHub repository: `holbertonschool-machine_learning`
*   Directory: `math/0x07-bayesian_prob`
*   File: [`100-continuous.py`](./100-continuous.py)

</details>
