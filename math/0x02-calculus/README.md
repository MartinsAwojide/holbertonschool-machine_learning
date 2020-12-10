# 0x02. Calculus
## Learning Objectives:
### General
- Summation and Product notation
- What is a series?
- Common series
- What is a derivative?
- What is the product rule?
- What is the chain rule?
- Common derivative rules
- What is a partial derivative?
- What is an indefinite integral?
- What is a definite integral?
- What is a double integral?

## Resources
<details>
<summary>Show</summary>
   
**Read or watch:**
- [Sigma Notation](https://www.youtube.com/watch?v=TjMLzklnn2c&ab_channel=EddieWoo) (starting at 0:32)
- [Π Product Notation](https://www.youtube.com/watch?v=sP1-EQJKSgk&ab_channel=EddieWoo) (up to 0:20)
- [Sigma and Pi Notation](https://mathmaine.com/2010/04/01/sigma-and-pi-notation/)
- [What is a Series?](https://virtualnerd.com/algebra-2/sequences-series/define/defining-series/series-definition)
- [What is a Mathematical Series?](https://www.quickanddirtytips.com/education/math/what-is-a-mathematical-series)
- [List of mathematical series: Sums of powers](https://en.wikipedia.org/wiki/List_of_mathematical_series#Sums_of_powers)
- [Bernoulli Numbers(Bn)](https://en.wikipedia.org/wiki/Bernoulli_number)
- [Bernoulli Polynomials(Bn(x))](https://en.wikipedia.org/wiki/Bernoulli_polynomials)
- [Derivative (mathematics)](https://simple.wikipedia.org/wiki/Derivative_%28mathematics%29)
- [Calculus for ML](https://ml-cheatsheet.readthedocs.io/en/latest/calculus.html)
- [1 of 2: Seeing the big picture](https://www.youtube.com/watch?v=tt2DGYOi3hc&ab_channel=EddieWoo)
- [2 of 2: First Principles](https://www.youtube.com/watch?v=50Bda5VKbqA&ab_channel=EddieWoo)
- [1 of 2: Finding the Derivative](https://www.youtube.com/watch?v=fXYhyyJpFe8&ab_channel=EddieWoo)
- [2 of 2: What do we discover?](https://www.youtube.com/watch?v=Un0RcTMPJ64&ab_channel=EddieWoo)
- [Deriving a Rule for Differentiating Powers of x](https://www.youtube.com/watch?v=I8IM9P-2TRU&ab_channel=EddieWoo)
- [1 of 3: Introducing a substitution](https://www.youtube.com/watch?v=U0m4MsOgETw&ab_channel=EddieWoo)
- [2 of 3: Combining derivatives](https://www.youtube.com/watch?v=z-tEsz0bSrA&ab_channel=EddieWoo)
- [How To Understand Derivatives: The Product, Power & Chain Rules](https://betterexplained.com/articles/derivatives-product-power-chain/)
- [Product Rule](https://en.wikipedia.org/wiki/Product_rule)
- [Common Derivatives and Integrals](https://www.coastal.edu/media/academics/universitycollege/mathcenter/handouts/calculus/deranint.PDF)
- [Introduction to partial derivatives](https://mathinsight.org/partial_derivative_introduction)
- [Partial derivatives - How to solve?](https://www.youtube.com/watch?v=rnoToCoEK48&ab_channel=KristaKing)
- [Integral](https://en.wikipedia.org/wiki/Integral)
- [Integration and the fundamental theorem of calculus](https://www.youtube.com/watch?v=rfG8ce4nNh0&ab_channel=3Blue1Brown)
- [Introduction to Integration](https://www.mathsisfun.com/calculus/integration-introduction.html)
- [Indefinite Integral - Basic Integration Rules, Problems, Formulas, Trig Functions, Calculus](https://www.youtube.com/watch?v=o75AqTInKDU&ab_channel=TheOrganicChemistryTutor)
- [Definite Integrals](https://www.mathsisfun.com/calculus/integration-definite.html)
- [Definite Integral](https://www.youtube.com/watch?v=Gc3QvUB0PkI&ab_channel=TheOrganicChemistryTutor)
- [Multiple integral](https://en.wikipedia.org/wiki/Multiple_integral)
- [Double integral 1](https://www.youtube.com/watch?v=85zGYB-34jQ&ab_channel=KhanAcademy)
- [Double integrals 2](https://www.youtube.com/watch?v=TdLD2Zh-nUQ&ab_channel=KhanAcademy)

</details>

---

## Tasks:
<details>
<summary>View tasks</summary>
### 0. Sigma is for Sum

Solve the next summation:

![\sum_{i=2}^{5} i](https://render.githubusercontent.com/render/math?math=%5Csum_%7Bi%3D2%7D%5E%7B5%7D%20i)

**1.** 3 + 4 + 5

**2.** 3 + 4

**3.** 2 + 3 + 4 + 5

**4.** 2 + 3 + 4

**Repo:**

* GitHub repository: holbertonschool-machine_learning
* Directory: math/0x02-calculus
* File: [`0-sigma_is_for_sum`](./0-sigma_is_for_sum)

### 1. It's actually pronounced sEEgma

Solve the next summation:

![\sum_{k=1}^{4} 9i - 2k](https://render.githubusercontent.com/render/math?math=%5Csum_%7Bk%3D1%7D%5E%7B4%7D%209i%20-%202k)   

**1.** 90 - 20

**2.** 36i - 20

**3.** 90 - 8k

**4.** 36i - 8k

**Repo:**

* GitHub repository: holbertonschool-machine_learning
* Directory: math/0x02-calculus
* File: [`1-seegma`](1-seegma)

### 2. Pi is for Product

Solve the next repeated multiplication:

![\prod_{i=1}^{m} i](https://render.githubusercontent.com/render/math?math=%5Cprod_%7Bi%3D1%7D%5E%7Bm%7D%20i)

**1.** (m - 1)!

**2.** 0

**3.** (m + 1)!

**4.** m!

**Repo:**

* GitHub repository: holbertonschool-machine_learning
* Directory: math/0x02-calculus
* File: [`2-pi_is_for_product`](2-pi_is_for_product)

### 3. It's actually pronounced pEE

Solve the next repeated multiplication:

![\prod_{i=0}^{10} i](https://render.githubusercontent.com/render/math?math=%5Cprod_%7Bi%3D0%7D%5E%7B10%7D%20i)

**1.** 10!

**2.** 9!

**3.** 100

**4.** 0

**Repo:**

* GitHub repository: holbertonschool-machine_learning
* Directory: math/0x02-calculus
* File: [3-pee](./3-pee)

### 4. Hello, derivatives!

Found  ![\frac{dy}{dx}](https://render.githubusercontent.com/render/math?math=%5Cfrac%7Bdy%7D%7Bdx%7D), where:

![y = x^{4} + 3x^{3} - 5x + 1](https://render.githubusercontent.com/render/math?math=y%20%3D%20x%5E%7B4%7D%20%2B%203x%5E%7B3%7D%20-%205x%20%2B%201)

**1.** ![3x^{3} + 6x^{2} - 4](https://render.githubusercontent.com/render/math?math=3x^{3}+6x^{2}-4)

**2.** ![4x^{3} + 6x^{2} - 5](https://render.githubusercontent.com/render/math?math=4x^{3}+6x^{2}-5)

**3.** ![4x^{3} + 9x^{2} - 5](https://render.githubusercontent.com/render/math?math=4x^{3}+9x^{2}-5)

**4.** ![4x^{3} + 9x^{2} - 4](https://render.githubusercontent.com/render/math?math=4x^{3}+9x^{2}-4)

### 5. A log on the fire

Find derivative of:

![\frac{d(xln(x))}{dx}](https://render.githubusercontent.com/render/math?math=%5Cfrac%7Bd(xln(x))%7D%7Bdx%7D)

### 6. It is difficult to free fools from the chains they revere 

Find derivative of:

![\frac{d(ln(x^{2}))}{dx}](https://render.githubusercontent.com/render/math?math=%5Cfrac%7Bd(ln(x%5E%7B2%7D))%7D%7Bdx%7D)

### 7. Partial truths are often more insidious than total falsehoods 

Find  ![\frac{\partial}{\partial y} f(x,y)](https://render.githubusercontent.com/render/math?math=%5Cfrac%7B%5Cpartial%7D%7B%5Cpartial%20y%7D%20f(x%2Cy)), where:

![f(x,y) = e^{xy}](https://render.githubusercontent.com/render/math?math=f(x%2Cy)%20%3D%20e%5E%7Bxy%7D)

and

![\frac{\partial x}{\partial y} = \frac{\partial y}{\partial x} = 0](https://render.githubusercontent.com/render/math?math=%5Cfrac%7B%5Cpartial%20x%7D%7B%5Cpartial%20y%7D%20%3D%20%5Cfrac%7B%5Cpartial%20y%7D%7B%5Cpartial%20x%7D%20%3D%200) 

### 8. Put it all together and what do you get?

Find ![\frac{\partial}{\partial y \partial x} e^{x^{2}y}](https://render.githubusercontent.com/render/math?math=%5Cfrac%7B%5Cpartial%7D%7B%5Cpartial%20y%20%5Cpartial%20x%7D%20e%5E%7Bx%5E%7B2%7Dy%7D), where:

![\frac{\partial x}{\partial y} = \frac{\partial y}{\partial x} = 0](https://render.githubusercontent.com/render/math?math=%5Cfrac%7B%5Cpartial%20x%7D%7B%5Cpartial%20y%7D%20%3D%20%5Cfrac%7B%5Cpartial%20y%7D%7B%5Cpartial%20x%7D%20%3D%200) 

###  9. Our life is the sum total of all the decisions we make every day, and those decisions are determined by our priorities

* File:  9-sum_total.py

Function that calculates ![\sum_{i=1}^{n} i^{2}](https://render.githubusercontent.com/render/math?math=%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%20i%5E%7B2%7D).

Example:

```python
>>> n = 5
>>> print(summation_i_squared(n))
55
```

### 10. Derive happiness in oneself from a good day's work

Function `def poly_derivative(poly):` that calculates the derivative of a polynomial. Where poly is a list of coefficients representing a polynomial.

Example:

f(x) = x³ + 3x + 5 

↓         ↓      ↓

f(x) = 5 + 3x  + x³ --->  f'(x) = 3 + 3x²

`poly = [5, 3, 0, 3]`

```python
>>> poly = [5, 3, 0, 1]
>>> print(poly_derivative(poly))
[3, 0, 3]
```

### 11. Good grooming is integral and impeccable style is a must 

Find the antiderivative of:

![\int x^{3}dx](https://render.githubusercontent.com/render/math?math=%5Cint%20x%5E%7B3%7Ddx)

### 12. We are all an integral part of the web of life

Find the antiderivative of:

![\int e^{2y}dy](https://render.githubusercontent.com/render/math?math=%5Cint%20e%5E%7B2y%7Ddy)

### 13. Create a definite plan for carrying out your desire and begin at once

Find the definite integration of:

![\int_{0}^{3}u^{2}du](https://render.githubusercontent.com/render/math?math=%5Cint_%7B0%7D%5E%7B3%7Du%5E%7B2%7Ddu)

### 14. My talents fall within definite limitations 

Find the definite integration of:

![\int_{-1}^{0}\frac{1}{v}dv](https://render.githubusercontent.com/render/math?math=%5Cint_%7B-1%7D%5E%7B0%7D%5Cfrac%7B1%7D%7Bv%7Ddv)

### 15. Winners are people with definite purpose in life

Find the definite integration of:

![\int_{0}^{5}xdy](https://render.githubusercontent.com/render/math?math=%5Cint_%7B0%7D%5E%7B5%7Dxdy)

### 16. Double whammy 

Answer the next double integration:

![\int_{1}^{2}\int_{0}^{3}x^{2}y^{-1}dxdy](https://render.githubusercontent.com/render/math?math=%5Cint_%7B1%7D%5E%7B2%7D%5Cint_%7B0%7D%5E%7B3%7Dx%5E%7B2%7Dy%5E%7B-1%7Ddxdy)

### 17. Integrate

Function `def poly_integral(poly, C=0):` that calculates the integral of a polynomial. Where `C` is the integration constant.

Example:

f(x) = x³ + 3x + 5 

↓         ↓      ↓

f(x) = 5 + 3x  + x³ -> `poly = [5, 3, 0, 1]`

∫f(x) ---> ∫(5 + 3x + x³)dx -> C + 5x + (3/2)x² + (1/4)x⁴

```python
>>> poly = [5, 3, 0, 1]
>>> print(poly_integral(poly))
[0, 5, 1.5, 0, 0.25]
>>> poly = [5, 3, 0, 1]
>>> print(poly_integral(poly, 1))
[1, 5, 1.5, 0, 0.25]
```
</details>

---
