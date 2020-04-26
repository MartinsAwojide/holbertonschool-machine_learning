#!/usr/bin/env python3

poly_integral = __import__('17-integrate').poly_integral

poly = [5, 3, 0, 1]
print(poly_integral(poly, 2))
print(poly_integral([1], 2))
print(poly_integral([9]))
print(poly_integral([0]))
print(poly_integral(5))
print(poly_integral(['a', 'b']))