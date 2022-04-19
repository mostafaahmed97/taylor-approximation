import math
import autograd.numpy as np
from autograd import grad

print(np)

def taylor_approx(f, a, n_terms):
    deriv = f
    coeffs = []

    for i in range(n_terms):
        c = (1 / math.factorial(i)) * deriv(a)
        coeffs.append(c)
        deriv = grad(deriv)


    def t(x):
        ys = []

        for val in x:
            if len(coeffs) == 1:
                ys.append(coeffs[0])
                continue

            y = coeffs[0] + sum(coeffs[i] * (val - a) ** i for i in range(1, len(coeffs)))
            ys.append(y)
        return ys  

    return t
