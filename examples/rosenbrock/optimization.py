#!/usr/bin/env python3
#-*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pickle
import sys
import os

import jax
import jax.numpy as jnp

def main():

    # Perform minimization
    N_dim = 4
    x0 = np.random.randn(N_dim)
    res = minimize(cost_function, x0, jac=grad_cost_function, method='BFGS')
    print(res.x)

    # Evaluate Hessian
    H = hess_cost_function(res.x)
    cov = np.linalg.inv(H)
    print(cov)

    # Store results
    with open('params_optimization.pkl', 'wb') as fid:
        pickle.dump(res.x, fid)
        pickle.dump(cov, fid)


def cost_function(x):
    """
    Rosenbrock cost function to minimize.
    """
    N_dim = x.shape[0]

    # Constants
    a = 1.0
    b = 5.0
    
    # Compute multi-dimensional Rosenbrock function
    f = 0.0
    for i in range(1, N_dim):
        df = b * (x[i] - x[i-1]**2)**2 + (a - x[i-1])**2
        f += df

    return f

# Define the Jacobian and Hessian funcitons
grad_cost_function = jax.jacfwd(cost_function)
hess_cost_function = jax.jacfwd(jax.jacrev(cost_function))


if __name__ == '__main__':
    main()

# end of file
