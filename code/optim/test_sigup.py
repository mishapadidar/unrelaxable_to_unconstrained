import numpy as np
from sigup import SIGUP
import sys
sys.path.append("../problems/")
from quadratic1 import Quadratic1
from rosenbrock import Rosenbrock
from convex_quadratic import ConvexQuadratic

dim  = 2

# problem 1
f = Rosenbrock(dim)

# problem 2
#yopt = np.ones(dim) 
#yopt = np.ones(dim) + np.array([-1e-7,1e-1])
#yopt = np.ones(dim) - np.array([1e-7,1e-1])
#yopt = np.ones(dim) - np.array([0.5,0.0])
#yopt = np.ones(dim) + np.array([0.5,0.0])
#A = np.diag(np.array([100,2]))
#lb = np.zeros(dim)
#ub = np.ones(dim)
#f = ConvexQuadratic(lb,ub,A,yopt)

# optimizer params
y0 = (f.lb+f.ub)/2
n_solves = 10
gamma = 1.0
eps = 1e-9
verbose=True

# optimize
solver = SIGUP(f,f.grad,f.lb,f.ub,y0,eps = eps,delta=eps,gamma=gamma)
z = solver.solve(verbose=verbose)


print("Optimal Value is ",f(z))
print("Minima Found is ",z)
print("Distance to Optima: ",np.linalg.norm(z- f.minimum))
