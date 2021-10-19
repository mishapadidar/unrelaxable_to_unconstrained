import numpy as np
from sigup import sigup
import sys
sys.path.append("../problems/")
from quadratic1 import Quadratic1
from rosenbrock import Rosenbrock
from convex_quadratic import ConvexQuadratic

dim  = 2
#f = Rosenbrock(dim)
# f    = Quadratic1(dim)
#lb   = f.lb
#ub   = f.ub
lb = np.zeros(dim)
ub = np.ones(dim)
yopt = np.ones(dim) + np.array([-1e-7,1e-1])
#yopt = np.ones(dim) 
A = np.diag(np.array([100,2]))
f = ConvexQuadratic(lb,ub,A,yopt)

# optimizer params
y0 = (lb+ub)/2
n_solves = 10
gamma = 1.0
eps = 1e-9
method = 'BFGS'
verbose=True

# optimize
xopt,yopt = sigup(f,f.grad,lb,ub,y0,eps = eps,gamma=gamma,n_solves=n_solves,method=method,verbose=verbose)

print("Optimal Value is ",f(yopt))
print("Minima Found is ",yopt)
print("Distance to Optima: ",np.linalg.norm(yopt- f.minimum))
