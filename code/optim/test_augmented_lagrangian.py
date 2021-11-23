import numpy as np
from augmented_lagrangian import AugmentedLagrangian
import sys
sys.path.append("../problems/")
from rosenbrock import Rosenbrock
from convex_quadratic import ConvexQuadratic

dim  = 2

# problem 1
#f = Rosenbrock(dim)

# problem 2
#yopt = np.ones(dim) 
#yopt = np.ones(dim) + np.array([-1e-7,1e-1])
yopt = np.ones(dim) - np.array([1e-7,1e-1])
#yopt = np.ones(dim) - np.array([0.5,0.0])
#yopt = np.ones(dim) + np.array([0.5,0.0])
A = np.diag(np.array([100,2]))
lb = np.zeros(dim)
ub = np.ones(dim)
f = ConvexQuadratic(lb,ub,A,yopt)

# optimizer params
y0 = (f.lb+f.ub)/2
verbose=True

# optimize
solver = AugmentedLagrangian(f,f.grad,f.lb,f.ub,gtol=1e-3,ctol=1e-5)
z = solver.solve(y0)

print("Optimal Value is ",f(z))
print("Minima Found is ",z)
print("Distance to Optima: ",np.linalg.norm(z- f.minimum))
