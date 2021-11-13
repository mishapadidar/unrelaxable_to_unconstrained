from bfgs_b import BFGS_B
import numpy as np
import sys
sys.path.append("../problems/")
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

# optimize
z = BFGS_B(f,f.grad,y0,f.lb,f.ub,gamma=0.4,max_iter=1000,gtol=1e-5)

print("Optimal Value is ",f(z))
print("Minima Found is ",z)
print("Distance to Optima: ",np.linalg.norm(z- f.minimum))
