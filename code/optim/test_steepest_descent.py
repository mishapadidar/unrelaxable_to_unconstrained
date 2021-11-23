from bfgs_b import BFGS_B
import numpy as np
import sys
sys.path.append("../problems/")
from rosenbrock import Rosenbrock
from convex_quadratic import ConvexQuadratic
from steepest_descent import SD
sys.path.append("../utils/")
sys.path.append("../generators/")
from rescale import *
from sigmoid import Sigmoid

dim  = 10

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

# initial pint
z0 = (f.lb+f.ub)/2

# start with a good choice of sigma
ub = f.ub
lb = f.lb
y0 = to_unit_cube(z0,f.lb,f.ub)
sigma0 = 1.0/(y0*(1-y0))
gen = Sigmoid(sigma0)

# get the starting point on the right domain
x0 = gen.inv(y0)

# solve
ft = lambda xx: f(from_unit_cube(gen(xx),lb,ub))
ft_jac = lambda xx: gen.jac(xx) @ np.diag(ub-lb) @ f.grad(from_unit_cube(gen(xx),lb,ub))
xopt = SD(ft,ft_jac,gen.jac,x0,gamma=0.5,max_iter=10000,gtol=1e-5,verbose=False)
# dont forget to map back!
z = from_unit_cube(gen(xopt),lb,ub)

print("Optimal Value is ",f(z))
print("Minima Found is ",z)
print("Distance to Optima: ",np.linalg.norm(z- f.minimum))
