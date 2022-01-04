from bfgs_b import BFGS_B
import numpy as np
import sys
sys.path.append("../problems/")
from convex_quadratic import ConvexQuadratic
from steepest_descent import SD
from gradient_descent import GD
sys.path.append("../utils/")
sys.path.append("../generators/")
from rescale import *
from sigmoid import Sigmoid


dim = 10
yopt = np.random.uniform(0,1,dim)
#yopt = np.ones(dim) +np.random.randn(dim)
Q = np.random.randn(dim,dim)
A = Q @ Q.T + 0.1*np.eye(dim)
lb = np.zeros(dim)
ub = np.ones(dim)
f = ConvexQuadratic(lb,ub,A,yopt)

# initial pint
z0 = np.random.uniform(lb,ub)
y0 = to_unit_cube(z0,lb,ub)
sigma0 = 1.0
gen = Sigmoid(sigma0)

# get the starting point on the right domain
x0 = gen.inv(y0)

# optimizer stuff
gtol = 1e-10
max_iter=1000

# solve
ff = lambda yy: f(from_unit_cube(yy,lb,ub))
ff_grad = lambda yy: np.diag(ub-lb) @ f.grad(from_unit_cube(yy,lb,ub))
xopt = SD(ff,ff_grad,gen,x0,max_iter=max_iter,gtol=gtol,verbose=False)
# dont forget to map back!
z = from_unit_cube(gen(xopt),lb,ub)
print("Steepest Descent")
print("Optimal Value is ",f(z))
print("Minima Found is ",z)
print("Distance to Optima: ",np.linalg.norm(z- f.minimum))
print("")

# now run gradient descent
ft = lambda xx: f(from_unit_cube(gen(xx),lb,ub))
ft_grad = lambda xx: gen.jac(xx) @ np.diag(ub-lb) @ f.grad(from_unit_cube(gen(xx),lb,ub))
xopt = GD(ft,ft_grad,x0,max_iter=max_iter,gtol=gtol,verbose=False)
z = from_unit_cube(gen(xopt),lb,ub)
print("Gradient Descent")
print("Optimal Value is ",f(z))
print("Minima Found is ",z)
print("Distance to Optima: ",np.linalg.norm(z- f.minimum))
