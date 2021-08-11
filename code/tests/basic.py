import numpy as np
from scipy.optimize import minimize
import sys
sys.path.append("../generators/")
sys.path.append("../problems/")
sys.path.append("../utils/")
from sigmoid import Sigmoid
from quadratic1 import Quadratic1
from rosenbrock import Rosenbrock
from rescale import *

dim  = 2
f = Rosenbrock(dim)
# f    = Quadratic1(dim)
lb   = f.lb
ub   = f.ub
# generator
sig = Sigmoid()
# merit
ft = lambda xx: f(from_unit_cube(sig(xx),lb,ub))
jac = lambda xx: sig.jac(xx) @ np.diag(ub-lb) @ f.grad(from_unit_cube(sig(xx),lb,ub))

# optimize
x0 = np.random.randn(dim)
res = minimize(ft,x0,jac=jac,method='BFGS',options={'gtol':1e-8})

xopt = res.x
print(res)
print("Minima Found is ",from_unit_cube(sig(xopt),lb,ub))
print("Distance to Optima: ",np.linalg.norm(from_unit_cube(sig(xopt),lb,ub) - f.minimum))