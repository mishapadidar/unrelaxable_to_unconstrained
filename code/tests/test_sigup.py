import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../problems/")
sys.path.append("../utils/")
sys.path.append("../optim/")
from rosenbrock import Rosenbrock
from scipy.optimize import minimize,Bounds
from convex_quadratic import ConvexQuadratic
from eval_wrapper import eval_wrapper
from sigup import *

dim  = 2

# problem 1
lb = np.zeros(dim)
ub = np.ones(dim)
yopt = np.ones(dim) + np.array([-1e-7,0.5])
A = np.diag(np.array([100,2]))
f = ConvexQuadratic(lb,ub,A,yopt)

# problem 2
#f = Rosenbrock(dim)

# optimizer params
y0 = (f.lb+f.ub)/2
#y0 = np.random.uniform(f.lb,f.ub)
gamma = 2.0
delta = 1e-2
eps = 1e-10
solve_method = "scipy"
update_method = "adaptive"
verbose=True

# optimize
sigmas = [0.01,0.1,1.0,10.0,100.0,1000.0]
for sigma0 in sigmas:
  print(f"\nsigup, sigma0 = {sigma0}")
  sigup = SIGUP(f,f.grad,f.lb,f.ub,y0,eps = eps,delta=delta,gamma=gamma,sigma0=sigma0,solve_method=solve_method,
                update_method=update_method)
  z = sigup.solve(verbose=verbose)
  #z = sigup(func,f.grad,f.lb,f.ub,y0,sigma0=sigma0,delta=eps,eps = eps,gamma=gamma,method=method,verbose=verbose)
  X = sigup.X
  fX = sigup.fX
  updates = sigup.updates
  print("Optimal Value is ",f(z))
  print("Minima Found is ",z)
  print("Distance to Optima: ",np.linalg.norm(z- f.minimum))
  plt.plot(np.minimum.accumulate(fX),linewidth=3,label=f'sigup, $\sigma_0$ = {sigma0}')
  plt.scatter(updates,np.minimum.accumulate(fX)[updates])

# optimize
func = eval_wrapper(f,dim)
bounds = Bounds(f.lb,f.ub)
res = minimize(func,y0,jac=f.grad,bounds=bounds,method='L-BFGS-B',options={'gtol':eps})
z = res.x
X = func.X
fX = func.fX
print("\nL-BFGS-B")
print("Optimal Value is ",f(z))
print("Minima Found is ",z)
print("Distance to Optima: ",np.linalg.norm(z- f.minimum))
plt.plot(np.minimum.accumulate(fX),linewidth=3,color='k',label='L-BFGS-B')

plt.xscale('log')
plt.yscale('log')
plt.ylabel("f(x)")
plt.xlabel("Number of function evaluations")
plt.legend()
plt.show()
