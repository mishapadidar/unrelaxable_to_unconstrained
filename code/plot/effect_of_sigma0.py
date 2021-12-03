import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../generators/")
sys.path.append("../problems/")
sys.path.append("../utils/")
sys.path.append("../optim/")
from sigmoid import Sigmoid
from rosenbrock import Rosenbrock
from convex_quadratic import ConvexQuadratic
from sigup import *

# Rosenbrock
dim  = 2
#f    = Rosenbrock(dim)
#lb   = f.lb
#ub   = f.ub
lb = np.zeros(dim)
ub = np.ones(dim)
yopt = np.array([1.1,1.1])
A = np.diag(np.array([100,2]))
f = ConvexQuadratic(lb,ub,A,yopt)

# optimizer params
y0 = np.array([0.2,0.3])
gamma = 1.0
eps = 1e-6
delta = eps
solve_method = "scipy"
update_method = "adaptive"
verbose=True

fig,ax = plt.subplots(figsize=(10,8))
# optimize
#sigmas = [0.01,0.1,1.0,10.0,100.0,1000.0]
sigmas = [0.001,0.1,100.0]
for sigma0 in sigmas:
  print(f"\nsigup, sigma0 = {sigma0}")
  sigup = SIGUP(f,f.grad,f.lb,f.ub,y0,eps = eps,delta=delta,gamma=gamma,sigma0=sigma0,solve_method=solve_method,
                update_method=update_method)
  z = sigup.solve(verbose=verbose)
  X = sigup.X
  fX = sigup.fX
  updates = sigup.updates
  print("Optimal Value is ",f(z))
  print("Minima Found is ",z)
  print("Distance to Optima: ",np.linalg.norm(z- f.minimum))
  ax.plot(X[:,0],X[:,1],'-o',linewidth=3,label=f'$\sigma_0: {sigma0}$')

# plot f
X,Y = np.meshgrid(np.linspace(lb[0],ub[0],100), np.linspace(lb[1],ub[1],100))
#ax.contour(X,Y,f([X,Y]),100)
Z = np.zeros_like(X)
for ii,x in enumerate(X):
  for jj,y in enumerate(Y):
    Z[ii,jj] = f(np.array([X[ii,jj],Y[ii,jj]]))
ax.contour(X,Y,Z,100)
ax.scatter(*f.minimum,color='r')
ax.set_xlim(-0.1,1.1)
ax.set_ylim(-0.1,1.1)
plt.legend()
plt.show()
