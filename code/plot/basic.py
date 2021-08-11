import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../generators/")
sys.path.append("../problems/")
sys.path.append("../utils/")
from sigmoid import Sigmoid
from rosenbrock import Rosenbrock
from rescale import *

# Rosenbrock
dim  = 2
f    = Rosenbrock(dim)
lb   = f.lb
ub   = f.ub
# generator
sigma = 1
sig = Sigmoid(sigma)
# merit
ft = lambda xx: f(from_unit_cube(sig(xx),lb,ub))
# plotting bounds for merit; captures epsilon-tightened feasible region
eps = 1e-2
lbt = sig.inv(to_unit_cube(lb+eps,lb,ub))
ubt = sig.inv(to_unit_cube(ub-eps,lb,ub))

# plot f
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,4))
X,Y = np.meshgrid(np.linspace(lb[0],ub[0],100), np.linspace(lb[1],ub[1],100))
ax1.contour(X,Y,f([X,Y]),100)
ax1.scatter(*f.minimum,color='r')
ax1.set_title('$f(x)$')

# plot ft
X,Y = np.meshgrid(np.linspace(lbt[0],ubt[0],100), np.linspace(lbt[1],ubt[1],100))
Z = np.zeros_like(X)
for ii,x in enumerate(X):
  for jj,y in enumerate(Y):
    Z[ii,jj] = ft([X[ii,jj],Y[ii,jj]])
ax2.contour(X,Y,Z,100)
ax2.scatter(*sig.inv(to_unit_cube(f.minimum,lb,ub)),color='r')
ax2.set_title('$\\tilde{f}_s(x)$')
plt.show()
