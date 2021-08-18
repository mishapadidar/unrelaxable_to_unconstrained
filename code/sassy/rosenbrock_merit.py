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

# plot ft
fig = plt.figure(figsize=(10,6))
X,Y = np.meshgrid(np.linspace(lbt[0],ubt[0],100), np.linspace(lbt[1],ubt[1],100))
Z = np.zeros_like(X)
for ii,x in enumerate(X):
  for jj,y in enumerate(Y):
    Z[ii,jj] = ft([X[ii,jj],Y[ii,jj]])
plt.contour(X,Y,Z,100)
#plt.scatter(*sig.inv(to_unit_cube(f.minimum,lb,ub)),color='r')
plt.xticks([])
plt.yticks([])
#plt.title('$\\tilde{f}_s(x)$')
plt.show()
