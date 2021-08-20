import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append("../generators/")
sys.path.append("../problems/")
sys.path.append("../utils/")
from sigmoid import Sigmoid
from rosenbrock import Rosenbrock
from quadratic1 import Quadratic1
from rescale import *
import matplotlib

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath,bm}')
matplotlib.rcParams.update({'font.size': 16})

# Rosenbrock
dim  = 2
rosen    = Rosenbrock(dim)
lbr   = rosen.lb
ubr   = rosen.ub

# map to unit cube
def f(y):
  return rosen(from_unit_cube(y,lbr,ubr)-ubr+rosen.minimum)  # minimum at y = 1
lb   = np.zeros(dim)
ub   = np.ones(dim)


# plot on larger window
shift=0.5

# penalty param for penalty and reflection
mu = 10

# penalty
def proj(x):
  x = np.array(x)
  x[x > ub] = ub[x>ub]
  x[x < lb] = lb[x<lb]
  return x
fpen = lambda x: f(proj(x)) + mu*np.linalg.norm(x-proj(x))

# reflection
fref = lambda x: f(from_unit_cube(2*np.abs(to_unit_cube(x,lb,ub)/2 - np.floor(to_unit_cube(x,lb,ub)/2+0.5)),lb,ub)) + mu*np.linalg.norm(x-proj(x))

# dilation
sigma = 1
sig = Sigmoid(sigma)
ft = lambda xx: f(from_unit_cube(sig(xx),lb,ub))
# plotting bounds for merit; captures epsilon-tightened feasible region
eps = 1e-3
lbt = sig.inv(to_unit_cube(lbr+eps,lbr,ubr))
ubt = sig.inv(to_unit_cube(ubr-eps,lbr,ubr))

# plot f
fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(10,10))
X,Y = np.meshgrid(np.linspace(lb[0],ub[0],100), np.linspace(lb[1],ub[1],100))
Z = np.zeros_like(X)
for ii,x in enumerate(X):
  for jj,y in enumerate(Y):
    Z[ii,jj] = f(np.array([X[ii,jj],Y[ii,jj]]))
ax1.set_xlim(lb[0]-shift,ub[0]+shift)
ax1.set_ylim(lb[1]-shift,ub[1]+shift)
ax1.contour(X,Y,Z,100)
ax1.scatter(*np.ones(dim),marker='*',color='r')
ax1.set_title('$f(x)$')

# plot penalty
n_depth = 600
X,Y = np.meshgrid(np.linspace(lb[0]-2*shift,ub[0]+2*shift,n_depth), np.linspace(lb[1]-2*shift,ub[1]+2*shift,n_depth))
Z = np.zeros_like(X)
for ii,x in enumerate(X):
  for jj,y in enumerate(Y):
    Z[ii,jj] = fpen(np.array([X[ii,jj],Y[ii,jj]]))
contours = np.linspace(np.min(Z)+3*mu*shift,np.max(Z)-3*mu*shift,100)
contours = np.hstack((contours,np.linspace(np.max(Z)-3*mu*shift,np.max(Z)+mu*shift,10)))
contours = np.hstack((contours,np.linspace(7387,7387+3*mu*shift,10)))
contours = np.hstack((contours,np.linspace(1677,1677+3*mu*shift,10)))
contours = np.hstack((contours,np.array([5,40,120,180])))
contours = np.sort(np.unique(contours))
ax2.set_xlim(lb[0]-shift,ub[0]+shift)
ax2.set_ylim(lb[1]-shift,ub[1]+shift)
ax2.contour(X,Y,Z,levels=contours)
ax2.scatter(*np.ones(dim),marker='*',color='r')
ax2.set_title('Penalty')

# plot relfection
X,Y = np.meshgrid(np.linspace(lb[0]-shift,ub[0]+shift,100), np.linspace(lb[1]-shift,ub[1]+shift,100))
Z = np.zeros_like(X)
for ii,x in enumerate(X):
  for jj,y in enumerate(Y):
    Z[ii,jj] = fref(np.array([X[ii,jj],Y[ii,jj]]))
contours = np.linspace(np.min(Z),np.max(Z),200)
contours = np.hstack(([ft(0.765*np.ones(dim))],contours))
contours = np.sort(np.unique(contours))
ax3.contour(X,Y,Z,levels=contours)
ax3.scatter(*np.ones(dim),marker='*',color='r')
ax3.set_title('Reflection')

# plot ft
X,Y = np.meshgrid(np.linspace(lbt[0],ubt[0],500), np.linspace(lbt[1],ubt[1],500))
Z = np.zeros_like(X)
for ii,x in enumerate(X):
  for jj,y in enumerate(Y):
    Z[ii,jj] = ft(np.array([X[ii,jj],Y[ii,jj]]))
contours = np.linspace(np.min(Z),np.max(Z),150)
contours = np.hstack(([ft(3*np.ones(dim)),ft(5.5*np.ones(dim))],contours))
contours = np.sort(np.unique(contours))
ax4.contour(X,Y,Z,levels=contours)
#ax4.scatter(*sig.inv(np.ones(dim)),marker='*',color='r')
ax4.arrow(5.7, 5.7, 1.0, 1.0, head_width=0.3, head_length=0.5, fc='r', ec='r')
ax4.set_title('Dilation')

fig.tight_layout(pad=2.0)
plt.show()

