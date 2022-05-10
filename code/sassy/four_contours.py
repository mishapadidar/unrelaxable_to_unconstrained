import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
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
plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\usepackage{amsmath,bm}')
matplotlib.rcParams.update({'font.size': 16})

# Rosenbrock
dim  = 2
rosen    = Rosenbrock(dim)
lbr   = rosen.lb
ubr   = rosen.ub

# map to unit cube
def f(x):
  return rosen(from_unit_cube(x,lbr,ubr))
lb   = np.zeros(dim)
ub   = np.ones(dim)

# plot on larger window
shift=0.5

# penalty param for penalty and reflection
mu = 10
mu_pen = 1000

# penalty
def proj(x):
  x = np.array(x)
  x[x > ub] = ub[x>ub]
  x[x < lb] = lb[x<lb]
  return x
fpen = lambda x: f(proj(x)) + mu_pen*np.linalg.norm(x-proj(x))

# reflection
fref = lambda x: f(from_unit_cube(2*np.abs(to_unit_cube(x,lb,ub)/2 - np.floor(to_unit_cube(x,lb,ub)/2+0.5)),lb,ub)) #+ mu*np.linalg.norm(x-proj(x))

# dilation
sigma = 1.0
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
levels=[3.0,50]
levels=np.append(levels,np.linspace(0.0,np.max(Z),15))
levels = np.sort(np.unique(levels))
ax1.set_xticks([0,1])
ax1.set_yticks([0,1])
ax1.contour(X,Y,Z,levels=levels)
ax1.scatter(*to_unit_cube(rosen.minimum,lbr,ubr),marker='*',s=80,color='r')
ax1.set_title('$f(x)$')
# Create a Rectangle patch
rect = Rectangle((lb[0],lb[1]),(ub-lb)[0],(ub-lb)[1],linewidth=3,edgecolor='k',facecolor='none',zorder=10)
ax1.add_patch(rect)

# plot penalty
n_depth = 600
X,Y = np.meshgrid(np.linspace(lb[0]-2*shift,ub[0]+2*shift,n_depth), np.linspace(lb[1]-2*shift,ub[1]+2*shift,n_depth))
Z = np.zeros_like(X)
for ii,x in enumerate(X):
  for jj,y in enumerate(Y):
    Z[ii,jj] = fpen(np.array([X[ii,jj],Y[ii,jj]]))
levels=[3.0,50]
levels=np.append(levels,np.linspace(0.0,np.max(Z),20))
#levels = np.linspace(np.min(Z),np.max(Z)+mu*shift,80)
#levels = np.hstack((levels,np.linspace(np.max(Z)-5*mu*shift,np.max(Z)+mu*shift,30)))
#levels = np.hstack((levels,[461,463,464,465,466,467,468,471,472,473,474,475,476]))
levels = np.sort(np.unique(levels))
ax2.set_xlim(lb[0]-shift,ub[0]+shift)
ax2.set_ylim(lb[1]-shift,ub[1]+shift)
ax2.set_xticks([0,1])
ax2.set_yticks([0,1])
ax2.contour(X,Y,Z,levels=levels)
ax2.scatter(*to_unit_cube(rosen.minimum,lbr,ubr),marker='*',s=80,color='r')
ax2.set_title('PPM')
# Create a Rectangle patch
rect = Rectangle((lb[0],lb[1]),(ub-lb)[0],(ub-lb)[1],linewidth=3,edgecolor='k',facecolor='none',zorder=10)
ax2.add_patch(rect)

# plot relfection
X,Y = np.meshgrid(np.linspace(lb[0]-shift,ub[0]+shift,200), np.linspace(lb[1]-shift,ub[1]+shift,200))
Z = np.zeros_like(X)
for ii,x in enumerate(X):
  for jj,y in enumerate(Y):
    Z[ii,jj] = fref(np.array([X[ii,jj],Y[ii,jj]]))
levels=[3.0,50]
levels=np.append(levels,np.linspace(0.0,np.max(Z),15))
levels = np.sort(np.unique(levels))
ax3.contour(X,Y,Z,levels=levels)
ax3.scatter(*to_unit_cube(rosen.minimum,lbr,ubr),marker='*',s=80,color='r')
ref_min = to_unit_cube(rosen.minimum,lbr,ubr)
ref_min2 = np.array([1.0,ref_min[1]])+ (np.array([1.0,ref_min[1]]) - ref_min)
ref_min3 = np.array([ref_min[0],1.0])+ (np.array([ref_min[1],1.0]) - ref_min)
ref_min4 = np.ones(2)+ (np.ones(2) - ref_min)
ax3.scatter(*ref_min2,marker='*',s=80,color='r')
ax3.scatter(*ref_min3,marker='*',s=80,color='r')
ax3.scatter(*ref_min4,marker='*',s=80,color='r')
ax3.set_xticks([0,1])
ax3.set_yticks([0,1])
ax3.set_title('Reflection')
# Create a Rectangle patch
rect = Rectangle((lb[0],lb[1]),(ub-lb)[0],(ub-lb)[1],linewidth=3,edgecolor='k',facecolor='none',zorder=10)
ax3.add_patch(rect)

# plot ft
X,Y = np.meshgrid(np.linspace(lbt[0],ubt[0],200), np.linspace(lbt[1],ubt[1],200))
Z = np.zeros_like(X)
for ii,x in enumerate(X):
  for jj,y in enumerate(Y):
    Z[ii,jj] = ft(np.array([X[ii,jj],Y[ii,jj]]))
levels=[3.0,50]
levels=np.append(levels,np.linspace(0.0,np.max(Z),15))
levels = np.sort(np.unique(levels))
ax4.contour(X,Y,Z,levels=levels)
ax4.scatter(*sig.inv(to_unit_cube(rosen.minimum,lbr,ubr)),marker='*',s=80,color='r')
ax4.set_xticks([-5,0,5])
ax4.set_yticks([-5,0,5])
ax4.set_title('Sigmoidal')

# fig.tight_layout(pad=2.0)
# plt.show()
plt.savefig('fig1.png',bbox_inches='tight',dpi=300)

