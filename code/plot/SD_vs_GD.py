import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib
import sys
sys.path.append("../generators/")
sys.path.append("../problems/")
sys.path.append("../utils/")
sys.path.append("../optim/")
from eval_wrapper import *
from rescale import *
from gradient_descent import GD
from steepest_descent import SD
from sigmoid import Sigmoid
from rosenbrock import Rosenbrock
from convex_quadratic import ConvexQuadratic

plt.rc('text.latex', preamble=r'\usepackage{amsmath,bm}')
matplotlib.rcParams.update({'font.size': 16})

## quadratic
#dim  = 2
#lb = np.zeros(dim)
#ub = np.ones(dim)
##yopt = np.array([0.7,0.7])
#yopt = np.array([1.2,0.95])
#theta = np.pi/24
#R = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
#Rinv = np.linalg.inv(R)
#A = np.diag(np.array([1,10]))
#f = ConvexQuadratic(lb,ub,R@A@Rinv,yopt)

dim = 2
f = Rosenbrock(dim)
lb = f.lb
ub = f.ub

# merit function
sig = Sigmoid(0.001)
ft = lambda xx: f(from_unit_cube(sig(xx),lb,ub))
ft_grad = lambda xx: sig.jac(xx) @ np.diag(ub-lb) @ f.grad(from_unit_cube(sig(xx),lb,ub))

# initial point
#Y0 = [np.array([0.02,0.45]),np.array([0.999,0.9])]
Y0 = [np.array([1.5,.3]),np.array([-1.8,0.6])]

# plot
#fig,ax = plt.subplots(figsize=(10,8))
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,4))


def linesearch(xx,dd,a0,nn):
  Y = [ft(xx-(0.8**ii)*a0*dd) for ii in range(nn)]
  am = np.argmin(Y)
  return xx - (0.8**am)*a0*dd

"""
# plot the arrows at the first point
"""
y0 = np.array([1.5,.3])
x0 = sig.inv(to_unit_cube(y0,lb,ub))

# plot steepest descent step
J = np.diag(sig.jac(x0))
#x1 = np.copy(linesearch(x0,(1.0/J/J)*ft_grad(x0),1.0,30))
x1 = np.copy(linesearch(x0,(1.0/J/J)*ft_grad(x0),0.1,50))
step = x1-x0
ax2.arrow(x0[0],x0[1],step[0],step[1],color='C0',width=80.0,lw=3,head_width=250,head_length=280,head_starts_at_zero=False)
y1 = from_unit_cube(sig(x1),lb,ub)
step = y1-y0
ax1.arrow(y0[0],y0[1],step[0],step[1],color='C0',width=0.04,lw=3,head_starts_at_zero=False)

# plot gradient descent step
#x1 = np.copy(linesearch(x0,ft_grad(x0),5e6,30))
x1 = np.copy(linesearch(x0,ft_grad(x0),5e5,50))
step = x1-x0
ax2.arrow(x0[0],x0[1],step[0],step[1],color='C1',width=80.0,lw=1.5,head_width=250,head_length=280,head_starts_at_zero=False)
y1 = from_unit_cube(sig(x1),lb,ub)
step = y1-y0
ax1.arrow(y0[0],y0[1],step[0],step[1],color='C1',width=0.04,lw=1.5,head_starts_at_zero=False)

"""
# plot the arrows at the second point
"""
y0 = np.array([-1.8,0.6])
x0 = sig.inv(to_unit_cube(y0,lb,ub))

# plot steepest descent step
J = np.diag(sig.jac(x0))
#x1 = np.copy(linesearch(x0,(1.0/J/J)*ft_grad(x0),1.0,30))
x1 = np.copy(linesearch(x0,(1.0/J/J)*ft_grad(x0),0.1,50))
step = x1-x0
ax2.arrow(x0[0],x0[1],step[0],step[1],color='C0',width=90.0,lw=3,head_width=250,head_length=250,head_starts_at_zero=False)
y1 = from_unit_cube(sig(x1),lb,ub)
step = y1-y0
ax1.arrow(y0[0],y0[1],step[0],step[1],color='C0',width=0.04,lw=3,head_starts_at_zero=False)

# plot gradient descent step
#x1 = np.copy(linesearch(x0,ft_grad(x0),5e6,30))
x1 = np.copy(linesearch(x0,ft_grad(x0),5e5,50))
step = x1-x0
ax2.arrow(x0[0],x0[1],step[0],step[1],color='C1',width=90.0,lw=1.5,head_width=250,head_length=250,head_starts_at_zero=False)
y1 = from_unit_cube(sig(x1),lb,ub)
step = y1-y0
ax1.arrow(y0[0],y0[1],step[0],step[1],color='C1',width=0.03,lw=1.5,head_starts_at_zero=False)

"""
make the contours
"""
# plot f
X,Y = np.meshgrid(np.linspace(lb[0],ub[0],200), np.linspace(lb[1],ub[1],200))
#ax.contour(X,Y,f([X,Y]),100)
Z = np.zeros_like(X)
for ii,x in enumerate(X):
  for jj,y in enumerate(Y):
    Z[ii,jj] = f(np.array([X[ii,jj],Y[ii,jj]]))
levels=[3.0,50]
levels=np.append(levels,np.linspace(0.0,np.max(Z),20))
levels = np.sort(np.unique(levels))
ax1.contour(X,Y,Z,levels=levels,alpha=0.7)
#ax1.set_xlim(-0.1,1.1)
#ax1.set_ylim(-0.1,1.1)
ax1.set_xlim(lb[0]-0.2,ub[0]+0.2)
ax1.set_ylim(lb[1]-0.2,ub[1]+0.2)
ax1.scatter(*f.minimum,marker='*',s=200,color='r',edgecolor='k',zorder=10)
ax1.set_xticks([])
ax1.set_yticks([])
# ax1.axis('off')  # command for hiding the outer box on the left plot.
# Create a Rectangle patch
rect = Rectangle((lb[0],lb[1]),(ub-lb)[0],(ub-lb)[1],linewidth=3,edgecolor='k',facecolor='none')
ax1.add_patch(rect)

# plotting bounds in x-domain
x_lb = sig.inv(np.array([0.01,0.01]))
x_ub = sig.inv(np.array([0.99,0.99]))
X,Y = np.meshgrid(np.linspace(x_lb[0],x_ub[0],200), np.linspace(x_lb[1],x_ub[1],200))
#ax.contour(X,Y,f([X,Y]),100)
Z = np.zeros_like(X)
# plot merit function
for ii,x in enumerate(X):
  for jj,y in enumerate(Y):
    Z[ii,jj] = ft(np.array([X[ii,jj],Y[ii,jj]]))
levels=[3.0,50]
levels=np.append(levels,np.linspace(0.0,np.max(Z),20))
levels = np.sort(np.unique(levels))
ax2.contour(X,Y,Z,levels=levels,alpha=0.7)
ax2.scatter(*sig.inv(to_unit_cube(f.minimum,lb,ub)),marker='*',s=200,color='r',edgecolor='k',zorder=10)
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_xlim(x_lb[0],x_ub[0])
ax2.set_ylim(x_lb[1],x_ub[1])
#ax.set_ylim(-0.1,1.1)
# plt.show()
plt.savefig('fig4.png',bbox_inches='tight',dpi=300)
