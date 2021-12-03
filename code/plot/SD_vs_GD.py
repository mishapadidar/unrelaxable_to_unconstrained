import numpy as np
import matplotlib.pyplot as plt
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

# Rosenbrock
dim  = 2
lb = np.zeros(dim)
ub = np.ones(dim)
yopt = np.array([0.7,0.7])
theta = np.pi/24
R = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
Rinv = np.linalg.inv(R)
A = np.diag(np.array([1,10]))
f = ConvexQuadratic(lb,ub,R@A@Rinv,yopt)

# merit function
sig = Sigmoid(0.001)
ft = lambda xx: f(from_unit_cube(sig(xx),lb,ub))
ft_grad = lambda xx: sig.jac(xx) @ np.diag(ub-lb) @ f.grad(from_unit_cube(sig(xx),lb,ub))

# initial point
y0 = np.array([0.02,0.45])
x0 = sig.inv(y0)

# plot
#fig,ax = plt.subplots(figsize=(10,8))
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,4))



def linesearch(xx,dd,a0,nn):
  Y = [ft(xx-(0.8**ii)*a0*dd) for ii in range(nn)]
  am = np.argmin(Y)
  return xx - (0.8**am)*a0*dd

# plot steepest descent step
J = np.diag(sig.jac(x0))
x1 = np.copy(linesearch(x0,(1.0/J/J)*ft_grad(x0),1.0,20))
step = x1-x0
ax2.arrow(x0[0],x0[1],step[0],step[1],color='C0',width=20.0,head_width=100,head_length=220)
y1 = sig(x1)
step = y1-y0
ax1.arrow(y0[0],y0[1],step[0],step[1],color='C0',width=0.01)

# plot gradient descent step
x1 = np.copy(linesearch(x0,ft_grad(x0),5e6,30))
step = x1-x0
ax2.arrow(x0[0],x0[1],step[0],step[1],color='C1',width=60.0,head_width=200,head_length=100)
y1 = sig(x1)
step = y1-y0
ax1.arrow(y0[0],y0[1],step[0],step[1],color='C1',width=0.01)

# plot f
X,Y = np.meshgrid(np.linspace(lb[0],ub[0],100), np.linspace(lb[1],ub[1],100))
#ax.contour(X,Y,f([X,Y]),100)
Z = np.zeros_like(X)
for ii,x in enumerate(X):
  for jj,y in enumerate(Y):
    Z[ii,jj] = f(np.array([X[ii,jj],Y[ii,jj]]))
ax1.contour(X,Y,Z,levels=40)
ax1.set_xlim(-0.1,1.1)
ax1.set_ylim(-0.1,1.1)
ax1.scatter(*f.minimum,marker='*',color='r')
ax1.set_xticks([])
ax1.set_yticks([])

# plotting bounds in x-domain
x_lb = sig.inv(np.array([0.005,0.35]))
x_ub = sig.inv(np.array([0.99,0.8]))
X,Y = np.meshgrid(np.linspace(x_lb[0],x_ub[0],100), np.linspace(x_lb[1],x_ub[1],100))
#ax.contour(X,Y,f([X,Y]),100)
Z = np.zeros_like(X)
# plot merit function
for ii,x in enumerate(X):
  for jj,y in enumerate(Y):
    Z[ii,jj] = ft(np.array([X[ii,jj],Y[ii,jj]]))
ax2.contour(X,Y,Z,levels=30)
ax2.scatter(*sig.inv(f.minimum),marker='*',color='r')
ax2.set_xticks([])
ax2.set_yticks([])
#ax.set_xlim(-0.1,1.1)
#ax.set_ylim(-0.1,1.1)
plt.show()
