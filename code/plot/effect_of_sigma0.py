import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import sys
sys.path.append("../generators/")
sys.path.append("../problems/")
sys.path.append("../utils/")
sys.path.append("../optim/")
from sigmoid import Sigmoid
from rosenbrock import Rosenbrock
from convex_quadratic import ConvexQuadratic
from sigup import *

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\usepackage{amsmath,bm}')
matplotlib.rcParams.update({'font.size': 16})

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

fig,[ax1,ax2] = plt.subplots(1,2,figsize=(12,6))
markers = ['-o','-^','-s']
# optimize
#sigmas = [0.01,0.1,1.0,10.0,100.0,1000.0]
sigmas = [0.001,0.1,100.0]
for ii,sigma0 in enumerate(sigmas):
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
  ax1.plot(X[:,0],X[:,1],markers[ii],markersize=10,linewidth=3,label=f'$\sigma_0: {sigma0}$')
  ax2.plot(range(1,len(fX)+1),fX,markers[ii],markersize=10,linewidth=3,label=f'$\sigma_0: {sigma0}$')

# plot f
ax1.scatter(*f.minimum,color='r',marker='*',edgecolors='black',s=300,label='minima',zorder=10)
X,Y = np.meshgrid(np.linspace(lb[0],ub[0],100), np.linspace(lb[1],ub[1],100))
#ax.contour(X,Y,f([X,Y]),100)
Z = np.zeros_like(X)
for ii,x in enumerate(X):
  for jj,y in enumerate(Y):
    Z[ii,jj] = f(np.array([X[ii,jj],Y[ii,jj]]))
ax1.contour(X,Y,Z,20)
ax1.set_xlim(-0.1,1.1)
ax1.set_ylim(-0.1,1.1)
ax1.set_xlabel(r'$y_1$')
ax1.set_ylabel(r'$y_2$')
ax1.legend()
#ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel('Number of Evaluations')
ax2.set_ylabel(r'$f\,(S(\bm{x}))$')
ax1.legend(loc=9,mode='expand',bbox_to_anchor=(-0.02, 1.02, 1.2, .102),ncol=4,prop={'size': 13})
ax1.set_xticks([0.0,0.5,1.0])
ax1.set_yticks([0.0,0.5,1.0])
ax2.set_ylim(0.8,150)
ax2.grid()
plt.rc('grid', linestyle="-", color='black')
# Create a Rectangle patch
rect = Rectangle((0,0),1,1,linewidth=3,edgecolor='k',facecolor='none')
ax1.add_patch(rect)
fig.tight_layout()
# plt.show()
plt.savefig('fig3.png',bbox_inches='tight',dpi=300)

