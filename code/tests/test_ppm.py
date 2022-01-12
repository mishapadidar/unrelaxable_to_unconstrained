import numpy as np
import sys
sys.path.append("../problems/")
from rosenbrock import Rosenbrock
from convex_quadratic import ConvexQuadratic
sys.path.append("../utils/")
sys.path.append("../generators/")
sys.path.append("../optim")
from gradient_descent import GD
from rescale import *
from project import project
from sigmoid import Sigmoid
sys.path.append("../../../NQN")
import NQN


## problem 1
#dim  = 10
#obj = Rosenbrock(dim)
#ub = np.copy(obj.ub)
#lb = np.copy(obj.lb)

## problem 2
dim = 12
Q = np.random.randn(dim,dim)
A = Q @ Q.T + 0.1*np.eye(dim)
lb = np.zeros(dim)
ub = np.ones(dim)
yopt = np.ones(dim) + np.random.randn(dim)
#yopt = np.random.uniform(lb,ub)
obj = ConvexQuadratic(lb,ub,A,yopt)

# initial pint
#x0 = (lb+ub)/2
x0 = np.random.uniform(lb,ub)

# solve
def dist_pen(xx):
  yy = np.copy(xx)
  return np.linalg.norm(yy - np.copy(project(yy,lb,ub)))

def proj_pen(xx):
  yy = np.copy(xx)
  return obj(project(yy,lb,ub)) + dist_pen(yy)

def proj_pen_grad(xx):
  """
  Gradient of the projected penalty
    f(project(x)) + ||x - project(x)||

  xx: 1d array, point
  return: subgradient g such that -g is a 
         descent direction.
  """
  bndry_tol=1e-14
  if np.all(xx<ub-bndry_tol) and np.all(xx>lb+bndry_tol):
    # interior point: grad = grad
    return obj.grad(xx)
  elif np.all(xx<=ub) and np.all(xx>=lb):
    # boundary point: negative grad = negative projected gradient
    return -(project(xx - obj.grad(xx),lb,ub) - xx)
    #Dpi = 0.0*np.zeros_like(xx)
    #idx_int = np.logical_and(xx<ub-bndry_tol,xx>lb+bndry_tol)
    #Dpi[idx_int] = 1.0
    #gg = Dpi*obj.grad(xx)
    #return gg
  else:
    # exterior point

    px = project(xx,lb,ub)

    ## indexes where a motion will not change the projection
    #idx_null = np.where(px != xx)[0]
    ## compute the projected gradient at the projected point
    ## return the negative negative gradient, so that GD gets a descent direction
    #gg = -(project(px - obj.grad(px),lb,ub) - px)
    #gg[idx_null] = 0.0
    #gg += (xx-px)/np.linalg.norm(xx-px)

    Dpi = 0.0*np.zeros_like(xx)
    idx_int = np.logical_and(xx<ub-bndry_tol,xx>lb+bndry_tol)
    Dpi[idx_int] = 1.0
    gg = np.copy(Dpi*obj.grad(project(xx,lb,ub)))
    gg += (xx-px)/np.linalg.norm(xx-px)

    #gg = (xx-px)/np.linalg.norm(xx-px)
    return np.copy(gg)

# solve
#xopt = GD(proj_pen,proj_pen_grad,x0,max_iter=500,gtol=1e-6,verbose=True)
res = NQN.fmin_l_bfgs_b(proj_pen, x0, proj_pen_grad, bounds=None, m=20, M=1, pgtol=1e-7, iprint=-1, maxfun=15000, maxiter=15000, callback=None, factr=0.)
xopt = res[0]

print("")
print("Optimal Value is ",obj(xopt))
print("Minima Found is ",xopt)
print("True Optima: ",obj.minimum)
print("PPM Gradient: ",proj_pen_grad(xopt))
print("Distance to Optima: ",np.linalg.norm(xopt- obj.minimum))
print("Norm Projected Gradient:",np.linalg.norm(project(xopt - obj.grad(xopt),lb,ub) - xopt))
