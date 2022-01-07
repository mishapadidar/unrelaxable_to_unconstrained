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


#np.random.seed(0)

# problem 1
dim  = 10
#obj = Rosenbrock(dim)

# problem 2
#dim = 2
yopt = np.ones(dim) + np.random.randn(dim)
#yopt = np.ones(dim) + np.array([-1e-7,1e-1])
#yopt = np.ones(dim) - np.array([1e-7,1e-1])
#yopt = np.ones(dim) - np.array([0.5,0.0])
#yopt = np.ones(dim) + np.array([0.5,0.0])
#A = np.diag(np.array([100,2]))
Q = np.random.randn(dim,dim)
A = Q @ Q.T + 0.1*np.eye(dim)
lb = np.zeros(dim)
ub = np.ones(dim)
obj = ConvexQuadratic(lb,ub,A,yopt)


# bounds
ub = np.copy(obj.ub)
lb = np.copy(obj.lb)

# initial pint
x0 = (lb+ub)/2

# solve
dist_pen = lambda xx: np.linalg.norm(xx - project(xx,lb,ub))
proj_pen = lambda xx: obj(project(xx,lb,ub)) + dist_pen(xx)

#def proj_pen_grad(xx):
#  """
#  Gradient of the projected penalty
#    f(project(x)) + ||x - project(x)||
#
#  The second term has derivative 
#    . (x-project(x))/||x-project(x)|| if x not feasible
#    . 0 if x is feasible 
#  The first term has derivative
#    . grad(x) if x is interior point 
#    . project(x-grad(x))-x (projected gradient) if x is on the boundary
#    . if x is exterior:
#      we compute the set of indexes where project(x) != x. Motion
#      along these indexes does not change the projection.
#      So these derivatives are zero. In the other indexes the derivatives
#      are the projected derivatives at the projected point
#      project(project(x) - grad(project(x))) - project(x)
#
#  xx: 1d array, point
#  return: subgradient g such that -g is a 
#         descent direction.
#  """
#  if np.all(xx<ub) and np.all(xx>lb):
#    # interior point: grad = grad
#    return obj.grad(xx)
#  elif np.all(xx<=ub) and np.all(xx>=lb):
#    # boundary point: grad = projected gradient
#    return project(xx - obj.grad(xx),lb,ub) - xx
#  else:
#    # exterior point
#    px = project(xx,lb,ub)
#    # indexes where a motion will not change the projection
#    idx_null = np.where(px != xx)[0]
#    # compute the projected gradient at the projected point
#    gg = project(px - obj.grad(px),lb,ub) - px
#    gg[idx_null] = 0.0
#    gg += (xx-px)/np.linalg.norm(xx-px)
#    return gg

def proj_pen_grad(xx):
  """
  Gradient of the projected penalty
    f(project(x)) + ||x - project(x)||

  The second term has derivative 
    . (x-project(x))/||x-project(x)|| if x not feasible
    . 0 if x is feasible 
  The first term has derivative
    . grad(x) if x is interior point 
    . project(x-grad(x))-x (projected gradient) if x is on the boundary
    . if x is exterior:
      we compute the set of indexes where project(x) != x. Motion
      along these indexes does not change the projection.
      So these derivatives are zero. In the other indexes the derivatives
      are the projected derivatives at the projected point
      project(project(x) - grad(project(x))) - project(x)

  xx: 1d array, point
  return: subgradient g such that -g is a 
         descent direction.
  """
  if np.all(xx<ub) and np.all(xx>lb):
    # interior point: grad = grad
    return obj.grad(xx)
  elif np.all(xx<=ub) and np.all(xx>=lb):
    # boundary point: negative grad = negative projected gradient
    return -(project(xx - obj.grad(xx),lb,ub) - xx)
  else:
    # exterior point
    px = project(xx,lb,ub)
    # indexes where a motion will not change the projection
    idx_null = np.where(px != xx)[0]
    # compute the projected gradient at the projected point
    gg = -(project(px - obj.grad(px),lb,ub) - px)
    gg[idx_null] = 0.0
    gg += (xx-px)/np.linalg.norm(xx-px)
    gg = (xx-px)/np.linalg.norm(xx-px)
    # return the negative negative gradient, so that GD gets a descent direction
    return gg
xopt = GD(proj_pen,proj_pen_grad,x0,max_iter=1000,gtol=1e-6,verbose=False)
# project xopt
xopt = project(xopt,lb,ub)
print(xopt)
print(np.linalg.norm(proj_pen_grad(xopt)))

print("Optimal Value is ",obj(xopt))
print("Minima Found is ",xopt)
print("Distance to Optima: ",np.linalg.norm(xopt- obj.minimum))
