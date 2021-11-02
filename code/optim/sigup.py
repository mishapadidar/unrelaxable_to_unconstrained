import numpy as np
from scipy.optimize import minimize,Bounds
import sys
sys.path.append("../generators/")
sys.path.append("../utils/")
from compute_lagrange import compute_lagrange
from sigmoid import Sigmoid
from rescale import *

"""
Optimization Algorithm using the sigma-update.
"""


def sigup(f,f_grad,lb,ub,y0,sigma0 = 1.0,eps = 1e-8,delta=1e-10,gamma=10.0,method='BFGS',verbose=False):
  """
  f: function handle, f:[lb,ub] -> R
  f_grad: the gradient of f
  lb,ub: lower and upper bound arrays
  y0: starting point within [lb,ub]
  eps: float, desired gradient tolerance
  gamma: float, increase paarameter for updating sigma
  n_solves: number of times to resolve
  """
  np.set_printoptions(precision=16)
 
  # update caps
  cap_sigma = 1e14 # maximum sigma
  cap_eta   = 1e-10 # minimum eta: to push away from the boundary

  # problem dimension
  dim = len(y0)

  # only work on the unit cube
  y0 = to_unit_cube(y0,lb,ub)

  # initialize sigma
  sigma = sigma0*np.ones(len(y0))

  # stopping criteria
  kkt = False

  while kkt == False:
    # generator
    gen = Sigmoid(sigma=sigma)
    # compute x0
    x0 = gen.inv(y0)
    # merit
    ft = lambda xx: f(from_unit_cube(gen(xx),lb,ub))
    ft_jac = lambda xx: gen.jac(xx) @ np.diag(ub-lb) @ f_grad(from_unit_cube(gen(xx),lb,ub))
    # optimize
    res = minimize(ft,x0,jac=ft_jac,method=method,options={'gtol':eps})
    xopt = res.x
    # compute y*
    yopt = gen(xopt)
    # compute distance to boundary
    eta = np.minimum(yopt,1-yopt)
    # zopt = y^* in original domain [lb,ub]
    zopt = from_unit_cube(yopt,lb,ub)
    g_z  = f_grad(zopt)

    if verbose:
      print('')
      print(f'sigma: {sigma}')
      print(f'eta: {eta}')
      print(f'xopt: {xopt}')
      print(f'yopt: {yopt}')
      print(f'grad_f: {g_z}')

    # exit because gradient is flat
    if np.all(np.abs(g_z)< eps): # exit if gradient is flat
      return zopt
    else: # check the KKT conditions
      lam,mu = compute_lagrange(zopt,g_z,lb,ub)
      kkt = np.all(np.abs(g_z + lam - mu)< eps) and np.all(np.abs(lam*(zopt-lb)) < eps) and np.all(np.abs(mu*(ub-zopt))< eps)
      if kkt: 
        print('')
        print('stationary')
        print(np.abs(g_z + lam - mu))
        print('complementary slackenss lambda*y')
        print(np.abs(lam*(zopt-lb)))
        print('complementary slackenss mu*(1-y)')
        print(np.abs(mu*(ub-zopt)))
        return zopt

    if np.all(sigma > cap_sigma-1):
      print("Breaking: max sigma reached")
      return zopt

    # update sigma
    eta[eta<cap_eta] = cap_eta # cap the update size
    sigma = gamma*sigma/np.sqrt(eta)
    sigma = np.minimum(sigma,cap_sigma)
    # reset for next iteration
    y0 = np.copy(yopt)

  return zopt


#def compute_lagrange(y,g):
#  """
#  Solve the QP to check approximate stationarity
#  y: point y
#  g: grad f(y)
#  """
#  # sizes
#  dim_y = len(y)
#  n_con = 2*dim_y
#
#  # setup the QP
#  I = np.eye(dim_y)
#  M = np.hstack((I + np.outer(y,y), -I))
#  M2 = np.hstack((-I, I + np.outer(1-y,1-y)))
#  M = np.vstack((M,M2))
#  b = np.hstack((g,-g))
#  f = lambda z: 0.5 * z @ M @ z + z @ b
#  jac = lambda z: M @ z + b
#
#  # solve
#  bounds = Bounds(-np.inf*np.ones(n_con),np.zeros(n_con))
#  res = minimize(f,np.ones(n_con),jac=jac,method="L-BFGS-B",bounds=bounds,options={'gtol':1e-8})
#  z = res.x
#  lam = z[:dim_y]
#  mu = z[dim_y:]
#
#  return lam,mu

