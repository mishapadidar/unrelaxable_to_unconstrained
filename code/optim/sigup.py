import numpy as np
from scipy.optimize import minimize
import sys
sys.path.append("../generators/")
sys.path.append("../utils/")
from sigmoid import Sigmoid
from rescale import *

"""
Optimization Algorithm using the sigma-update.
"""


def sigup(f,f_grad,lb,ub,y0,eps = 1e-8,delta=1e-10,gamma=10.0,n_solves=10,method='BFGS',verbose=False):
  """
  f: function handle, f:[lb,ub] -> R
  f_grad: the gradient of f
  lb,ub: lower and upper bound arrays
  y0: starting point within [lb,ub]
  eps: float, desired gradient tolerance
  gamma: float, increase paarameter for updating sigma
  n_solves: number of times to resolve
  """

  # only work on the unit cube
  y0 = to_unit_cube(y0,lb,ub)

  # initialize sigma
  sigma = np.ones(len(y0))

  np.set_printoptions(precision=16)
  for ii in range(n_solves):
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
    Delta = np.minimum(yopt,1-yopt)

    if verbose:
      print(f'{ii}) sigma: {sigma}')
      print(f'{ii}) xopt: {xopt}')
      print(f'{ii}) yopt: {yopt}')
      print(f'{ii}) grad: {f_grad(from_unit_cube(gen(xopt),lb,ub))}')
      print(f'{ii}) Delta: {Delta}')

    # exit because gradient is flat
    yopt_orig = from_unit_cube(yopt,lb,ub)
    if np.all(np.abs(f_grad(yopt_orig))<= eps): 
      break
    # exit because on boundary
    if np.any(Delta<delta):
      break

    # update sigma
    #sigma = gamma*sigma
    sigma = gamma*sigma/np.sqrt(Delta)
    # reset for next iteration
    y0 = np.copy(yopt)

  yopt = from_unit_cube(gen(xopt),lb,ub)
  return xopt,yopt

