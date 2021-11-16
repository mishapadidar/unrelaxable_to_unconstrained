import numpy as np
from scipy.optimize import minimize,Bounds
import nlopt
import sys
sys.path.append("../generators/")
sys.path.append("../utils/")
#from compute_lagrange import compute_lagrange
from check_kkt import compute_kkt_tol
from sigmoid import Sigmoid
from rescale import *


class SIGUP():

  def __init__(self,func,f_grad,lb,ub,z0,sigma0 = 1.0,eps = 1e-8,delta=1e-10,gamma=10.0,solve_method="nlopt",
    update_method="adaptive"):
    """
    Sigup is an optimization method for solving nonlinear bound constrained problems
      min_x f(x) s.t lb < x< ub
    
    func: objective handle, for minimization
    f_grad: gradient of objective
    lb, ub: finite lower and upper bound arrays
    z0: initial starting point within the open set (lb,ub)
    sigma0: initial sigma value, must be positive
    eps: desired kkt tolerance
    delta: kkt tolerance at each solve
    gamma: >1, sigma increase parameter
    solve_method: either "nlopt" or "scipy", subproblem solver
    update_method: either "adaptive" or "exp"
    """
    assert np.all(lb < z0) and np.all(z0 < ub), "bad initialization"
    assert np.all(np.isfinite(lb)) and np.all(np.isfinite(ub)), "need finite bounds"
    assert solve_method in ["nlopt","scipy"], "improper solve method"
    assert update_method in ["adaptive","exp"], "improper update method"
    
    self.func = func
    self.f_grad = f_grad
    self.lb = lb
    self.ub = ub
    self.z0 = z0
    self.sigma0 = sigma0
    self.eps = eps
    self.delta = delta
    self.gamma = gamma
    self.solve_method = solve_method
    self.update_method = update_method
    self.dim = len(z0)

    # update caps
    self.cap_sigma = 1e14 # maximum sigma
    self.cap_eta   = 1e-16 # minimum eta: to push away from the boundary
   
    # save the function values
    self.X = np.zeros((0,self.dim))
    self.fX = np.zeros(0)
    # save the indexes of the sigma updates
    self.updates = [0]

  def fwrap(self,xx):
    """
    wrap the objective so we can save the values
    """ 
    ff = self.func(xx)
    self.X = np.append(self.X,[xx],axis=0)
    self.fX = np.append(self.fX,ff)
    return ff

  def solve(self,verbose=False):
    """
    f: function handle, f:[lb,ub] -> R
    f_grad: the gradient of f
    lb,ub: lower and upper bound arrays
    z0: starting point within [lb,ub]
    eps: float, desired gradient tolerance
    gamma: float, increase paarameter for updating sigma
    n_solves: number of times to resolve
    """
    if verbose == True:
      np.set_printoptions(precision=16)
   
    # load in bounds... we use them alot
    lb = np.copy(self.lb)
    ub = np.copy(self.ub)

    # only work on the unit cube
    y0 = to_unit_cube(self.z0,lb,ub)
  
    # initialize sigma
    if self.sigma0 == 'auto':
      g0 = f_grad(self.z0)
      sigma = np.ones(len(y0))
      idx = g0 != 0.0
      sigma[idx] = 1.0/(y0[idx]*(1-y0[idx])*np.abs(g0[idx]))
    else:
      sigma = self.sigma0*np.ones(len(y0))
  
    # stopping criteria
    kkt = False
  
    while kkt == False:
      # generator
      gen = Sigmoid(sigma=np.copy(sigma))
      # compute x0
      x0 = gen.inv(y0)
  
      # merit
      if self.solve_method == "scipy":
        ft = lambda xx: self.fwrap(from_unit_cube(gen(xx),lb,ub))
        ft_jac = lambda xx: gen.jac(xx) @ np.diag(ub-lb) @ self.f_grad(from_unit_cube(gen(xx),lb,ub))
        res = minimize(ft,x0,jac=ft_jac,method="BFGS",options={'gtol':self.delta,'maxiter':1e6})
        xopt = res.x
      elif self.solve_method == "nlopt": 
        # nlopt objective
        def objective_with_grad(xx,g):
          ft =self.fwrap(from_unit_cube(gen(xx),lb,ub))
          g[:] = gen.jac(xx) @ np.diag(ub-lb) @ self.f_grad(from_unit_cube(gen(xx),lb,ub))
          return ft
        opt = nlopt.opt(nlopt.LD_LBFGS, self.dim)
        opt.set_min_objective(objective_with_grad)
        opt.set_ftol_rel(self.delta)
        opt.set_ftol_abs(0.0)
        opt.set_xtol_rel(self.delta)
        opt.set_xtol_abs(self.delta)
        opt.set_maxeval(int(1e6))
        try:
          # nlopt may fail
          xopt = opt.optimize(x0)
        except:
          print("EXITING: nlopt failed")
          return from_unit_cube(yopt,lb,ub)
  
      # compute y*
      yopt = np.copy(gen(xopt))
      # compute distance to boundary
      eta = np.minimum(yopt,1-yopt)
      # zopt = y^* in original domain [lb,ub]
      zopt = np.copy(from_unit_cube(yopt,lb,ub))
      g_z  = np.copy(self.f_grad(zopt))
  
      # check the kkt conditions
      kkt_eps = compute_kkt_tol(zopt,g_z,lb,ub,eps=1.0)
      if verbose:
        print(f"f(x): {self.func(zopt)}, kkt satisfaction: {kkt_eps}")
      if kkt_eps <= self.eps: 
        kkt = True
        return zopt
  
      if np.all(sigma >= self.cap_sigma-1):
        if verbose:
          print("Breaking: max sigma reached")
        return zopt
  
      # update sigma
      if self.update_method == "adaptive":
        eta[eta<self.cap_eta] = self.cap_eta # cap the update size
        sigma = self.gamma*sigma/np.sqrt(eta)
        #sigma = self.gamma*sigma/eta
        sigma = np.minimum(sigma,self.cap_sigma)
      elif self.update_method == "exp":
        sigma = np.minimum(self.gamma*sigma,self.cap_sigma)

      # reset for next iteration
      y0 = np.copy(yopt)

      # save the index of the update
      self.updates.append(len(self.fX)-1)
  
    return zopt

