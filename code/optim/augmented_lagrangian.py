from scipy.optimize import minimize,Bounds
from bfgs_b import BFGS_B
from gradient_descent import GD
import numpy as np

class AugmentedLagrangian():
  """
  Augmented Lagrangian method from Nocedal and Wright
  Frameworks 17.3 and 17.4, adapted for finite bound constrained problems.
  
  min f(x)
      l < x < u
  
  We turn this into an equality constrained problem by introducing
  slack variables s,t. We optimize over z=(x,s,t). We form equality 
  constraints c(z) = [x-s-l=0, x+t-u=0] and dont forget the non-negativity
  constraints on the slack s,t>0.
  
  We form the augmented lagrangian
    L(z,lam,mu) = f(x) - sum_i lam_i*c_i(z) + (mu/2)sum_i c_i^2(z)
  where lam are the lagrange multipliers and mu is the penalty parameter.

  There are a total of 2*dim constraints. The variables have sizes
  x: dim, s: dim, t: dim, lam: 2*dim, mu: 1
  """

  def __init__(self,func,grad,lb,ub,eta_star = 1e-6,w_star=1e-6,max_iter=1000):
    """
    func: objective function
    grad: gradient of the objective
    lb,ub: finite lower and upper bound arrays
    eta_star: gradient tolerance on lagrangian
    w_star: constraint violation tolerance
    max_iter: maximum number of iterations
    """
    assert len(lb) == len(ub), "bad bounds"
    self.func = func
    self.grad = grad
    self.lb = lb
    self.ub = ub
    self.eta_star = eta_star
    self.w_star = w_star
    self.method = "BFGS-B"
    self.max_iter = max_iter

    # helpful sizes
    self.dim_x = len(lb)
    self.dim_s = self.dim_x # slack on lower bound
    self.dim_t = self.dim_x # slack on upper bound
    self.dim_z = self.dim_x + self.dim_s + self.dim_t # z = [x,s,t]
    self.n_con   = 2*self.dim_x # number of constraints
    self.dim_lam = 2*self.dim_x # lagrange multipliers
    self.dim_mu  = 1 # penalty parameter

    # lagrange multipliers
    self.lam = np.zeros(self.dim_lam)
    self.mu  = 10.0

  def con(self,z):
    """
    Evaluate the 2*dim equality constraints
    z = [x,s,t] is our variable and slack variables.
    return: constraint values, length n_con vector

    The two equality constraints are
    c_1(x,s,t) = x-s-lb = 0
    c_2(x,s,t) = x+t-ub = 0
    """
    x = z[:self.dim_x] 
    s = z[self.dim_x:2*self.dim_x]
    t = z[2*self.dim_x:]
    return np.hstack((x - s - self.lb, x+t- self.ub))

  def con_jac(self,z):
    """
    Evaluate equality constraint jacobian
    z = [x,s,t] is our variable and slack variables.
    return: constraint jacobian, (n_con,3*dim) array
    """
    x = z[:self.dim_x] 
    s = z[self.dim_x:2*self.dim_x]
    t = z[2*self.dim_x:]
    Dx = np.vstack([np.eye(self.dim_x),np.eye(self.dim_x)])
    Ds = np.vstack([-np.eye(self.dim_s),np.zeros((self.dim_s,self.dim_s))])
    Dt = np.vstack([np.zeros((self.dim_t,self.dim_t)),np.eye(self.dim_t)])
    jac = np.hstack([Dx,Ds,Dt])
    return jac

  # augmented lagrangian 
  def lagrangian(self,z):
    """
    Evaluate the Augmented Lagrangian
      L(z,lam,mu) = f(x) - sum_i lam_i*c_i(z) + (mu/2)sum_i c_i^2(z)
    A: vector of inputs, A = [z,lam,mu] = [x,s,t,lam,mu]
    A: length 3*dim + n_con + 1
    return: scalar, augmented lagrangian value
    """
    assert len(z) == self.dim_z, "input is wrong size"
    x   = z[:self.dim_x]
    lam = self.lam
    mu  = self.mu
    # evaluate the equality constraints
    c = self.con(z)
    # augmented lagrangian
    L = self.func(x) - lam @ c + (mu/2)*np.sum(c**2)
    return L

  def lagrangian_grad(self,z):
    """
    Evaluate the Augmented Lagrangian gradient with respect to z
    where z = [x,s,t].
      gradL(z,lam,mu) = gradf(x) - sum_i lam_i*grad(c_i(z)) + mu*sum_i c_i(z)*grad(c_i(z))
    A: vector of inputs, A = [z,lam,mu] = [x,s,t,lam,mu]
    A: length 3*dim + n_con + 1
    return: (dim_z,) array, augmented lagrangian value
    """
    assert len(z) == self.dim_z, "input is wrong size"
    x   = z[:self.dim_x]
    lam = self.lam
    mu  = self.mu
    # evaluate the equality constraints
    c = self.con(z)
    # constraint jacobian
    c_jac = self.con_jac(z)
    # lagranian gradient
    f_grad = np.hstack([self.grad(x),np.zeros(self.dim_s+self.dim_t)])
    Lg =  f_grad - c_jac.T @ lam + mu*c_jac.T @ c

    return Lg

  def project(self,y,lb,ub):
    """
    project a vector y onto [lb,ub]
    """
    y = np.copy(y)
    idx_up = y> ub
    y[idx_up] = ub[idx_up]
    idx_low = y< lb
    y[idx_low] = lb[idx_low]
    return y

  def solve(self,x0):
    """
    solve the problem with the augmented lagrangian method
    x0: feasible starting point, lb <= x0 <= ub
    return array, approximate minima
    """
    assert len(x0) == self.dim_x, "x0 is not same size as lb,ub"
    assert np.all(self.lb <= x0) and np.all(x0 <= self.ub), "x0 not feasible"

    """
    WARNING: This method is not functional!
    - the inner subproblem is not being solved to the desired accuracy. 
      To fix, set up a solver that can solve the subproblem to 
      accuracy so that the norm of the projected gradient is < w_k
    """

    # lagrange multipliers
    lam_k = self.lam
    mu_k = self.mu
    # tolerances
    w_k = 1.0/mu_k
    eta_k = 1.0/(mu_k**0.1)
    # initialize s,t
    s_k = x0 - self.lb
    t_k = self.ub - x0
    z_k = np.hstack((x0,s_k,t_k))
    # constraints on slack for subproblem
    z_lb = np.zeros(self.dim_z) # s,t >0
    z_lb[:self.dim_x] = -np.inf # unconstrained x
    z_ub = np.inf*np.ones(self.dim_z) # no upper bound
 
    for k in range(self.max_iter):

        # minimize the lagrangian subject to non-negative slack
        #z_kp1 = BFGS_B(self.lagrangian,self.lagrangian_grad,z_k,z_lb,z_ub,gamma=0.5,gtol=0.0,xtol=w_k,max_iter=np.inf)
        z_kp1 = GD(self.lagrangian,self.lagrangian_grad,z_k,z_lb,z_ub,gamma=0.5,gtol=0.0,max_iter=np.inf)

        # evaluate the constraints
        cc = self.con(z_kp1)
        # evaluate gradient of augmented lagrangian
        Lg = self.lagrangian_grad(z_kp1)
  
        if np.linalg.norm(cc) <=eta_k: # check constraint satisfaction

          # stop if we satisfy convergence and constraint violation tol
          proj_cond = z_kp1 - self.project(z_kp1-Lg,z_lb,z_ub)
          if np.linalg.norm(cc) <=self.eta_star and np.linalg.norm(proj_cond) <= self.w_star:
              x_kp1 = z_kp1[:self.dim_x]
              return x_kp1
  
          # update multipliers and tighten tolerances
          lam_k = np.copy(lam_k - mu_k*cc)
          self.lam = lam_k # make sure to save lambda
          eta_k = eta_k/(mu_k**0.9)
          w_k   = w_k/mu_k

        else:
          # increase penalty param and tighten tolerances
          mu_k = 100*mu_k
          self.mu = mu_k # make sure to save mu
          eta_k = 1.0/(mu_k**0.1)
          w_k  = 1.0/mu_k

        # setup for next iteration
        z_k = np.copy(z_kp1)

    return z_k[:self.dim_x]
