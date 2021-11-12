from scipy.optimize import minimize,Bounds
import numpy as np

class AugmentedLagrangian():
  """
  Augmented Lagrangian method from Nocedal and Wright
  Framework 17.3, adapted for finite bound constrained problems.
  
  min f(x)
      l < x < u
  
  We turn this into an equality constrained problem by introducing
  slack variables s,t. We optimize over z=(x,s,t). We form equality 
  constraints c(z) = [x-s-l=0, x+t-u=0] and dont forget the non-negativity
  constraints on the slack s,t>0.
  
  We form the augmented lagrangian
    L(z,lam,mu) = f(x) - sum_i lam_i*c_i(z) + (mu/2)sum_i c_i^2(z)

  There are a total of 2*dim constraints. The variables have sizes
  x: dim, s: dim, t: dim, lam: 2*dim, mu: 1
  """

  def __init__(self,func,grad,lb,ub,eta_star = 1e-6,w_star=1e-6,method='BFGS'):
    """
    eta_star: gradient tolerance on lagrangian
    w_star: constraint violation tolerance
    """
    assert len(lb) == len(ub), "bad bounds"
    self.func = func
    self.grad = grad
    self.dim_x = len(lb)
    self.lb = lb
    self.ub = ub
    self.eta_star = eta_star
    self.w_star = w_star
    self.method = method

    # helpful sizes
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
    x = z[:dim] 
    s = z[dim:2*dim]
    t = z[2*dim:]
    return np.hstack((x - s - self.lb, x+t- self.ub))

  def con_jac(self,z):
    """
    Evaluate equality constraint jacobian
    z = [x,s,t] is our variable and slack variables.
    return: constraint jacobian, (n_con,3*dim) array
    """
    x = z[:dim] 
    s = z[dim:2*dim]
    t = z[2*dim:]
    Dx = np.vstack([np.eye(self.dim),np.eye(self.dim)])
    Ds = np.vstack([-np.eye(self.dim),np.zeros(self.dim,self.dim)])
    Dt = np.vstack([np.zeros(self.dim,self.dim),np.eye(self.dim)])
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
    c_jac = self.con(z)
    # lagranian gradient
    Lg =  self.grad(x) - c_jac.T @ lam + mu*c_jac.T @ c)
    return Lg

  def project(self,y):
    """
    project y onto [lb,ub]
    """
    idx_up = y>ub
    y[idx_up] = ub[idx_up]
    idx_low = y<lb
    y[idx_low] = lb[idx_low]
    return y
    
  def solve(self,x0):
    """
    solve the problem with the augmented lagrangian method
    """
    assert len(x0) == self.dim_x, "x0 is not same size as lb,ub"
    assert np.all(self.lb <= x0) and np.all(x0 <= self.ub), "x0 not feasible"

    # lagrange multipliers
    lam_k = self.lam
    mu_k = self.mu
    # tolerances
    w_k = 1/mu_k
    eta_k = 1/(mu_k**0.1)
    # initialize s,t
    s_k = x0 - self.lb
    t_k = self.ub - x0
    z_k = np.hstack((x0,s_k,t_k))
  
    for k in range(max_solves):

        # minimize the lagrangian
        res = minimize(self.lagrangian,z_k,jac=self.lagrangian_grad,method=self.method,options={'gtol':w_k})
        z_kp1 = res.x

        # evaluate the constraints
        cc = self.con(z_kp1)
        # evaluate gradient of augmented lagrangian
        Lg = self.lagrangian_grad(z_kp1)
  
        if np.linalg.norm(cc) <=eta_k: # check constraint satisfaction

          # stop if we satisfy convergence and constraint violation tol
          if np.linalg.norm(cc) <=eta_star and Lg <= w_star:
              x_kp1 = z_kp1[:dim]
              return x_kp1
  
          # update multipliers and tighten tolerances
          lam_k = np.copy(lam_k - mu_k*c(z_kp1))
          self.lam = lam_k # make sure to save lambda
          eta_k = eta_k/(mu_k**0.9)
          w_k   = w_k/mu_k

        else:
          # increase penalty param and tighten tolerances
          mu_k = 100*mu_k
          self.mu = mu_k # make sure to save mu
          eta_k = 1.0/(mu_k**0.1)
          w_k  = 1.0/mu_k
