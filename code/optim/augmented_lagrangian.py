from scipy.optimize import minimize
from gradient_descent import GD
import numpy as np

class AugmentedLagrangian():
  """
  Augmented Lagrangian method from Nocedal and Wright
  Frameworks 17.3 and 17.4, adapted for finite bound constrained problems.
  
  min f(x)
      l < x < u

  We rewrite the constraints as c(x) <= 0
    l-x <= 0
    x-u <= 0
  
  We form the augmented lagrangian
    L(z,lam,mu) = f(x) - sum_i lam_i*c_i(x) + (mu/2)sum_i max(c_i(x),0)^2
  where lam are the lagrange multipliers and mu is the penalty parameter.

  There are a total of 2*dim constraints. The variables have sizes
  x: dim, lam: 2*dim, mu: 1
  """

  def __init__(self,func,grad,lb,ub,mu_factor=10,gtol=1e-3,ctol=1e-3,max_iter=1000):
    """
    func: objective function
    grad: gradient of the objective
    lb,ub: finite lower and upper bound arrays
    mu_factor: growth rate of mu
    gtol: desired gradient tolerance on Lagrangian
    ctol: desired constraint satisfaction tolerance
    max_iter: maximum number of iterations
    """
    assert len(lb) == len(ub), "bad bounds"
    assert np.all(lb < ub), "bad bounds"
    self.func = func
    self.grad = grad
    self.lb = lb
    self.ub = ub
    self.gtol = gtol
    self.ctol = ctol
    self.max_iter = max_iter

    # solver tolerances
    self.method = "BFGS-B"
    self.solver_max_iter = int(1e5)

    # helpful sizes
    self.dim_x = len(lb)
    self.n_con   = 2*self.dim_x # number of constraints
    self.dim_lam = 2*self.dim_x # lagrange multipliers
    self.dim_mu  = 1 # penalty parameter

    # lagrange multipliers
    self.lam = np.zeros(self.dim_lam)
    self.mu_factor = mu_factor
    self.mu  = 10.0

  def con(self,x):
    """
    Evaluate the 2*dim inequality constraints

    The two inequality constraints are
    c_1(x,s,t) = lb - x <= 0 
    c_2(x,s,t) = x - ub <= 0 
    """
    return np.hstack((self.lb-x,x-self.ub))

  def con_jac(self,x):
    """
    Evaluate the constraint jacobian
    return: constraint jacobian, (n_con,dim)
    """
    Dx = np.vstack([-np.eye(self.dim_x),np.eye(self.dim_x)])
    return Dx

  # augmented lagrangian 
  def lagrangian(self,x):
    """
    Evaluate the Augmented Lagrangian
      L(x,lam,mu) = f(x) - sum_i lam_i*c_i(x) + (mu/2)sum_i max(c_i(x),0)^2(x)
    return: scalar, augmented lagrangian value
    """
    lam = self.lam
    mu  = self.mu
    # evaluate the equality constraints
    c = self.con(x)
    # augmented lagrangian
    L = self.func(x) - lam @ c + (mu/2)*np.sum(np.maximum(c,0)**2)
    return L

  def lagrangian_grad(self,x):
    """
    Evaluate the Augmented Lagrangian gradient with respect to x
      gradL(x,lam,mu) = gradf(x) - sum_i lam_i*grad(c_i(x)) + mu*sum_i max(c_i(x),0)*gradc_i(x)
    where J_c is the jacobian of the constraints
    return: (dim_x,) array, augmented lagrangian value
    """
    lam = self.lam
    mu  = self.mu
    # evaluate the constraints
    c = self.con(x)
    # constraint jacobian
    c_jac = self.con_jac(x)
    # lagranian gradient
    Lg =  self.grad(x) - c_jac.T @ lam + mu*c_jac.T @ np.maximum(c,0.0)
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

  def projected_grad(self,x,g,lb,ub):
    """
    projected gradient 
    proj(x-grad(x)) - x
    """
    y = np.copy(x-g)
    idx_up = y> ub
    y[idx_up] = ub[idx_up]
    idx_low = y< lb
    y[idx_low] = lb[idx_low]
    return y - x

  def solve(self,x0):
    """
    solve the problem with the augmented lagrangian method
    x0: feasible starting point, lb <= x0 <= ub
    return array, approximate minima
    """
    assert len(x0) == self.dim_x, "x0 is not same size as lb,ub"
    assert np.all(self.lb <= x0) and np.all(x0 <= self.ub), "x0 not feasible"

    #x0 = np.random.randn(self.dim_x)
    #fjac = fdiff_jac(self.lagrangian,x0)
    #print(fjac)
    #print(self.lagrangian_grad(x0))
    #quit()

    # lagrange multipliers
    lam_k = self.lam
    mu_k = self.mu

    x_k = np.copy(x0)
 
    for k in range(self.max_iter):
        # minimize the lagrangian
        #x_kp1 = GD(self.lagrangian,self.lagrangian_grad,x_k,gtol=self.gtol,max_iter=self.solver_max_iter)
        res = minimize(self.lagrangian,x_k,jac=self.lagrangian_grad,method='BFGS',options={'gtol':self.gtol})
        x_kp1 = res.x

        # evaluate the constraints
        cc = np.maximum(self.con(x_kp1),0.0)
        # evaluate gradient of augmented lagrangian
        Lg = self.lagrangian_grad(x_kp1)
        # projected gradient
        pg = self.projected_grad(x_kp1,Lg,self.lb,self.ub)
  
        print(f'{k+1}) f: {self.func(x_kp1)}, c: {np.linalg.norm(cc)}, pg: {np.linalg.norm(pg)}')
        # stop if we satisfy convergence and constraint violation tol
        if np.linalg.norm(cc) <=self.ctol: # check constraint satisfaction
          if np.linalg.norm(pg) <= self.gtol: 
            return x_kp1
          else:
            # update multipliers and tighten tolerances
            lam_k = np.copy(lam_k - mu_k*cc)
            self.lam = lam_k # make sure to save lambda
        else:
          # increase penalty param and tighten tolerances
          lam_k = np.copy(lam_k - mu_k*cc)
          self.lam = lam_k # make sure to save lambda
          mu_k = np.copy(self.mu_factor*mu_k)
          self.mu = mu_k

        # setup for next iteration
        x_k = np.copy(x_kp1)

    return x_kp1
