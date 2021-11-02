import numpy as np
from scipy.optimize import minimize,Bounds

def compute_lagrange(y,g,lb,ub):
  """
  Check approximate stationarity 
  for bound constrained problems with finite bounds
  min_y f(y) s.t. lb < y < ub

  Lagrangian is 
  L(y,lam,mu) = f(y) + lam*(y-lb) + mu*(ub-y)

  epsilon-approximate kkt conditions are 
  |gradf + lam -mu| < eps
  |lam*(y-lb)| < eps
  |mu*(ub-y)| < eps
  lam,mu <0

  We solve the QP
  min_{lam,mu} ||gradf + lam -mu||^2  + ||lam*(y-lb)||^2 + ||mu*(ub-y)||^2
  s.t. lam,mu < 0

  We assume that y is feasible.
  y: feasible point y
  g: gradient vector f(y)
  lb,ub: lower and upper bound vectors
  """

  # sizes
  dim_y = len(y)
  n_con = 2*dim_y
  def f(z):
    lam = z[:dim_y]
    mu  = z[dim_y:]
    one  = (g+lam-mu) @ (g+lam-mu)
    two  = lam @ (y-lb) * (y-lb) @ lam
    three = mu @ (ub-y) * (ub-y) @ mu
    return 0.5*(one + two + three)

  def jac(z):
    lam = z[:dim_y]
    mu = z[dim_y:]
    dlam = (g+lam-mu) + (y-lb)*(y-lb)@lam
    dmu = -(g+lam-mu) + (ub-y)*(ub-y)@mu
    return np.hstack((dlam,dmu))

  # solve
  bounds = Bounds(-np.inf*np.ones(n_con),np.zeros(n_con))
  res = minimize(f,np.ones(n_con),jac=jac,method="L-BFGS-B",bounds=bounds,options={'gtol':1e-10})
  z = res.x
  lam = z[:dim_y]
  mu = z[dim_y:]

  return lam,mu

if __name__=="__main__":
  import sys
  sys.path.append("../problems/")
  from rosenbrock import Rosenbrock
  from convex_quadratic import ConvexQuadratic

  dim  = 2
  lb = np.zeros(dim)
  ub = np.ones(dim)
  
  #yopt = np.ones(dim) 
  yopt = np.zeros(dim) 
  #yopt = np.ones(dim) + np.array([-1e-7,1e-1])
  #yopt = np.ones(dim) - np.array([1e-7,1e-1])
  #yopt = np.ones(dim) - np.array([0.5,0.0])
  #yopt = np.ones(dim) + np.array([0.5,0.0])
  A = np.diag(np.array([100,2]))
  f = ConvexQuadratic(lb,ub,A,yopt)
  
  # point
  y= yopt + 1e-6

  # compute the multipliers
  g = f.grad(y)
  lb = f.lb
  ub = f.ub
  lam,mu = compute_lagrange(y,g,lb,ub)

  print('')
  print('lam')
  print(lam)
  print('mu')
  print(mu)
  print('')
  print('stationary')
  print(np.abs(g + lam - mu))
  print('complementary slackenss lambda*y')
  print(np.abs(lam*(y-lb)))
  print('complementary slackenss mu*(1-y)')
  print(np.abs(mu*(ub-y)))
