import numpy as np

def check_kkt(y,g,lb,ub,eps):
  """
  Check approximate stationarity 
  for bound constrained problems with finite bounds
    min_y f(y) s.t. lb < y < ub

  Lagrangian is 
    L(y,lam,mu) = f(y) + lam*(y-lb) + mu*(ub-y)

  epsilon-approximate kkt conditions are 
    |gradf + lam -mu| <= eps
    |lam*(y-lb)| <= eps
    |mu*(ub-y)| <= eps
    lam,mu <=0
    y <= ub + eps
    y >= lb - eps

  We check this system of linear constraints analytically. 
  There are two keys ideas in solving this system of linear
  inequalities:
    - Each dimension can be solved separate of all other dimensions.
    - We can simplify our system based off the sign of the derivative.
      If grad_i <0 then lam_i = 0. If grad_i>0 then mu_i = 0. This holds
      because of lam,mu<=0 and the stationary condition.

  y: point y
  g: gradient vector grad_f(y)
  lb,ub: lower and upper bound vectors
  eps: kkt tolerance

  if True: 
    return True, [lam,mu]
  if False:
    return False, []
  """
  # check for nans
  assert not np.any(np.isnan(g)), "gradient has NaN entries"

  # check primal feasibility
  if np.all(np.logical_or(y>=lb-eps,y<=ub+eps)): 
    pass
  else: 
    return False,[]

  # dimension
  dim = len(y)

  # initialize lambda, mu
  lam = np.zeros(dim)
  mu = np.zeros(dim)

  # determine active lam or mu from sign of gradient
  idx_mu  = g< 0.0
  idx_lam = g> 0.0

  for ii in range(dim):
    if idx_mu[ii] == True:
      # compute the inequalities
      # check for true activity
      if ub[ii] == y[ii]: 
        ub_mu = min(0.0,eps+g[ii])
        lb_mu = g[ii]-eps
      else:
        ub_mu = min(0.0,eps+g[ii])
        # TODO: this is the only potentially unstable calculation
        lb_mu = max(g[ii]-eps,-np.abs(eps/(ub[ii]-y[ii])))
      # check kkt 
      if lb_mu > ub_mu:
        return False,[]
      else:
        # assign a value to mu
        mu[ii] = (ub_mu + lb_mu)/2.0

    elif idx_lam[ii] == True:
      # compute the inequalities
      # check for true activity
      if y[ii] - lb[ii]  ==0.0:
        ub_lam = min(0.0,eps-g[ii])
        lb_lam = -eps-g[ii]
      else:
        ub_lam = min(0.0,eps-g[ii])
        # TODO: this is the only potentially unstable calculation
        lb_lam = max(-eps-g[ii],-np.abs(eps/(y[ii]-lb[ii])))
      # check kkt 
      if lb_lam > ub_lam:
        return False,[]
      else:
        # assign a value to lam
        lam[ii] = (ub_lam + lb_lam)/2.0

  # double check results
  assert np.all(lam<=0.0), "ERROR: Bad multiplier: lambda > 0"
  assert np.all(mu<=0.0), "ERROR: Bad multiplier: mu > 0"

  return True,[lam,mu]


def compute_kkt_tol(y,g,lb,ub,eps=1.0):
  """
  Compute the minimum value epsilon such that
  the epsilon-approximate kkt conditions hold.
     min  eps
     s.t. |gradf + lam -mu| <= eps
          |lam*(y-lb)| <= eps
          |mu*(ub-y)| <= eps
          lam,mu <=0
          y <= ub + eps
          y >= lb - eps

  To avoid any problems with poorly conditioned numbers
  we opt to solely rely on the check_kkt function above
  rather than solving the LP explicitly. We use a binary 
  search on eps. Our alg is as follows:

  1. If the initial guess for the minima does not satisfy
     the epsilon-kkt conditions then increase it until
     we find an epsilon that does. This will give 
     us an interval [lb_eps,ub_eps] that contains the optima.
     The lb_eps will not satisfy the epsilon kkt conditions but
     ub_eps will satisfy the epsilon kkt conditions.
  2. Now perform a binary search on [lb_eps, ub_eps] to find the 
     optima. We stop when ub_eps is approximate 2*lb_eps, i.e.
     the relative error is (ub_eps-lb_eps)/lb_eps < 1.
  3. return ub_eps
  
  y: point y
  g: gradient vector grad_f(y)
  lb,ub: lower and upper bound vectors
  eps: initial guess for kkt tolerance
  return optimal epsilon
  """
  # check for nans
  if np.any(np.isnan(g)) or np.any(np.isnan(y)):
    return np.inf 

  # increase/decrease factor for search
  gamma = 2.0
  assert gamma > 1.0,"gamma must be > 1.0"

  # first find an interval containing epsion_opt
  ub_eps = eps
  lb_eps = 0.0
  while check_kkt(y,g,lb,ub,ub_eps)[0] == False:
    # move the interval
    lb_eps = ub_eps
    ub_eps = gamma*ub_eps
  
  # now binary search until ub_eps ~ 2lb_eps
  while (ub_eps-lb_eps) > lb_eps:
    eps = lb_eps + (ub_eps-lb_eps)/gamma
    kkt = check_kkt(y,g,lb,ub,eps)[0]
    if kkt == True:
      ub_eps = eps
    else:
      lb_eps = eps

  return ub_eps


if __name__=="__main__":
  import sys
  sys.path.append("../problems/")
  from rosenbrock import Rosenbrock
  from convex_quadratic import ConvexQuadratic

  dim  = 2
  lb = np.zeros(dim)
  ub = np.ones(dim)
  
  #yopt = np.ones(dim) 
  #yopt = np.zeros(dim) 
  #yopt = np.ones(dim) + np.array([-1e-7,1e-1])
  #yopt = np.ones(dim) - np.array([1e-7,1e-1])
  #yopt = np.ones(dim) - np.array([0.5,0.0])
  yopt = np.ones(dim) + np.array([0.5,0.0])
  A = np.diag(np.array([100,2]))
  f = ConvexQuadratic(lb,ub,A,yopt)
  
  # point
  y= yopt + 1e-6
  #y = np.ones(dim)

  # compute the multipliers
  g = f.grad(y)
  lb = f.lb
  ub = f.ub
  eps = 1e-2
  kkt,lagrange = check_kkt(y,g,lb,ub,eps)
  if kkt == True:
    lam,mu = lagrange
    print('')
    print('grad')
    print(g)
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
  else:
    print("Not KKT to tol",eps)

  eps_opt = compute_kkt_tol(y,g,lb,ub,eps=1e-8)
  print("")
  print("Found optimal eps: ",eps_opt)
  kkt,lagrange = check_kkt(y,g,lb,ub,eps_opt)
  lam,mu = lagrange
  print('lam')
  print(lam)
  print('mu')
  print(mu)
  print('stationary')
  print(np.abs(g + lam - mu))
  print('complementary slackenss lambda*y')
  print(np.abs(lam*(y-lb)))
  print('complementary slackenss mu*(1-y)')
  print(np.abs(mu*(ub-y)))
