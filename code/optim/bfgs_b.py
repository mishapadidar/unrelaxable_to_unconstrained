import numpy as np

def project(y,lb,ub):
  """
  project a vector y onto [lb,ub]
  """
  y = np.copy(y) # prevent aliasing
  idx_up = y> ub
  y[idx_up] = ub[idx_up]
  idx_low = y< lb
  y[idx_low] = lb[idx_low]
  return y

def check_stop(x_k,g_k,lb,ub,nn,gtol,xtol,max_iter):
  """
  Check the stopping criteria
  """
  if np.linalg.norm(project(g_k,lb,ub)) < gtol :
    return True
  elif np.linalg.norm(x_k - project(x_k - g_k,lb,ub)) < xtol :
    return True
  elif nn > max_iter:
    return True
  else:
    return False


def BFGS_B(func,grad,x0,lb,ub,gamma=0.5,max_iter=10000,gtol=1e-3,xtol=1e-9,c_1=1e-4,c_2=0.9):
  """
  This is an implementation of the BFGS algorithm (Nocedal and Wright Alg 16.1)
  for solving bound constrained optimization problems with gradients:
      min f(x)
      s.t. lb < x < ub
  There are two modifications to the N & W algorithm include bound constraints
    1. We project steps onto the feasible region with
       x_kp1 = projection(x_k - alpha_k*p_k,lb,ub)
    2. Our stopping criteria is now the bound constrained stopping criteria:
       if x is a minima to the bound constrained problem then
         x = projection(x - grad(x),lb,ub)
       Thus we check the stopping criteria 
         ||x - projection(x-grad(x),lb,ub)|| < gtol

  func: objective function handle, for minimization
  grad: gradient handle
  x0: feasible starting point
  lb, ub: lower and upper bound arrays
  gamma: linesearch decrease parameter
  max_iter: maximimum number of iterations
  gtol: projected gradient tolerance
  c_1,c_2: Wolfe condition parameters for linesearch.
           must satisfy 0 < c_1 < c_2 < 1
  """
  assert 0 < c_1 and c_1 < c_2 and c_2 < 1, "unsuitable linesearch parameters"
  assert np.all(lb<=x0) and np.all(x0<=ub),"need feasible starting point"

  # inital guess
  x_k = np.copy(x0)
  dim = len(x_k)

  # minimum step size
  alpha_min = 1e-18
  # conditioning parameter
  eps = 1e-14

  # compute gradient
  g_k    = np.copy(grad(x_k))
  # compute function value
  f_k    = func(x_k)

  # initialize inverse hessian 
  H_k = np.eye(dim)

  # identity
  I = np.eye(dim)

  # stop when gradient is flat (within tolerance)
  nn = 0
  stop = False
  while stop==False:
    # always try alpha=1 first
    alpha_k = 1.0

    # compute search direction
    p_k = - np.copy(H_k @ g_k)

    # compute step 
    x_kp1 = np.copy(project(x_k + alpha_k*p_k,lb,ub))
    f_kp1 = func(x_kp1);
    g_kp1 = np.copy(grad(x_kp1))

    # linsearch with Wolfe Conditions
    armijo = f_kp1 <= f_k + c_1*g_k @ (x_kp1 - x_k)
    # TODO: derive the curvature condition for bound constrained problems
    #curv_cond = g_kp1 @ (x_kp1-x_k) >= c_2*g_k @ (x_kp1 - x_k)
    while armijo==False: #or curv_cond==False:
      # reduce our step size
      alpha_k = gamma*alpha_k;
      # take step
      x_kp1 = np.copy(project(x_k + alpha_k*p_k,lb,ub))
      # f_kp1
      f_kp1 = func(x_kp1);
      g_kp1 = np.copy(grad(x_kp1))
      # compute the armijo condition
      armijo = f_kp1 <= f_k + c_1*g_k @ (x_kp1 - x_k)
      # compute the curvature condition
      #curv_cond = g_kp1 @ (x_kp1-x_k) >= c_2*g_k @ (x_kp1 - x_k)

      # break if alpha is too small
      if alpha_k <= alpha_min:
        print('Exiting: alpha too small.')
        return x_k

    # compute step difference
    s_k = np.copy(x_kp1 - x_k)
    y_k = np.copy(g_kp1 - g_k)
    # check curvature condition
    if s_k @ y_k <= eps*y_k@y_k:
      # dont update hessian 
      H_kp1 = H_k
    else:
      rho_k = 1.0/(y_k@s_k)
      # update Hessian
      H_kp1 = (I - rho_k*np.outer(s_k,y_k)) @ H_k @ (I - rho_k*np.outer(y_k,s_k)) + rho_k*np.outer(s_k,s_k)

    # reset for next iteration
    x_k  = np.copy(x_kp1)
    f_k  = f_kp1;
    g_k  = np.copy(g_kp1);
    H_k  = np.copy(H_kp1)

    # update iteration counter
    nn += 1

    # check stopping criteria
    stop = check_stop(x_k,g_k,lb,ub,nn,gtol,xtol,max_iter)

  return x_k

