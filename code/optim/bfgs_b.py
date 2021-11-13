import numpy as np

def project(y,lb,ub):
  """
  project a vector y onto [lb,ub]
  """
  idx_up = y> ub
  y[idx_up] = ub[idx_up]
  idx_low = y< lb
  y[idx_low] = lb[idx_low]
  return y

def BFGS_B(func,grad,x0,lb,ub,gamma=0.5,max_iter=10000,gtol=1e-3,c_1=1e-4,c_2=0.9):
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

  # compute gradient
  g_k    = grad(x_k)
  # compute function value
  f_k    = func(x_k)

  # initialize inverse hessian 
  H_k = np.eye(dim)

  # identity
  I = np.eye(dim)

  # stop when gradient is flat (within tolerance)
  nn = 0
  while np.linalg.norm(x_k - project(x_k - g_k,lb,ub)) > gtol and nn < max_iter:
    # always try alpha=1 first
    alpha_k = 1.0

    # compute search direction
    p_k = - H_k @ g_k

    # gradient times p
    gp_k = g_k @ p_k
    
    # compute step 
    x_kp1 = project(x_k + alpha_k*p_k,lb,ub)
    f_kp1 = func(x_kp1);
    g_kp1 = grad(x_kp1)

    # linsearch with Wolfe Conditions
    while f_kp1 > f_k + c_1*alpha_k*gp_k or g_kp1 @ p_k < c_2 *gp_k:
      # reduce our step size
      alpha_k = gamma*alpha_k;
      # take step
      x_kp1 = np.copy(project(x_k + alpha_k*p_k,lb,ub))
      # f_kp1
      f_kp1 = func(x_kp1);
      g_kp1 = np.copy(grad(x_kp1))

      # break if alpha is too small
      if alpha_k <= alpha_min:
        print('Exiting: alpha too small.')
        return x_k

    # compute step difference
    s_k = np.copy(x_kp1 - x_k)
    y_k = np.copy(g_kp1 - g_k)
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

  return x_k

