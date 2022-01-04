import numpy as np

def dist_to_boundary(yy):
  return np.min(np.minimum(yy,1-yy))

def SD(func,grad,sig,x0,gamma=0.5,max_iter=10000,gtol=1e-3,ftol_rel=1e-10,c_1=1e-4,verbose=False):
  """
  Steepest descent with armijo linesearch for minimization of the sigmoidal
  connection function 
    min_x ft(x) = f(Sigmoid(x))

  The step sequence is 
    x_k+1 = x_k - alpha*(J^TJ)^{-1}grad(ft)
          = x_k - alpha/(sigma*y_k*(1-y_k)) * grad(ft)
    where y_k = Sigmoid(x_k) and J is the jacobian of the sigmoid.

  Optimization will stop if any of the stopping criteria are met.

  func: objective function handle, f
  grad: gradient handle of grad f
  sig: handle for sigmoid
  x0: feasible starting point
  gamma: linesearch decrease parameter
  max_iter: maximimum number of iterations
  gtol: projected gradient tolerance
  c_1: Armijo parameters for linesearch.
           must satisfy 0 < c_1 < c_2 < 1
  """
  assert 0 < c_1 and c_1< 1, "unsuitable linesearch parameters"

  # inital guess
  x_k = np.copy(x0)
  dim = len(x_k)

  # minimum step size
  alpha_min = 1e-18
  # initial step size
  alpha_k = 1.0

  y_k = sig(x_k)
  # compute gradient
  g_k    = np.copy(grad(y_k))
  # compute function value
  f_k    = np.copy(func(y_k))

  # stop when gradient is flat (within tolerance)
  nn = 0
  stop = False
  while stop==False:
    if verbose:
      print(f_k,np.linalg.norm(g_k),dist_to_boundary(y_k))
    # increase alpha to counter backtracking
    alpha_k = alpha_k/gamma
    #alpha_k = 1.0

    # vectorized jacobian
    J_k    = np.copy(np.diag(sig.jac(x_k)))

    # compute search direction
    p_k = - (1.0/J_k)*g_k

    # compute step 
    x_kp1 = x_k + alpha_k*p_k
    y_kp1 = np.copy(sig(x_kp1))
    f_kp1 = func(y_kp1);

    # linsearch with Armijo condition
    armijo = f_kp1 <= f_k + c_1*J_k*g_k @ (x_kp1 - x_k)
    while armijo==False or dist_to_boundary(y_kp1)<1e-15:
      # reduce our step size
      alpha_k = gamma*alpha_k;
      # take step
      x_kp1 = np.copy(x_k + alpha_k*p_k)
      y_kp1 = np.copy(sig(x_kp1))
      # f_kp1
      f_kp1 = func(y_kp1);
      # compute the armijo condition
      armijo = f_kp1 <= f_k + c_1*J_k*g_k @ (x_kp1 - x_k)

      # break if alpha is too small
      if alpha_k <= alpha_min:
        if verbose:
          print('Exiting: alpha too small.')
        return x_k

    # gradient
    g_kp1 = np.copy(grad(y_kp1))

    # reset for next iteration
    y_k  = np.copy(y_kp1)
    x_k  = np.copy(x_kp1)
    f_k  = f_kp1;
    g_k  = np.copy(g_kp1);

    # update iteration counter
    nn += 1

    # stopping criteria
    if np.linalg.norm(g_k) <= gtol:
      if verbose:
        print("Exiting: gtol reached")
      stop = True
    elif nn >= max_iter:
      if verbose:
        print("Exiting: max_iter reached")
      stop = True

  return x_k

