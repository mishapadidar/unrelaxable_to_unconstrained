import numpy as np


def SBFGS(func,grad,sigmoid,x0,gamma=0.5,max_iter=10000,gtol=1e-3,c_1=1e-4,c_2=0.9,verbose=False):
  """
  A composite BFGS method for the minimization of the sigmoid based
  merit function ft(x) = f(sigmoid(x)) using a BFGS hessian approximation.
  We solve the unconstrained optimization
    min ft(x) := f(sigmoid(x))
  with an approximate newton iteration
    x_{k+1} = x_k - alpha_k*Ht^{-1}(x_k) @ gt(x_k)
  where Ht(x_k) is the hessian of ft, and gt(x_k) is the gradient of ft.

  We build a strucutred approximation to Ht by leverging the composite form
  of ft. 
  1. Build a BFGS hessian approximation, B, for f:
     - s_k = y_kp1 - y_k, y_k = grad_f(S(x_kp1)) - grad_f(S(x_k))
     - B_kp1 = B_k - np.outer(Bs,Bs)/(s_k @ Bs) + np.outer(y_k,y_k)/(y_k @ s_k)
       where Bs = B_k @ s_k
  2. Use the analytic form of the hessian and B to compute Ht
     Ht = H_sigma @ diag(grad_f(S(x))) + J @ B @ J

  We should be using the Wolfe conditions to ensure that B_kp1 is positive definite.
  However, at the moment we do not know how to ensure the curvature condition (part of the
  Wolfe conditions) holds for f (not ft). So we just perform and Armijo
  linesearch to determine the step length. We only update the BFGS approximation
  if curvature condition s_k @ y_k > 0 is satisfied, otherwise B_kp1 = B_k.

  func: objective function handle for ft, for minimization
  grad: gradient handle for gradient of ft
  sigmoid: handle to sigmoidal connection, S
  x0: starting point in R^n
  gamma: linesearch decrease parameter
  max_iter: maximimum number of iterations
  gtol: gradient tolerance
  c_1: Armijo condition parameters for linesearch.
           must satisfy 0 < c_1 < 1
  c_2: Curvature condition parameter for linesearch.
           must satisfy c_1 < c_2 < 1

  return: approximate minima x, in unconstrained domain
  rtype: 1d-array
  """
  assert 0 < gamma and gamma < 1, "unsuitable linesearch parameters"
  assert 0 < c_1 and c_1 < 1, "unsuitable linesearch parameters"
  assert c_1 < c_2 and c_2 < 1, "unsuitable linesearch parameters"

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

  # initialize hessian
  B_k = np.eye(dim)

  # identity
  I = np.eye(dim)

  # always try alpha=1 first
  alpha_k = 1.0

  # stop when gradient is flat (within tolerance)
  nn = 0
  stop = False
  while stop==False:

    if verbose:
      print(f_k,np.linalg.norm(g_k))
    # increase alpha to counter backtracking
    alpha_k = alpha_k/gamma

    # jacobian (matrix)
    J_k = sigmoid.jac(x_k)
    # gradient of f (not merit)
    gf_k = g_k/np.diag(J_k)
    # compute the hessian
    H_k = np.diag(sigmoid.hess_vec(x_k) * gf_k) + J_k @ B_k @ J_k

    # compute search direction
    p_k = - np.linalg.solve(H_k,g_k)

    # compute step 
    x_kp1 = np.copy(x_k + alpha_k*p_k)
    f_kp1 = func(x_kp1);
    g_kp1 = grad(x_kp1)
    # gradient of f (not merit)
    gf_kp1 = g_kp1/np.diag(sigmoid.jac(x_kp1))
    # step in original domain
    s_k = np.copy(sigmoid(x_kp1) - sigmoid(x_k))

    # linsearch with Wolfe Conditions
    armijo = f_kp1 <= f_k + c_1*g_k @ (x_kp1 - x_k)
    #curv = gf_kp1 @ s_k >= c_2*gf_k @ s_k
    while armijo==False: 
      # reduce our step size
      alpha_k = gamma*alpha_k;
      # take step
      x_kp1 = np.copy(x_k + alpha_k*p_k)
      # f_kp1
      f_kp1 = func(x_kp1);
      g_kp1 = np.copy(grad(x_kp1))
      # gradient of f (not merit)
      gf_kp1 = np.copy(g_kp1/np.diag(sigmoid.jac(x_kp1)))
      # step in original domain
      s_k = np.copy(sigmoid(x_kp1) - sigmoid(x_k))
      # recompute wolfe conditions
      armijo = f_kp1 <= f_k + c_1*g_k @ (x_kp1 - x_k)
      #curv = gf_kp1 @ s_k >= c_2*gf_k @ s_k

      # break if alpha is too small
      if alpha_k <= alpha_min:
        print('Exiting: alpha too small.')
        return x_k

    # compute gradient difference
    y_k = np.copy(gf_kp1 - gf_k)

    # check curvature condition
    if s_k @ y_k <= eps*y_k@y_k:
      # dont update hessian 
      B_kp1 = B_k
    else:
      Bs = B_k@s_k
      # update Hessian
      B_kp1 = B_k - np.outer(Bs,Bs)/(s_k @ Bs) + np.outer(y_k,y_k)/(y_k @ s_k)

    # reset for next iteration
    x_k  = np.copy(x_kp1)
    f_k  = f_kp1;
    g_k  = np.copy(g_kp1);
    B_k  = np.copy(B_kp1)

    # update iteration counter
    nn += 1

    # stopping criteria
    if np.linalg.norm(g_k) <= gtol:
      if verbose:
        print('Exiting: gtol reached')
      stop = True
    elif nn >= max_iter:
      if verbose:
        print('Exiting: max_iter reached')
      stop = True

  return x_k

