import numpy as np
import pickle
import pycutest
import os
import sys
from scipy.optimize import minimize,Bounds
import nlopt
import pandas as pd
# our libs
sys.path.append("../../generators/")
sys.path.append("../../optim/")
sys.path.append("../../utils/")
from project import project
from rescale import *
from sigmoid import Sigmoid
from sigup import SIGUP
from gradient_descent import GD
from eval_wrapper import eval_wrapper
sys.path.append("../../../../NQN")
import NQN

# directory to dump data
outputdir = "./data/"
if os.path.exists(outputdir) is False:
  os.mkdir(outputdir)

# load the cutest problems
problems = pd.read_pickle("../../problems/cutest_problems.pickle")

# sigup parameters
sig_sigma0_method = 'fixed'
sig_sigma0 = 1e-3
sig_eps    = 0.0 # set to zero for infinite run
sig_delta  = 0.0 # use finite value so we update sigma
sig_gamma  = 10.0
sig_solve_method = "nlopt"
sig_update_method = "adaptive"

# sigmoid-fixed
sigmoid_fixed_sigmas = [0.001,1.0,10.0]

# BFGS params
gtol = sig_eps
maxfun = int(20000)
maxiter = int(20000)

# PPM params
ppm_max_iter = int(20000)
ppm_gtol = 1e-12

for pname in problems['name']:
  # skip some problems
  #if "DIAG" in pname or pname =="HADAMALS" or pname=="PROBPENL" or pname=="POWELLBC":
  #  continue
  if pname=="PROBPENL":
    continue

  print(f"\nproblem: {pname}")
  problem_data = {}
  problem_data['problem'] = pname
  problem_data['runs'] = [] # list to contain runs

  # load problem
  prob = pycutest.import_problem(pname)
  lb,ub = prob.bl,prob.bu
  dim = prob.n
  y0 = prob.x0
  obj = prob.obj
  grad = lambda xx: prob.obj(xx,gradient=True)[1]

  # check y0 and bounds
  assert np.all(lb < ub), "bound violation lb > ub"
  if np.any(lb >= y0) or np.any(y0>= ub):
    # generate a strict interior point
    factor = 1e-3
    idx_up = y0 >= ub
    y0[idx_up] = ub[idx_up] -factor*(ub[idx_up]-lb[idx_up])
    idx_low = y0 <= lb
    y0[idx_low] = lb[idx_low] +factor*(ub[idx_low]-lb[idx_low])
    if np.any(np.isnan(grad(y0))) == False and np.any(np.isnan(obj(y0))) == False:
      print("y0 infeasible... using shifted y0")
    else:
      print("ERROR: Cant find feasible y0")
      quit()

  # double check initial point
  assert np.all(lb < y0) and np.all(y0 < ub), f"y0 is infeasible {y0}"
  assert np.any(np.isnan(grad(y0))) == False and  np.any(np.isnan(obj(y0))) == False, "y0 has nan objective or grad"

  # collect the dimension
  problem_data['dim'] = dim

  # call sigup
  method = f'sigup-{sig_sigma0}'
  sigup = SIGUP(obj,grad,lb,ub,y0,eps = sig_eps,delta=sig_delta,gamma=sig_gamma,sigma0=sig_sigma0,
          solve_method=sig_solve_method,update_method=sig_update_method,sigma0_method=sig_sigma0_method)
  z = sigup.solve()
  #try:
  #  #z = sigup(func,grad,lb,ub,y0,sigma0=sig_sigma0,eps =sig_eps,delta=sig_delta,gamma=sig_gamma,method=sig_method,verbose=False)
  #  z = sigup.solve()
  #except:
  #  z = sigup.X[np.argmin(sigup.fX)]
  X = sigup.X
  fX = sigup.fX
  fopt = np.min(fX)
  print(f"{method}: {fopt}")
  print(f"num_updates: {len(sigup.updates)}")
  method_data = {}
  method_data['method'] = method
  method_data['solve_method'] = sig_solve_method
  method_data['update_method'] = sig_update_method
  method_data['X'] = X
  method_data['fX'] = fX
  method_data['sigma0'] = sig_sigma0
  method_data['gamma'] = sig_gamma
  method_data['eps'] = sig_eps
  method_data['delta'] = sig_delta
  problem_data['runs'].append(method_data)


  # projected gradient
  method = "L-BFGS-B"
  func = eval_wrapper(obj,dim)
  # nlopt objective
  def objective_with_grad(x,g):
    f = func(x)
    g[:] = grad(x)
    return f
  opt = nlopt.opt(nlopt.LD_LBFGS, dim)
  opt.set_min_objective(objective_with_grad)
  opt.set_lower_bounds(lb)
  opt.set_upper_bounds(ub)
  opt.set_maxeval(maxfun)
  opt.set_ftol_abs(0)
  opt.set_ftol_rel(0)
  opt.set_xtol_abs(0)
  opt.set_xtol_rel(0)
  try:
    #res = minimize(func,y0,jac=grad,method=method,options={'gtol':1e-10,'ftol':1e-20,'maxiter':maxiter,'maxfun':maxfun})
    #z = res.x
    z = opt.optimize(y0)
  except:
    idx = np.argmin(func.fX)
    z = func.X[idx]
    pass
  X = func.X
  fX = func.fX
  fopt = np.min(fX)
  print(f"{method}: {fopt}")
  print("L-BFGS-B NLOPT return code: ",opt.last_optimize_result())
  method_data = {}
  method_data['method'] = method
  method_data['X'] = X
  method_data['fX'] = fX
  problem_data['runs'].append(method_data)


  # call sigma method with no update
  for sigma0 in sigmoid_fixed_sigmas:
    method = f'sigmoid-fixed-{sigma0}'
    if sigma0 == 'adaptive':
      yy = np.copy(to_unit_cube(y0,lb,ub))
      sigma0 = 1.0/(yy*(1-yy))
    func = eval_wrapper(obj,dim) # wrap the objective
    sig = Sigmoid(sigma=sigma0)
    def objective_with_grad(xx,g):
      f = func(from_unit_cube(sig(xx),lb,ub))
      g[:] = sig.jac(xx) @ np.diag(ub-lb) @ grad(from_unit_cube(sig(xx),lb,ub))
      return f
    opt = nlopt.opt(nlopt.LD_LBFGS, dim)
    opt.set_min_objective(objective_with_grad)
    opt.set_maxeval(maxfun)
    opt.set_ftol_abs(0)
    opt.set_ftol_rel(0)
    opt.set_xtol_abs(0)
    opt.set_xtol_rel(0)
    try:
      z = opt.optimize(sig.inv(to_unit_cube(y0,lb,ub)))
    except:
      z = func.X[-1]
      pass
    X = func.X
    fX = func.fX
    fopt = np.min(fX)
    print(f"{method}: {fopt}")
    method_data = {}
    method_data['method'] = method
    method_data['sigma0'] = sigma0
    method_data['X'] = X
    method_data['fX'] = fX
    problem_data['runs'].append(method_data)

  # call projection penalty method
  method = "PPM"
  func = eval_wrapper(obj,dim)
  def dist_pen(xx):
    yy = np.copy(xx)
    return np.linalg.norm(yy - np.copy(project(yy,lb,ub)))
  
  def proj_pen(xx):
    yy = np.copy(xx)
    return func(project(yy,lb,ub)) + dist_pen(yy)
  
  def proj_pen_grad(xx):
    """
    Gradient of the projected penalty
      f(project(x)) + ||x - project(x)||
  
    xx: 1d array, point
    return: subgradient g such that -g is a 
           descent direction.
    """
    yy = np.copy(xx)
    bndry_tol=1e-14
    if np.all(yy<ub-bndry_tol) and np.all(yy>lb+bndry_tol):
      # interior point: grad = grad
      return grad(yy)
    elif np.all(yy<=ub) and np.all(yy>=lb):
      # boundary point: negative grad = negative projected gradient
      return -(project(yy - grad(yy),lb,ub) - yy)
    else:
      # exterior point
      px = project(yy,lb,ub)
  
      Dpi = 0.0*np.zeros_like(yy)
      #idx_int = np.logical_and(yy<ub-bndry_tol,yy>lb+bndry_tol)
      #Dpi[idx_int] = 1.0
      # TODO: consider changing this to a projected gradient plus distance
      idx_feas = np.logical_and(yy<=ub,yy>=lb)
      Dpi[idx_feas] = 1.0
      gg = np.copy(Dpi*grad(project(yy,lb,ub)))
      gg += (yy-px)/np.linalg.norm(yy-px)
      return np.copy(gg)


  #xopt = GD(proj_pen,proj_pen_grad,np.copy(y0),max_iter=ppm_max_iter,gtol=ppm_gtol,verbose=False)
  #res = minimize(proj_pen,y0,jac=proj_pen_grad,method="BFGS",options={'gtol':ppm_gtol,'maxiter':ppm_max_iter})
  #xopt = res.x
  res = NQN.fmin_l_bfgs_b(proj_pen, y0, proj_pen_grad, bounds=None, m=20, M=1, pgtol=ppm_gtol, iprint=-1, maxfun=ppm_max_iter, maxiter=ppm_max_iter, callback=None, factr=0.)
  xopt = np.copy(res[0])
  X = func.X
  fX = func.fX
  fopt = np.min(fX)
  print(f"{method}: {fopt}")
  method_data = {}
  method_data['method'] = method
  method_data['X'] = X
  method_data['fX'] = fX
  problem_data['runs'].append(method_data)

  # save data
  problem_filename = outputdir + f"{pname}.pickle"
  pickle.dump(problem_data,open(problem_filename,"wb"))
  
