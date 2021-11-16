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
from rescale import *
from sigmoid import Sigmoid
from sigup import SIGUP
from eval_wrapper import eval_wrapper
#import augmented_lagrangian

# directory to dump data
outputdir = "./data/"
if os.path.exists(outputdir) is False:
  os.mkdir(outputdir)

# load the cutest problems
problems = pd.read_pickle("../../problems/cutest_problems.pickle")

# sigup parameters
sig_sigma0 = 0.01
sig_eps    = 0.0 # set to zero for infinite run
sig_delta  = 0.0 # use finite value so we update sigma
sig_gamma  = 1.0
sig_solve_method = "nlopt"
sig_update_method = "adaptive"

# sigmoid-fixed
sigmoid_fixed_sigmas = [0.01,0.1,1.0,10.0]

# LBFGS params
gtol = sig_eps
ftol = 0.0 # for infinite run
xtol = 0.0
maxfun = int(1e6)
maxiter = int(1e6)

for pname in problems['name']:
  # skip some problems
  if "DIAG" in pname or pname =="HADAMALS" or pname=="PROBPENL":
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
    # try to generate a strict interior point

    # try a deterministic guess
    y0 = (lb+ub)/2
    if np.any(np.isnan(grad(y0))) == False and np.any(np.isnan(obj(y0))) == False:
      print("y0 is the midpoint of the box")
    else:
      # look for a random feasible initial point
      for attempts in range(1000):
        y0 = np.random.uniform(lb,ub)
        if np.any(np.isnan(grad(y0))) == False and np.any(np.isnan(obj(y0))) == False:
          print("y0 is a random point in the box")
          break
      if np.any(np.isnan(grad(y0))) == True or  np.any(np.isnan(obj(y0))) == True:
        print("Warning: Skipping problem",pname)
        print("Cant find initial feasible point")
        continue

  # double check initial point
  assert np.all(lb < y0) and np.all(y0 < ub), "y0 is infeasible"
  assert np.any(np.isnan(grad(y0))) == False and  np.any(np.isnan(obj(y0))) == False, "y0 has nan objective or grad"

  # collect the dimension
  problem_data['dim'] = dim

  # call sigup
  method = 'sigup'
  sigup = SIGUP(obj,grad,lb,ub,y0,eps = sig_eps,delta=sig_delta,gamma=sig_gamma,sigma0=sig_sigma0,
          solve_method=sig_solve_method,update_method=sig_update_method)
  try:
    #z = sigup(func,grad,lb,ub,y0,sigma0=sig_sigma0,eps =sig_eps,delta=sig_delta,gamma=sig_gamma,method=sig_method,verbose=False)
    z = sigup.solve()
  except:
    z = sigup.X[np.argmin(sigup.fX)]
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
  opt.set_ftol_rel(ftol)
  opt.set_ftol_abs(ftol)
  opt.set_xtol_rel(xtol)
  try:
    #res = minimize(func,y0,jac=grad,method=method,options={'gtol':gtol,'ftol':ftol,'maxiter':maxiter,'maxfun':maxfun})
    #z = res.x
    z = opt.optimize(y0)
  except:
    z = func.X[-1]
    pass
  X = func.X
  fX = func.fX
  fopt = np.min(fX)
  print(f"{method}: {fopt}")
  method_data = {}
  method_data['method'] = method
  method_data['X'] = X
  method_data['fX'] = fX
  problem_data['runs'].append(method_data)


  # call sigma method with no update
  for sigma0 in sigmoid_fixed_sigmas:
    method = f'sigmoid-fixed-{sigma0}'
    func = eval_wrapper(obj,dim) # wrap the objective
    sig = Sigmoid(sigma=sigma0)
    def objective_with_grad(xx,g):
      f = func(from_unit_cube(sig(xx),lb,ub))
      g[:] = sig.jac(xx) @ np.diag(ub-lb) @ grad(from_unit_cube(sig(xx),lb,ub))
      return f
    opt = nlopt.opt(nlopt.LD_LBFGS, dim)
    opt.set_min_objective(objective_with_grad)
    opt.set_lower_bounds(lb)
    opt.set_upper_bounds(ub)
    opt.set_maxeval(maxfun)
    opt.set_ftol_rel(ftol)
    opt.set_ftol_abs(ftol)
    opt.set_xtol_rel(xtol)
    try:
      z = opt.optimize(y0)
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

  # TODO: call sigup with a fixed update rule

  # TODO: call augmented lagrangian

  # TODO: call projection penalty method

  # save data
  problem_filename = outputdir + f"{pname}.pickle"
  pickle.dump(problem_data,open(problem_filename,"wb"))
  
