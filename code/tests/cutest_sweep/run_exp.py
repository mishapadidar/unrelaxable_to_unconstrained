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
from sigup import sigup
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
sig_delta  = 0.0
sig_gamma  = 1.0
sig_method = "BFGS"

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

  # construct the merit function
  sig = Sigmoid()
  ft = lambda xx: obj(from_unit_cube(sig(xx),lb,ub))
  ft_grad = lambda xx: sig.jac(xx) @ np.diag(ub-lb) @ grad(from_unit_cube(sig(xx),lb,ub))

  # collect the dimension
  problem_data['dim'] = dim

  # call sigup
  method = 'sigup'
  func = eval_wrapper(obj,dim)
  try:
    z = sigup(func,grad,lb,ub,y0,sigma0=sig_sigma0,eps =sig_eps,delta=sig_delta,gamma=sig_gamma,method=sig_method,verbose=False)
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
  method_data['sigma0'] = sig_sigma0
  method_data['gamma'] = sig_gamma
  method_data['eps'] = sig_eps
  method_data['delta'] = sig_delta
  method_data['submethod'] = sig_method
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
  method_data['gtol'] = gtol
  problem_data['runs'].append(method_data)

  # TODO: call augmented lagrangian

  # save data
  problem_filename = outputdir + f"{pname}.pickle"
  pickle.dump(problem_data,open(problem_filename,"wb"))
  
