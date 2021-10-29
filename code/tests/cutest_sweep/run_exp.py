import numpy as np
import pickle
import pycutest
import os
import sys
from scipy.optimize import minimize,Bounds
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

# generator
sig = Sigmoid()

# sigup parameters
sig_sigma0 = 1.0
sig_eps = 1e-8
sig_gamma = 1.0
sig_method = "BFGS"

# LBFGS params
gtol = sig_eps

for pname in problems['name']:
  print(f"\nproblem: {pname}")
  problem_data = {}
  problem_data['problem'] = pname

  # load problem
  prob = pycutest.import_problem(pname)
  lb,ub = prob.bl,prob.bu
  dim = prob.n
  y0 = prob.x0
  obj = prob.obj
  grad = lambda xx: prob.obj(xx,gradient=True)[1]

  # construct the merit function
  ft = lambda xx: obj(from_unit_cube(sig(xx),lb,ub))
  ft_grad = lambda xx: sig.jac(xx) @ np.diag(ub-lb) @ grad(from_unit_cube(sig(xx),lb,ub))

  # call sigup
  method = 'sigup'
  func = eval_wrapper(obj,dim)
  z = sigup(func,grad,lb,ub,y0,sigma0=sig_sigma0,eps =sig_eps,gamma=sig_gamma,method=sig_method,verbose=False)
  fopt = obj(z)
  X = func.X
  fX = func.fX
  print(f"{method}: {fopt}")
  method_data = {}
  method_data['method'] = method
  method_data['X'] = X
  method_data['fX'] = fX
  method_data['sigma0'] = sig_sigma0
  method_data['gamma'] = sig_gamma
  method_data['eps'] = sig_eps
  method_data['submethod'] = sig_method
  problem_data[method] = method_data

  # projected gradient
  method = "L-BFGS-B"
  func = eval_wrapper(obj,dim)
  res = minimize(func,y0,jac=grad,method=method,options={'gtol':gtol})
  z = res.x
  fopt = obj(z)
  X = func.X
  fX = func.fX
  print(f"{method}: {fopt}")
  method_data = {}
  method_data['method'] = method
  method_data['X'] = X
  method_data['fX'] = fX
  method_data['gtol'] = gtol
  problem_data[method] = method_data

  # TODO: call augmented lagrangian

  # save data
  problem_filename = outputdir + f"{pname}.pickle"
  pickle.dump(problem_data,open(problem_filename,"wb"))
  
