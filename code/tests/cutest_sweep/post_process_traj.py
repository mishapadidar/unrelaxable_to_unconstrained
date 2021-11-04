import numpy as np
import glob
import pickle
import pycutest
import sys
sys.path.append("../../utils/")
from compute_lagrange import compute_lagrange

# find the data files
data_loc = "./data/*.pickle"
filelist = glob.glob(data_loc)

profiles = {}
for ff in filelist:
  # data dictionary
  prob_data = pickle.load(open(ff,'rb'))
  pname = prob_data['problem']
  runs = prob_data['runs'] # get the runs
  dim = prob_data['dim']

  # load the cutest problem
  prob = pycutest.import_problem(pname)

  print("")
  print(f"Processing problem {pname}")
  for ii,method_data in enumerate(runs): 
    print(f"Computing KKT violation for {method_data['method']}")
  
    # load the traj
    X = method_data['X']
  
    # storage for worst kkt violation
    KKT = np.zeros(len(X))

    for jj,z in enumerate(X):
     
      # compute the gradient at z
      g_z = prob.obj(z,gradient=True)[1]
      # get the bounds
      lb,ub = prob.bl,prob.bu
      # compute the kkt system at each point
      lam,mu = compute_lagrange(z,g_z,lb,ub)
      kkt = np.max([np.abs(g_z + lam - mu),np.abs(lam*(z-lb)),np.abs(mu*(ub-z))])

      # store the value
      KKT[jj] = kkt

    # save the data
    method_data["KKT"] = KKT
    runs[ii]  = method_data
  
  # save the data
  prob_data['runs'] = runs
  pickle.dump(prob_data,open(ff,"wb"))
