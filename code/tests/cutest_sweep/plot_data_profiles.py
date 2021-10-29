import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle

# find the data files
data_loc = "./data/*.pickle"
filelist = glob.glob(data_loc)

profiles = {}
for ff in filelist:
  # data dictionary
  prob_data = pickle.load(open(ff,'rb'))
  runs = prob_data['runs'] # get the runs
  dim = prob_data['dim']
  for dd in runs: 
    # get number of function evaluations
    method = dd['method']
    n_evals = len(dd['fX'])
    lhs = n_evals/(dim+1) # lhs of data profile

    # save the data
    if method in profiles: 
      profiles[method].append(lhs)
    else:
      profiles[method] = [lhs]
  
print(profiles)
# now compute the profile for each alpha
alpha = np.linspace(0.1,1000,1000)
data_profiles = {}
for method in profiles:
  data_profiles[method] = []
  for aa in alpha:
    frac = np.mean(profiles[method] < aa)
    data_profiles[method].append(frac)
  plt.plot(data_profiles[method],label=method)

plt.legend()
plt.show()
