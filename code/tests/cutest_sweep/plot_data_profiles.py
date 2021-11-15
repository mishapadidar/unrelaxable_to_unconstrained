import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle
import matplotlib
#matplotlib.rcParams['text.usetex'] = True

# find the data files
data_loc = "./data/*.pickle"
filelist = glob.glob(data_loc)

# data profile tolerance
kkt_tol = 1e-3

profiles = {}
for ff in filelist:
  # data dictionary
  prob_data = pickle.load(open(ff,'rb'))
  runs = prob_data['runs'] # get the runs
  dim = prob_data['dim']
  for dd in runs: 
    # get number of function evaluations
    method = dd['method']
    kkt = dd['KKT']
    # compute the relative tolerance
    rel_tol = kkt_tol*np.linalg.norm(dd["grad_x0"])
    try:
      n_evals = np.where(kkt<rel_tol)[0][0] + 1
    except:
      print('kkt conditions never satisfied by ',method,'on', prob_data['problem'])
      print('best kkt value is ',np.min(kkt))
      n_evals = np.inf
    lhs = n_evals/(dim+1) # lhs of data profile

    # save the data
    if method in profiles: 
      profiles[method].append(lhs)
    else:
      profiles[method] = [lhs]
  
print(profiles)
# now compute the profile for each alpha
alpha = np.linspace(0.1,40,1000)
data_profiles = {}
for method in profiles:
  data_profiles[method] = []
  for aa in alpha:
    frac = np.mean(profiles[method] < aa)
    data_profiles[method].append(frac)
  plt.plot(data_profiles[method],label=method)

plt.title(f"Data profiles for KKT tolerance {kkt_tol}")
plt.xlabel(r"$\alpha$")
plt.legend()
plt.show()
