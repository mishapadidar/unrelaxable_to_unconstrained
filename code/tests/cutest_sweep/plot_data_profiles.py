import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle
import matplotlib
matplotlib.rcParams['text.usetex'] = True
plt.rc('text.latex', preamble=r'\usepackage{amsmath,bm}')
matplotlib.rcParams.update({'font.size': 18})

def latex_float(f):
    float_str = "{0:.2g}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        #return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
        return r"10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str


def plot_profile(kkt_tol,filelist):
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
    
  markers = ['s-','.-','o--','^:','+-.','*-']
  skip_methods = ['sigmoid-fixed-adaptive','sigmoid-fixed-0.01','sigmoid-fixed-0.1']
  # now compute the profile for each alpha
  alpha = np.linspace(0,100,1000)
  data_profiles = {}
  ii = 0
  plt.figure(figsize=(9,8))
  for method in profiles:
    data_profiles[method] = []
    for aa in alpha:
      frac = np.mean(profiles[method] < aa)
      data_profiles[method].append(frac)
  
    # skip some methods
    if method in skip_methods:
      continue
    # make labels
    label=method
    if 'sigmoid-fixed' in method:
      label = method.split("-")[-1]
      label = '$\sigma = '+label+ "$"
    elif 'sigup' in method:
      #label = method.split("-")[-1]
      #label = 'sigup $\sigma_0 = '+label+ "$"
      label = 'AdaWarp'
    plt.plot(alpha,data_profiles[method],markers[ii],linewidth=3,markersize=9,label=label,markevery=75)
    ii +=1
  
  plt.title(r"$\tau= %s$"%latex_float(kkt_tol))
  plt.xlabel(r"Normalized number of function evaluations, $\alpha$",fontsize=20)
  plt.legend(loc=4)
  plt.tight_layout()
  plt.yticks([0.0,0.2,0.4,0.6,0.8,1.0])
  plt.show()


# find the data files
data_loc = "./data/*.pickle"
filelist = glob.glob(data_loc)
print(filelist)

# plot 1
# data profile tolerance
kkt_tol = 1e-2
plot_profile(kkt_tol,filelist)
kkt_tol = 1e-4
plot_profile(kkt_tol,filelist)


# Appendix plot: no active constraints at solution
inactive = ['CHEBYQAD', 'DEVGLA1B', 'DIAGPQB', 'DIAGPQE', 'DIAGPQT', 'HART6', 'HS110', 'HS38', 'LEVYMONT', 'LEVYMONT10', 'LEVYMONT6', 'LEVYMONT7', 'LEVYMONT8', 'LEVYMONT9', 'POWERSUMB', 'PROBPENL', 'QINGB', 'SANTALS', 'SINEALI', 'SPECAN', 'STRTCHDVB', 'TRIGON1B']
filelist1 = [ff for ff in filelist if ff.split('/')[-1].split('.')[0] in inactive]
#print(filelist1)
kkt_tol = 1e-2
plot_profile(kkt_tol,filelist1)
kkt_tol = 1e-4
plot_profile(kkt_tol,filelist1)


# Appendix plot: at least one active constraint at solution
inactive = ['CHEBYQAD', 'DEVGLA1B', 'DIAGPQB', 'DIAGPQE', 'DIAGPQT', 'HART6', 'HS110', 'HS38', 'LEVYMONT', 'LEVYMONT10', 'LEVYMONT6', 'LEVYMONT7', 'LEVYMONT8', 'LEVYMONT9', 'POWERSUMB', 'PROBPENL', 'QINGB', 'SANTALS', 'SINEALI', 'SPECAN', 'STRTCHDVB', 'TRIGON1B']
filelist2 = [ff for ff in filelist if ff.split('/')[-1].split('.')[0] not in inactive]
#print(filelist2)
kkt_tol = 1e-2
plot_profile(kkt_tol,filelist2)
kkt_tol = 1e-4
plot_profile(kkt_tol,filelist2)

