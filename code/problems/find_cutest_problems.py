import pycutest
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.optimize import Bounds

problems = pycutest.find_problems(constraints='B',regular=True)
print('starting with problems')
print(problems)

good_probs = []
for pname in problems:
    # load the problem
    prob = pycutest.import_problem(pname)
    # remove unbounded problems
    if np.any(prob.bu>1e10): # no infinite bounds
        continue
    elif np.any(prob.bl<-1e10): # no infinite bounds
        continue
    # remove probs that are too high dim
    elif prob.n > 1000:
      continue
    # remove low dim problems
    elif prob.n < 3:
      continue
    # ensure there are no other constraints
    elif prob.m != 0:
      continue
    # make a dictionary of problem attributes
    prop = pycutest.problem_properties(pname)
    attrib = {}
    attrib['name'] = prob.name
    attrib['dim'] = prob.n
    attrib['degree'] = prop['degree']
    attrib['objective'] = prop['objective']
    # optimize to find the activities
    bounds = Bounds(prob.bl,prob.bu)
    res = minimize(prob.obj,prob.x0,method='L-BFGS-B',bounds=bounds,options={'gtol':1e-6})
    xopt = res.x
    atol=1e-2
    attrib['n_activities'] = np.sum(np.isclose(res.x,prob.bl,atol=atol)) + np.sum(np.isclose(res.x,prob.bu,atol=atol))
    # save the problem
    good_probs.append(attrib)

# convert to a pandas df
df = pd.DataFrame(good_probs)

print("")
print(f"found {len(good_probs)} problems:")
print(df)
pd.to_pickle(df,open("cutest_problems.pickle","wb"))
