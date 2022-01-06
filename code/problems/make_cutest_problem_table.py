import pickle
import sys

filename = "cutest_problems.pickle"

# load the test point
with open(filename, "rb") as f:
   d= pickle.load(f)
   n_problems = 40
   table_str = ''
   for ii in range(n_problems):
      table_str +=f"{d['name'][ii]} &{d['objective'][ii]} &{d['dim'][ii]} &{d['n_activities'][ii]}"+"\\" +"\\" 

   print(table_str)
