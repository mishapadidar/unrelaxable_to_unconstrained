import numpy as np

def projected_grad(x,g,lb,ub):
  """
  projected gradient 
  proj(x-grad(x)) - x
  """
  y = np.copy(x-g)
  idx_up = y> ub
  y[idx_up] = ub[idx_up]
  idx_low = y< lb
  y[idx_low] = lb[idx_low]
  return y - x
