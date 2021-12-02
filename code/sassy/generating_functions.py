import numpy as np
import matplotlib.pyplot as plt
import matplotlib

plt.rc('text.latex', preamble=r'\usepackage{amsmath,bm}')
matplotlib.rcParams.update({'font.size': 16})

# three generating functions
s_1 = lambda x: 1/(1+np.exp(-4*(x-0.5))) # sigmoid
s_2 = lambda x: np.minimum(1,np.maximum(0,x)) # projection
s_3 = lambda x: 2*np.abs(x/2 - np.floor(x/2+0.5)) # traingle wave
s_4 = lambda x: x - np.floor(x) # sawtooth wave

x = np.linspace(-1,2,200)
plt.plot(x,s_2(x),linewidth=3,label=r'$\pi(x)$')
plt.plot(x,s_3(x),linewidth=3,label=r'$R(x)$')
plt.plot(x,s_1(x),linewidth=3,label=r'$S(x)$')
# plt.plot(x,s_4(x),linewidth=3,label='repetition')
#plt.title("Generating Functions",fontsize=14)
plt.legend(loc="upper left")
plt.show()
