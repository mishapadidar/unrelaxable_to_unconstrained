
import numpy as np

from generating_func import GeneratingFunction

class Sigmoid(GeneratingFunction):
    """Componentwise Sigmoid function: s(x) = 1/(1+e^(-sigma*x))
    """

    def __init__(self, sigma=1.0):
        """
        Parameter sigma can be positive float or np.array
        with same length as inputs x.
        """
        assert np.all(sigma >= 0), "sigma must be non-negative"
        self.sigma = sigma 

    def eval(self, x):
        """Evaluate the sigmoid at x
        :param x: Data point
        :type x: numpy.array
        :return: Value at x
        :rtype: numpy.array
        """
        if isinstance(self.sigma,float):
            self.sigma = self.sigma*np.ones(len(x))
        x = np.array(x)
        s = np.zeros_like(x)

        # stable computation of sigmoid
        idx_pos = x>0
        s[idx_pos] = 1/(1+np.exp(-self.sigma[idx_pos]*x[idx_pos]))
        idx_neg = x<=0.0
        z= np.exp(self.sigma[idx_neg]*x[idx_neg])
        s[idx_neg] = z/(1.+z)

        return s

    def jac(self, x):
        """Evaluate the sigmoid jacobian at x
        :param x: Data point
        :type x: numpy.array
        :return: Value at x
        :rtype: 2D numpy.array
        """
        x = np.array(x)
        #return np.diag(self.sigma*np.exp(-self.sigma*x)/(1+np.exp(-self.sigma*x))**2)
        y = self.eval(x)
        return np.diag(self.sigma*y*(1-y))

    def inv(self,x):
        x = np.array(x)
        # truncate for stability
        tol = 1e-16
        x[x<tol] = tol
        x[1-x<tol] = 1-tol
        return (np.log(x) - np.log(1-x))/self.sigma
