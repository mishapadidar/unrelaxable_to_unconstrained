
import numpy as np

from generating_func import GeneratingFunction

class Sigmoid(GeneratingFunction):
    """Componentwise Sigmoid function: s(x) = 1/(1+e^(-sigma*x))
    """

    def __init__(self, sigma=1):
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
        x = np.array(x)
        return 1/(1+np.exp(-self.sigma*x))

    def jac(self, x):
        """Evaluate the sigmoid jacobian at x
        :param x: Data point
        :type x: numpy.array
        :return: Value at x
        :rtype: 2D numpy.array
        """
        x = np.array(x)
        return np.diag(self.sigma*np.exp(-self.sigma*x)/(1+np.exp(-self.sigma*x))**2)

    def inv(self,x):
        x = np.array(x)
        return np.log(x/(1-x))
