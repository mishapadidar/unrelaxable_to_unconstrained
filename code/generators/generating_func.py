import abc
from abc import ABC, abstractmethod

class GeneratingFunction(object):
    """Base class for generating functions."""

    __metaclass__ = abc.ABCMeta

    def __call__(self,xx):
      return self.eval(xx)

    @abstractmethod
    def eval(self):  # pragma: no cover
        pass

    @abstractmethod
    def jac(self):  # pragma: no cover
        pass

    @abstractmethod
    def inv(self):  # pragma: no cover
        pass