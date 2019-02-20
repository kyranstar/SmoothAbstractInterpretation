import abc
import torch
import math

class AbstractObject(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def affine_transform(self, M, C):
        raise NotImplementedError('users must define affine_transform to use this base class')
    @abc.abstractmethod
    def meet(self, b):
        """
        The abstract meet operator defined by A meet B where A is this abstract
        object and B is the abstract object representing the points that satisfy 
        boolean condition b.
        
        Requires:
            b: a boolean condition
        """
        raise NotImplementedError('users must define meet to use this base class')
    @abc.abstractmethod
    def join(self, A):
        raise NotImplementedError('users must define join to use this base class')
    @abc.abstractmethod
    def smooth_join(self, A):
        raise NotImplementedError('users must define smooth_join to use this base class')

        
        
class AbsInterval(AbstractObject):    
    def __init__(self, L, H, alpha):
        """
        Args:
            L: A vector that represents the interval's lower bound in each dimension
            H: A vector that represents the interval's upper bound in each dimension
            alpha: A number that represents how close this interval is to disappearing
        """
        self.L = L
        self.H = H
        self.alpha = alpha
    
    def __str__(self):
        return "AbsInterval{%s, %f}" % (str(torch.cat([self.L, self.H], 0)), self.alpha)
    
    def __repr__(self):
        return str(self)
    
    def __eq__(self, other):
        """Overrides the default implementation"""
        if isinstance(other, AbsInterval):
            return torch.eq(self.L, other.L) and torch.eq(self.H, other.H) and math.isclose(self.alpha, other.alpha, rel_tol=1e-5)
        return False
        
    def affine_transform(self, M, C):
        """
        Represents the transform Mx + C where x is this interval.
        
        Args:
            M: An nxn matrix where n is the number of dimensions
            C: An n dimensional vector
        """
        return AbsInterval(M*self.L + C, M*self.H + C, self.alpha)
    
    def meet(self, b):
        """
        Represents this interval meeting with an interval boolean condition b. 
        It returns an interval that contains the portion of this interval that satisfies b.
        """        
        # bi*xi + ci >= 0
        # -> x <= c/b if b < 0
        # -> x >= -c/b if b > 0
        boundary_point = torch.div(b.c, b.b) * -((b.b > 0).float())
        
        # First calculate the interval that contains the points satisfying b
        Lb = torch.ones(self.L.shape).fill_(float('-inf'))
        Hb = torch.ones(self.H.shape).fill_(float('inf'))
        
        Lb[b.b > 0] = boundary_point
        Hb[b.b < 0] = boundary_point
        
        # When b is 0, if c > 0 then we get [inf, -inf] (empty interval), else [-inf, inf]
        Lb[b.b == 0 and b.c >= 0] = float('-inf')
        Lb[b.b == 0 and b.c < 0] = float('inf')
        Hb[b.b == 0 and b.c >= 0] = float('inf')
        Hb[b.b == 0 and b.c < 0] = float('-inf')
        

        # Now we meet with [Lb, Hb]
        Lout = torch.max(self.L, Lb)
        Lout[max(self.L, Lb) > min(self.H, Hb)] = float('inf')
        
        Hout = torch.min(self.H, Hb)
        Hout[max(self.L, Lb) > min(self.H, Hb)] = float('-inf')
        
        return AbsInterval(Lout, Hout, self.update_alpha(Lout, Hout))

    def join(self, A):
        """
        Hard-join of two intervals. Returns the hull of the two.
        """
        alpha = min(self.alpha + A.alpha, 1)
        lowbound = torch.min(self.L, A.L)
        highbound = torch.max(self.H, A.H)
        
        return AbsInterval(lowbound, highbound, alpha)
    
    def smooth_join(self, A):
        pass
    
    def update_alpha(self, L, H):
        """
        Creates an updated alpha value based on the new interval.
        i.e., given the L and H calculated in O.meet(b), returns rho(O, b).
        """
        return 1
    
    def volume(self):
        return torch.prod(self.H - self.L, 1)