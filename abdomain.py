import abc
import torch
import math

class AbstractObject(object, metaclass=abc.ABCMeta):
    """
    Represents an object in an abstract domain for smooth abstract interpretation.
    """
    
    @abc.abstractmethod
    def affine_transform(self, M, C):
        """
        Computes the affine transformation of the object Mx + C.
        """
        raise NotImplementedError('users must define affine_transform to use this base class')
    @abc.abstractmethod
    def meet(self, b):
        """
        Takes the classical meet between this abstract object and 
        the abstract object representing the points that satisfy 
        boolean condition b. By treating the abstract domain as a lattice, the
        meet operator is defined as the greatest lower bound of the two.
        
        Requires:
            b: a BoolConditional
        """
        raise NotImplementedError('users must define meet to use this base class')
    @abc.abstractmethod
    def join(self, A):
        """
        Takes the classical join between this abstract object and 
        another abstract object. By treating the abstract domain as a lattice,
        the join operator is defined as the least upper bound of the two.
        """
        raise NotImplementedError('users must define join to use this base class')
    @staticmethod
    @abc.abstractmethod
    def smooth_join(obs):
        """
        The novel operation that allows smoothing of programs. Its operation 
        is described in S. Chaudhuri, M. Clochard, and A. Solar-Lezama, 
        "Bridging boolean and quantitative synthesis using smoothed proof 
        search," 2014.
        
        Requires:
            obs: An iterable of abstract objects to join together
        """
        raise NotImplementedError('users must define smooth_join to use this base class')

        
class AbsInterval(AbstractObject):    
    """
    Represents an object in the interval abstract domain. This can represent constraints
    of the form c <= x and x <= c.
    """
    def __init__(self, L, H, alpha):
        """
        Args:
            L: A vector that represents the interval's lower bound in each dimension
            H: A vector that represents the interval's upper bound in each dimension
            alpha: A number that represents how close this interval is to disappearing
        """
        assert(not torch.isnan(L).any())
        assert(not torch.isnan(H).any())
        assert(not math.isnan(alpha))
        assert(not math.isinf(alpha))
        
        self.L = L
        self.H = H
        self.alpha = alpha
    
    def __str__(self):
        return "AbsInterval{%s, %f}" % (str(torch.stack((self.L, self.H)).transpose(0, 1)), self.alpha)
    
    def __repr__(self):
        return str(self)
    
    def __eq__(self, other):
        if isinstance(other, AbsInterval):
            return torch.equal(self.L, other.L) and torch.equal(self.H, other.H) and math.isclose(self.alpha, other.alpha, rel_tol=1e-5)
        return False
        
    def affine_transform(self, M, C):
        """
        Represents the transform Mx + C where x is this interval.
        
        Args:
            M: An nxn matrix of finite real numbers where n is the number of dimensions
            C: An n dimensional vector of finite real numbers
        """
        assert(not torch.isinf(M).any())
        assert(not torch.isinf(C).any())
        assert(not torch.isnan(M).any())
        assert(not torch.isnan(C).any())
        
        newl = M*self.L + C
        newh = M*self.H + C
        # Replace nans, created with 0*inf, with 0
        newl[newl != newl] = 0.0
        newh[newh != newh] = 0.0
        return AbsInterval(newl, newh, self.alpha)
    
    def meet(self, b, optim_state):
        """
        Represents this interval meeting with an interval boolean condition b. 
        It returns an interval that contains the portion of this interval that satisfies b.
        """        
        # bi*xi + ci >= 0
        # -> x <= c/b if b < 0
        # -> x >= -c/b if b > 0
        boundary_point = torch.div(b.c, b.b) * -((b.b > 0).float())
        
        # First calculate the interval that contains the points satisfying b
        Lb = torch.ones_like(self.L).fill_(float('-inf'))
        Hb = torch.ones_like(self.H).fill_(float('inf'))
        Lb[b.b > 0] = boundary_point[b.b > 0]
        Hb[b.b < 0] = boundary_point[b.b < 0]
        
        # When b is 0, if c > 0 then we get [inf, -inf] (empty interval), else [-inf, inf]
        # This is the condition 0x + c > 0
        Lb[(b.b == 0) & (b.c >= 0)] = float('-inf')
        Lb[(b.b == 0) & (b.c < 0)] = float('inf')
        Hb[(b.b == 0) & (b.c >= 0)] = float('inf')
        Hb[(b.b == 0) & (b.c < 0)] = float('-inf')

        # Now we meet with [Lb, Hb]
        Lout = torch.max(self.L, Lb)
        Hout = torch.min(self.H, Hb)
        
        # When our lower bound is higher than our upper bound, set to the empty interval
        emptymask = Lout > Hout
        Lout[emptymask] = float('inf')
        Hout[emptymask] = float('-inf')
        
        return AbsInterval(Lout, Hout, self.alpha * self.rho(Lout, Hout, optim_state))

    def join(self, A):
        """
        Sound-join of two intervals. Returns the hull of the two.
        """
        alpha = min(self.alpha + A.alpha, 1)
        lowbound = torch.min(self.L, A.L)
        highbound = torch.max(self.H, A.H)
        
        return AbsInterval(lowbound, highbound, alpha)
    
    @staticmethod
    def smooth_join(obs):
        """
        Requires:
            obs: an iterable of AbsInterval objects
        """
        # just do finite for now
        assert(not torch.isinf(torch.tensor([AbsInterval.volume(O.L, O.H) for O in obs])).any())
        
        alphas = torch.tensor([o.alpha for o in obs])
        a_sum = torch.sum(alphas)
        ap = alphas/torch.max(alphas)
        # Centers and widths for each obj
        c = torch.stack([(o.L + o.H)/2 for o in obs])
        w = torch.stack([(o.H - o.L)/2 for o in obs])
    
        # the center of gravity of interval centers
        c_out = (torch.sum(c*alphas[:, None], 0))/a_sum

        cp = ap[:, None]*c + (1 - ap[:, None])*c_out
        wp = ap[:, None]*w
        
        # All of the created intervals
        Li = cp - wp
        Hi = cp + wp
        
        # Create bounding interval
        L, _ = torch.min(Li, 0)
        H, _ = torch.max(Hi, 0)
        
        a_out = min(a_sum, 1.0)
        return AbsInterval(L, H, a_out)

    
    def rho(self, L, H, optim_state):
        """
        Creates an updated alpha value based on the new interval.
        i.e., given the L and H calculated in O.meet(b), returns rho(O, b).
        """        
        f_beta = min(1/2, optim_state.lambda_const * optim_state.beta)
        if math.isinf(AbsInterval.volume(self.L, self.H)):
            return 1 - f_beta
        else:
            return min(1, AbsInterval.volume(L, H) / (f_beta * AbsInterval.volume(self.L, self.H)))
    
    @staticmethod
    def volume(L, H):
        vol = abs(torch.prod(H - L, 0))
        if math.isnan(vol):
            return float('inf')
        return vol
    
