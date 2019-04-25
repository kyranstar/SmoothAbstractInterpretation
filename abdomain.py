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
    def meet(self, b, optim_state):
        """
        Takes the classical meet between this abstract object and 
        the abstract object representing the points that satisfy 
        boolean condition b. By treating the abstract domain as a lattice, the
        meet operator is defined as the greatest lower bound of the two.
        
        Requires:
            b: a BoolConditional
            optim_state:
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
        #assert(not torch.isinf(L).any())
        #assert(not torch.isinf(H).any())
        assert(not math.isnan(alpha))
        assert(not math.isinf(alpha))
        
        self.L = L
        self.H = H
        self.alpha = alpha
    
    def __str__(self):
        return "AbsInterval{L:%s,H:%s, %f}" % (str(self.L), str(self.H), self.alpha)
        #return "AbsInterval{%s, %f}" % (str(torch.stack((self.L, self.H)).transpose(0, 1)), self.alpha)
    
    def __repr__(self):
        return str(self)
    
    def __eq__(self, other):
        if isinstance(other, AbsInterval):
            Leq = torch.all(torch.lt(torch.abs(torch.add(self.L, -other.L)), 1e-12))
            Heq = torch.all(torch.lt(torch.abs(torch.add(self.H, -other.H)), 1e-12))
            return Leq and Heq and math.isclose(self.alpha, other.alpha, rel_tol=1e-5)
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
        
        empties = (self.L == float('inf')) & (self.H == float('-inf'))
        
        newl = torch.mv(M, self.L) + C
        newl[empties] = float('inf')
        newh = torch.mv(M, self.H) + C
        newh[empties] = float('-inf')
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
        
        outInterval = AbsInterval(Lout, Hout, 1.0)
        outInterval.alpha = self.alpha * self.rho(outInterval, optim_state)
        
        if outInterval.is_empty():
            outInterval.L = outInterval.L.fill_(float('inf'))
            outInterval.H = outInterval.H.fill_(float('-inf'))
        
        return outInterval

    def join(self, A):
        """
        Sound-join of two intervals. Returns the hull of the two.
        """
        alpha = min(self.alpha + A.alpha, 1.0)
        lowbound = torch.min(self.L, A.L)
        highbound = torch.max(self.H, A.H)
        
        return AbsInterval(lowbound, highbound, alpha)
    
    @staticmethod
    def smooth_join(obs):
        """
        Requires:
            obs: an iterable of AbsInterval objects
        """
        # TODO implement for infinite intervals
        assert(not torch.isinf(torch.tensor([O.volume() for O in obs])).any())
        
        finite_obs = [o for o in obs if o.alpha > 0.0 and not (o.L.eq(float('-inf')) & o.H.eq(float('inf'))).any()]
        infinite_obs = [o for o in obs if o.alpha > 0.0 and (o.L.eq(float('-inf')) & o.H.eq(float('inf'))).any()]
        
        alphas = torch.tensor([o.alpha for o in finite_obs])
        a_sum = torch.sum(alphas)
        ap = alphas/torch.max(alphas)
        # Centers and widths for each obj
        c = torch.stack([(o.L + o.H)/2 for o in finite_obs])
        w = torch.stack([(o.H - o.L)/2 for o in finite_obs])
        
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
        
        # sum the infinite objects into one universal element
        #inf_ob = AbsInterval(torch.zeros(L.size()), torch.zeros(H.size()), min(sum([o.alpha for o in infinite_obs]), 1.0))
        #for ob in infinite_obs:
        #    inf_ob.L += ob.L
        #    inf_ob.H += ob.H
        
        return AbsInterval(L, H, a_out)

    
    def rho(self, newInt, optim_state):
        """
        Creates an updated alpha value based on the new interval.
        i.e., given the L and H calculated in O.meet(b), returns rho(O, b).
        """        
        f_beta = min(0.5, optim_state.lambda_const * optim_state.beta)
        if math.isinf(self.volume()):
            return 1 - f_beta
        else:
            vol = self.volume()
            if vol == 0:
                return 1.0
            return min(1.0, newInt.volume() / (f_beta * vol))
    
    def volume(self):
        # If we are empty in any dimension, volume is 0
        if ((self.H < self.L).any()):
            return 0.0
        vol = torch.abs((torch.prod(self.H - self.L, 0)))
        if math.isnan(vol):
            return float('inf')
        return vol
    
    def hausdorrf_distance(self, ob):
        # NOT REALLY HAUSDORRF DISTANCE TODO?
        # http://cgm.cs.mcgill.ca/~godfried/teaching/cg-projects/98/normand/main.html
        return torch.abs(self.L - ob.L) + torch.abs(self.H - ob.H)
    
    def signed_volume(self):
        # TODO return loss
        return torch.sum(self.L + self.H)
    
    def is_empty(self):
        return ((self.H == float('-inf')) & (self.L == float('inf'))).any()
