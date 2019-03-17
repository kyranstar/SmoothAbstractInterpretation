import abc
from functools import reduce
import torch
from abdomain import AbsInterval


class ProgramStatement(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def propagate(self, A, optim_state):
        """
        Taking an abstract object A, this represents applying the semantics of 
        this program statement to A.
        
        Requires:
            A: The abstract object to manipulate.
            optim_state: 
        """
        raise NotImplementedError('users must define propagate to use this base class')
        
class AssignStatement(ProgramStatement):
    """
    Represents assignment statements of the form x := Mx + C
    Arguments:
        M: A tensor of size bxb
        C: A tensor of size b 
    """
    def __init__(self, M, C):
        self.M = M
        self.C = C
    
    def propagate(self, A, optim_state):
        return A.affine_transform(self.M, self.C)
    
class StatementBlock(ProgramStatement):
    """
    Represents multiple statements in a row: S1; S2; S3...
    """
    def __init__(self, statements):
        self.statements = statements
    
    def propagate(self, A, optim_state):
        res = A
        for stmt in self.statements:
            res = stmt.propagate(res, optim_state)
        return res
    
class IfThenElse(ProgramStatement):
    """
    Represents an if then else statement:
        if b then:
            s1;
        else:
            s2;
    Requires:
        b: A BoolConditional
        s1: the "then" block
        s2: the "else" block
    """
    def __init__(self, b, s1, s2):
        self.b = b
        self.s1 = s1
        self.s2 = s2
    
    def propagate(self, A, optim_state):
        res1 = self.s1.propagate(A.meet(self.b, optim_state), optim_state)
        res2 = self.s2.propagate(A.meet(self.b.negate(), optim_state), optim_state)
        if optim_state.smooth:
            return res1.smooth_join([res1, res2])
        else:
            return res1.join(res2)

class AssertStatement(ProgramStatement):
    def __init__(self, b):
        self.b = b
        
    def propagate(self, A, optim_state):
        optim_state.add_loss(self.b, A)
        return A

class BoolConditional(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def negate(self):
        raise NotImplementedError('users must define negate to use this base class')
    
    
class InferParameter:
    def __init__(self, ind):
        self.ind = ind
        
class ReturnStatement(ProgramStatement):
    def __init__(self, v, c):
        self.v = v
        self.c = c
        
    def propagate(self, A, optim_state):
        # TODO
        Ap = AbsInterval(self.v*A.L + self.c, self.v*A.H + self.c, A.alpha)
        optim_state.add_loss(Ap.signed_volume())
        return A
        
class IntervalBool(BoolConditional):
    """
    Represents a boolean conditional for the interval domain. 
    It says that for each dimension i, bi*xi + ci >= 0, where xi
    
    Example:
        To represent 2*x1 >= 1000 && x2 <= 5: b = [2, -1], c = [-1000, 5]
    
    Requires:
        b: 
        c:
    """
    def __init__(self, b, c):
        assert(not torch.isinf(b).any())
        assert(not torch.isinf(c).any())
        assert(not torch.isnan(b).any())
        assert(not torch.isnan(c).any())
        self.b = b
        self.c = c
        
    def negate(self):
        return IntervalBool(-self.b, -self.c)
    
    def __str__(self):
        return "IntervalBool{b=%s, c=%s" % (str(self.b), str(self.c))