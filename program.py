import abc
from functools import reduce

class ProgramStatement(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def propagate(self, A):
        """
        Taking an abstract object A, this represents applying the semantics of 
        this program statement to A.
        """
        raise NotImplementedError('users must define propagate to use this base class')
        
class AssignStatement(ProgramStatement):
    """
    Represents assignment statements of the form x := Mx + C
    """
    
    def __init__(self, M, C):
        self.M = M
        self.C = C
    
    def propagate(self, A):
        return A.affine_transform(self.M, self.C)
    
class StatementBlock(ProgramStatement):
    def __init__(self, statements):
        self.statements = statements
    
    def propagate(self, A):
        res = A
        for stmt in self.statements:
            res = stmt.propagate(res)
        return res
    
class IfThenElse(ProgramStatement):
    def __init__(self, b, s1, s2):
        self.b = b
        self.s1 = s1
        self.s2 = s2
    
    def propagate(self, A):
        res1 = self.s1.propagate(A.meet(self.b))
        res2 = self.s2.propagate(A.meet(self.b.negate()))
        return res1.join(res2)
    
class BoolConditional(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def negate(self):
        raise NotImplementedError('users must define negate to use this base class')
    
class IntervalBool(BoolConditional):
    """
    Represents a boolean conditional for the interval domain. 
    It says that for each dimension i, bi*xi + ci >= 0, where xi
    
    Example:
        To represent 2*x1 > 1000 && x2 < 5: b = [2, -1], c = [-1000, 5]
    
    Requires:
        b: 
    """
    def __init__(self, b, c):
        self.b = b
        self.c = c
        
    def negate(self):
        return IntervalBool(-self.b, -self.c)
    
    def __str__(self):
        return "IntervalBool{b=%s, c=%s" % (str(self.b), str(self.c))