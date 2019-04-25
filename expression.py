import abc
import torch


class Expression(object, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def evaluate(self, A):
        """
        Taking an abstract object A, this represents evulating this program 
        statement with A and giving its value.
        
        Requires:
            A: The abstract object to manipulate.
            optim_state: 
        """
        raise NotImplementedError('users must define propagate to use this base class')
    def __add__(self, other):
        return AddOp(self, other)
    def __sub__(self, other):
        return SubOp(self, other)
    def __mul__(self, other):
        return MulOp(self, other)
    def __div__(self, other):
        return DivOp(self, other)
    def __neg__(self):
        return NegOp(self)

class BiOp(Expression):
    """
    """
    def __init__(self, op, e1, e2):
        self.op = op
        self.e1 = e1
        self.e2 = e2
        
    def evaluate(self, A):
        #print("Evaluating %s with e1 %s and e2 %s" % (str(self), str(self.e1.evaluate(A)), str(self.e2.evaluate(A))))
        x1, x2 = self.e1.evaluate(A)
        y1, y2 = self.e2.evaluate(A)        
        L = min(self.op(x1, y1), self.op(x1, y2), self.op(x2, y1), self.op(x2, y2))
        H = max(self.op(x1, y1), self.op(x1, y2), self.op(x2, y1), self.op(x2, y2))
        return clean(L, H)
    
    def __str__(self):
        return "%s(%s, %s)" % (str(self.op), str(self.e1), str(self.e2))    
    
class AddOp(BiOp):
    def __init__(self, e1, e2):
        BiOp.__init__(self, lambda a,b: a + b, e1, e2)
        
    def __str__(self):
        return "(%s + %s)" % (str(self.e1), str(self.e2))  
        
class SubOp(BiOp):
    def __init__(self, e1, e2):
        BiOp.__init__(self, lambda a,b: a - b, e1, e2)
        
    def __str__(self):
        return "(%s - %s)" % (str(self.e1), str(self.e2))  
        
class MulOp(BiOp):
    def __init__(self, e1, e2):
        BiOp.__init__(self, lambda a,b: a * b, e1, e2)
        
    def __str__(self):
        return "(%s * %s)" % (str(self.e1), str(self.e2))  
        
class DivOp(BiOp):
    def __init__(self, e1, e2):
        BiOp.__init__(self, lambda a,b: a / b, e1, e2)
        
    def __str__(self):
        return "(%s / %s)" % (str(self.e1), str(self.e2))  
        
class UnOp(Expression):
    """
    """
    def __init__(self, op, e1):
        self.op = op
        self.e1 = e1
        
    def evaluate(self, A):
        x1, x2 = self.e1.evaluate(A)
        L = min(self.op(x1), self.op(x2))
        H = max(self.op(x1), self.op(x2))
        return clean(L, H)
    
    def __str__(self):
        return "%s(%s)" % (str(self.op), str(self.e1))
    
class NegOp(UnOp):
    def __init__(self, e1):
        UnOp.__init__(self, lambda a: -a, e1)
        
    def __str__(self):
        return "-%s" % (str(self.e1))
    
class VarExpr(Expression):
    def __init__(self, i):
        assert isinstance(i, int)
        self.i = i
    
    def evaluate(self, A):
        #print(A)
        #print(self.i)
        #print(A.L[self.i], A.H[self.i])
        return A.L[self.i], A.H[self.i]
    
    def __str__(self):
        return "Var(%s)" % (str(self.i))
    
class ConstExpr(Expression):
    def __init__(self, c):
        assert isinstance(c, float)
        self.c = torch.Tensor([c])
    def evaluate(self, A):
        return self.c, self.c
    def __str__(self):
        return "Const(%s)" % (str(self.c))
    
def clean(L, H):
    L[L > H] = float('inf')
    H[L > H] = float('-inf')
    L[torch.isnan(L)] = float('inf')
    H[torch.isnan(H)] = float('-inf')
    return L, H