import unittest
import torch
from abdomain import AbsInterval
from program import IfThenElse, IntervalBool, AssignStatement

class ProgramTest(unittest.TestCase):    
    def test_simple_ite(self):
        # x in [-1, 5]
        x = AbsInterval(torch.tensor([-1.0]), torch.tensor([5.0]), 1.0)
        # if x > c
            # then x = 2x
            # else x = -0.5
        program = IfThenElse(IntervalBool(torch.tensor([1.0]), torch.tensor([0.0])), 
                              AssignStatement(torch.tensor([2.0]), torch.tensor([0.0])), 
                              AssignStatement(torch.tensor([0.0]), torch.tensor([-0.5])))
        # x in [-0.5, 10]
        self.assertEqual(program.propagate(x),
                         AbsInterval(torch.tensor([-0.5]), torch.tensor([10.0]), 1.0))
    def test_ite_alwaystrue(self):
        # x in [-1, 5]
        x = AbsInterval(torch.tensor([-1.0]), torch.tensor([5.0]), 1.0)
        # if 0x + 3 >= 0
            # then x = 2x
            # else x = -5
        program = IfThenElse(IntervalBool(torch.tensor([0.0]), torch.tensor([3.0])), 
                              AssignStatement(torch.tensor([2.0]), torch.tensor([0.0])), 
                              AssignStatement(torch.tensor([0.0]), torch.tensor([-5.0])))
        # x in [-2, 10]
        self.assertEqual(program.propagate(x),
                         AbsInterval(torch.tensor([-2.0]), torch.tensor([10.0]), 1.0))
    def test_ite_alwaystrue2(self):
        # x in [-5, 5]
        x = AbsInterval(torch.tensor([-1.0]), torch.tensor([5.0]), 1.0)
        # if 2x - 11 >= 0
            # then x = 2x
            # else x = -5
        program = IfThenElse(IntervalBool(torch.tensor([2.0]), torch.tensor([-11.0])), 
                              AssignStatement(torch.tensor([2.0]), torch.tensor([0.0])), 
                              AssignStatement(torch.tensor([0.0]), torch.tensor([-5.0])))
        # x in [-5, -5]
        self.assertEqual(program.propagate(x),
                         AbsInterval(torch.tensor([-5.0]), torch.tensor([-5.0]), 1.0))
        
if __name__ == '__main__':
    unittest.main()