import unittest
import torch
from abdomain import AbsInterval
from program import IfThenElse, IntervalBool, AssignStatement, StatementBlock, ReturnStatement
from optimizer import OptimizerState


t_os = OptimizerState(beta=1, lambda_const=1/2, smooth=False)

class ProgramTest(unittest.TestCase):    

    def test_simple_ite(self):
        # x in [-1, 5]
        x = AbsInterval(torch.tensor([-1.0]), torch.tensor([5.0]), 1.0)
        # if x > c
            # then x = 2x
            # else x = -0.5
        program = IfThenElse(IntervalBool(torch.tensor([1.0]), torch.tensor([0.0])), 
                              AssignStatement(torch.tensor([[2.0]]), torch.tensor([0.0])), 
                              AssignStatement(torch.tensor([[0.0]]), torch.tensor([-0.5])))
        # x in [-0.5, 10]
        self.assertEqual(program.propagate(x, t_os),
                         AbsInterval(torch.tensor([-0.5]), torch.tensor([10.0]), 1.0))
    def test_ite_alwaystrue(self):
        # x in [-1, 5]
        x = AbsInterval(torch.tensor([-1.0]), torch.tensor([5.0]), 1.0)
        # if 0x + 3 >= 0
            # then x = 2x
            # else x = -5
        program = IfThenElse(IntervalBool(torch.tensor([0.0]), torch.tensor([3.0])), 
                              AssignStatement(torch.tensor([[2.0]]), torch.tensor([0.0])), 
                              AssignStatement(torch.tensor([[0.0]]), torch.tensor([-5.0])))
        # x in [-2, 10]
        self.assertEqual(program.propagate(x, t_os),
                         AbsInterval(torch.tensor([-2.0]), torch.tensor([10.0]), 1.0))
    def test_ite_alwaysfalse(self):
        # x in [-5, 5]
        x = AbsInterval(torch.tensor([-5.0]), torch.tensor([5.0]), 1.0)
        # if 2x - 11 >= 0
            # then x = 2x
            # else x = -5
        program = IfThenElse(IntervalBool(torch.tensor([2.0]), torch.tensor([-11.0])), 
                              AssignStatement(torch.tensor([[2.0]]), torch.tensor([0.0])), 
                              AssignStatement(torch.tensor([[0.0]]), torch.tensor([-5.0])))
        # x in [-5, -5]
        self.assertEqual(program.propagate(x, t_os),
                         AbsInterval(torch.tensor([-5.0]), torch.tensor([-5.0]), 1.0))
        # Discontinuity exhibit 1: as input bound increases by 1, output jumps by 17
        # x in [-5, 6]
        x = AbsInterval(torch.tensor([-5.0]), torch.tensor([6.0]), 1.0)
        # x in [-5, 12]
        self.assertEqual(program.propagate(x, t_os),
                         AbsInterval(torch.tensor([-5.0]), torch.tensor([12.0]), 1.0))
        
    def test_simple_ite_full(self):
        x1 = AbsInterval(torch.tensor([-10.0, 0.5]), torch.tensor([1.0, 0.5]), 1.0)
        x2 = AbsInterval(torch.tensor([-10.0, -0.5]), torch.tensor([1.0, -0.5]), 1.0)
        
        program = StatementBlock([
                    IfThenElse(IntervalBool(torch.tensor([0.0, 1.0]), torch.tensor([0.0, 0.0])), 
                                  AssignStatement(torch.tensor([[2.0, -1.0], 
                                                               [0.0, 1.0]]), torch.tensor([0.0, 0.0])), 
                                  AssignStatement(torch.tensor([[0.0, 0.0], 
                                                               [0.0, 1.0]]), torch.tensor([700.0, 0.0]))),
                    ReturnStatement(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 0.0]))
                    ])
        self.assertEqual(program.propagate(x1, t_os),
                         AbsInterval(torch.tensor([-20.5, 0.5]), torch.tensor([1.5, 0.5]), 1.0))
        self.assertEqual(program.propagate(x2, t_os),
                         AbsInterval(torch.tensor([700.0, -0.5]), torch.tensor([700.0, -0.5]), 1.0))
if __name__ == '__main__':
    unittest.main()
    #suite = unittest.TestSuite()
    #suite.addTest(ProgramTest("test_simple_ite_full"))
    #runner = unittest.TextTestRunner()
    #runner.run(suite)