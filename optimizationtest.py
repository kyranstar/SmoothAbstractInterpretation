import unittest
import torch
from abdomain import AbsInterval
from program import IfThenElse, IntervalBool, AssignStatement, StatementBlock, ReturnStatement, InferParameter
from optimizer import OptimizerState, optimize



class OptimizationTest(unittest.TestCase):    
    def simple_ite_optim(self):
        # Create random input and output data
        #xL = torch.ones((1, 2), dtype=torch.float16, requires_grad=True)
        #xH = torch.ones((1, 2), dtype=torch.float16, requires_grad=True)
        xL = torch.tensor([-10.0])
        xH = torch.tensor([1.0])
        var_map = {'x': 0, 'c' : 1}
        # x in [-10, 1.0]
        # c approaches 0 but c < 0
        program = StatementBlock([
                        # if c <= 0
                        IfThenElse(IntervalBool(torch.tensor([0.0, 1.0]), torch.tensor([0.0, 0.0])), 
                                       # x = 2x + c
                                      AssignStatement(torch.tensor([[2.0, 1.0], 
                                                                   [0.0, 1.0]]), torch.tensor([0.0, 0.0])), 
                                       # x = 700
                                      AssignStatement(torch.tensor([[0.0, 0.0], 
                                                                   [0.0, 1.0]]), torch.tensor([700.0, 0.0]))),
                        # return x
                        ReturnStatement(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 0.0]))
                        ])
        result_c = optimize(xL, xH, [InferParameter(var_map['c'])], program)[0]
        self.assertTrue(result_c >= 0.0)
        self.assertTrue(result_c <= 0.25)
     
if __name__ == '__main__':
    unittest.main()
    #suite = unittest.TestSuite()
    #suite.addTest(ProgramTest("test_simple_ite_full"))
    #runner = unittest.TextTestRunner()
    #runner.run(suite)