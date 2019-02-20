from __future__ import print_function
import torch
from program import IfThenElse, IntervalBool, AssignStatement
from abdomain import AbsInterval


x = AbsInterval(torch.tensor([-1.0]), torch.tensor([5.0]), 1.0)
print("Input: \n" + str(x))

#Test program 1
# x in [-1, 5]
# if x > 0
    # then x = 2x
    # else x = -0.5
# x in [-0.5, 10]
program1 = IfThenElse(IntervalBool(torch.tensor([1.0]), torch.tensor([0.0])), 
                      AssignStatement(torch.tensor([2.0]), torch.tensor([0.0])), 
                      AssignStatement(torch.tensor([0.0]), torch.tensor([-0.5])))

print("Program1 output: \n" + str(program1.propagate(x)))

#Test program 2
# x in [-1, 5]
# if 0x + 3 >= 0
    # then x = 2x
    # else x = -0.5
# x in [-2, 10]
program2 = IfThenElse(IntervalBool(torch.tensor([0.0]), torch.tensor([3.0])), 
                      AssignStatement(torch.tensor([2.0]), torch.tensor([0.0])), 
                      AssignStatement(torch.tensor([0.0]), torch.tensor([-0.5])))
print("Program2 output: \n" + str(program2.propagate(x)))