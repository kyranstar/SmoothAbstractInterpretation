import torch
from program import IfThenElse, IntervalBool, AssignStatement, StatementBlock, InferParameter, ReturnStatement
from abdomain import AbsInterval
from torch import optim


class OptimizerState:
    def __init__(self, beta, lambda_const, smooth):
        self.beta = beta
        self.lambda_const = lambda_const
        self.smooth = smooth
        self.loss = torch.tensor([0.0], requires_grad=True)
        
    def add_loss(self, newloss):
        self.loss = self.loss + newloss
        

def optimize(XL, xH, infer_args, program):   

    beta = 1.0  # starting beta value
    ki = 0.8    # beta multiplier each iteration
    epsb = 0.01 # lower beta cutoff
    
    inferred_param = torch.rand(len(infer_args), requires_grad=True)
    
    optimizer = optim.SGD([inferred_param], lr = 0.0001)
    while beta >= epsb:
        optim_state = OptimizerState(beta=beta, lambda_const=1/2, smooth=True)
        for i in range(500):
            def closure():
                optimizer.zero_grad()
                optim_state.loss = torch.tensor([0.0], requires_grad=True)
                inferred_xL = torch.cat((xL, inferred_param), 0)
                inferred_xH = torch.cat((xH, inferred_param), 0)
                interval = AbsInterval(inferred_xL, inferred_xH, 1.0)
                y = program.propagate(interval, optim_state)
                loss = optim_state.loss 
                loss.backward()    
                print("b: %f, Loss: %s,\n in: %s, \n out: %s" % (beta, str(loss.item()), str(interval), str(y)))
                return loss
            optimizer.step(closure)
            #with torch.no_grad():
            #    xL -= learning_rate * xL.grad
            #    xH -= learning_rate * xH.grad
            
        beta = beta*ki
    return inferred_param

if __name__ == '__main__':
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
    
    #var_list = ['h', 'tOn', 'tOff', 'lin', 'ltarget', "isOn", "K", "curL", "i"]
    #var_map = {k:v for (k,v ) in enumerate(var_list)}
    #program = Program([var_map["lin"], var_map["ltarget"]],
    #                  StatementBlock([
    #                          InferAssignStatement(0, )
    #                          ])))
    print(optimize(xL, xH, [InferParameter(var_map['c'])], program))