import torch
from program import IfThenElse, IntervalBool, AssignStatement, StatementBlock, ReturnStatement, AssertStatement, NoOp
from expression import VarExpr, ConstExpr
from abdomain import AbsInterval
from torch import optim
from torch.distributions import uniform
import scipy.optimize
import numpy as np


class OptimizerState:
    def __init__(self, beta, lambda_const, smooth):
        self.beta = beta
        self.lambda_const = lambda_const
        self.smooth = smooth
        self.loss = torch.tensor([0.0], requires_grad=True)
        self.satisfied = True
        
    def add_loss(self, newloss):
        self.loss = self.loss + newloss
        

def optimize(xL, xH, infer_args, program):   

    beta = 10.0  # starting beta value
    ki = 0.95    # beta multiplier each iteration
    epsb = 0.05 # lower beta cutoff
    
    intmin = 0
    intmax = 100
    distribution = uniform.Uniform(torch.Tensor([intmin]),torch.Tensor([intmax]))
    inferred_paramL = torch.tensor(len(infer_args))
    inferred_paramH = torch.tensor(len(infer_args))
    
    # whether we have an input that satisfies program assertions
    satisfies = False
    while not satisfies:
        inferred_paramL = distribution.sample(inferred_paramL.size())
        inferred_paramH = distribution.sample(inferred_paramH.size())
        while not (inferred_paramL < inferred_paramH).all():
            inferred_paramL = distribution.sample(inferred_paramL.size())
            inferred_paramH = distribution.sample(inferred_paramH.size())
        inferred_paramL.requires_grad = True
        inferred_paramH.requires_grad = True
        
        #with torch.no_grad():
        #    inferred_paramL.fill_(-5.0)
        #    inferred_paramH.fill_(5.0)
        
        interval = None
        optimizer = optim.SGD([inferred_paramL, inferred_paramH], lr = 0.01)
        while beta >= epsb:
            optim_state = OptimizerState(beta=beta, lambda_const=1/2, smooth=True)
            for i in range(500):
                def closure():
                    optimizer.zero_grad()
                    optim_state.loss = torch.tensor([0.0], requires_grad=True)
                    inferred_xL = torch.cat((xL, inferred_paramL), 0)
                    inferred_xH = torch.cat((xH, inferred_paramH), 0)
                    interval = AbsInterval(inferred_xL, inferred_xH, 1.0)
                    #print("in: %s" % (str(interval)))
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
        sat_state = OptimizerState(beta=beta, lambda_const=1/2, smooth=False)
        if program.satisfied_by(interval, sat_state):
            satisfies = True
    return inferred_paramL, inferred_paramH

def optimizeNM(xL, xH, program, num_infer):   
    """
    numvars: The number of variables to infer
    """
    beta = 10.0  # starting beta value
    ki = 0.95    # beta multiplier each iteration
    epsb = 0.05 # lower beta cutoff
    #"i", "isOn", "K", "curL", 'h', 'tOn', 'tOff']
    #inferred_paramL = torch.tensor([0.0, -5.0, 5.0, 0.0, 0.1, 1.0])
    #inferred_paramH = torch.tensor([40.0, -5.0, 5.0, 0.0, 0.1, 1.0])
    intmin = -100
    intmax = 100
    
    # encodes two inputs into one vector
    def toVector(w, z):
        assert w.shape == (num_infer,)
        assert z.shape == (num_infer,)
        return np.hstack([w.flatten(), z.flatten()])

    def toWZ(vec):
        # breaks apart vector into two inputs
        assert vec.shape == (num_infer*2,)
        return vec[:num_infer].reshape(num_infer,), vec[num_infer:].reshape(num_infer,)
    
    def create_interval(inferred_param, xL, xH):
        inferred_paramL, inferred_paramH = toWZ(inferred_param)
        inferred_paramL, inferred_paramH = torch.tensor(inferred_paramL).type(torch.FloatTensor), torch.tensor(inferred_paramH).type(torch.FloatTensor)
        
        inferred_paramL, inferred_paramH = torch.min(inferred_paramL, inferred_paramH), torch.max(inferred_paramL, inferred_paramH)
        
        optim_state.loss = torch.tensor([0.0])
        inferred_xL = torch.cat((xL, inferred_paramL), 0)
        inferred_xH = torch.cat((xH, inferred_paramH), 0)
        return AbsInterval(inferred_xL, inferred_xH, 1.0)
    
    # whether we have an input that satisfies program assertions
    inferred_param = None
    satisfies = False
    while not satisfies:
        inferred_paramL = np.random.rand(num_infer)*(intmax-intmin) + intmin
        inferred_paramH = np.random.rand(num_infer)*(intmax-intmin) + intmin
        while not (inferred_paramL < inferred_paramH).all():
            inferred_paramL = np.random.rand(num_infer)*(intmax-intmin) + intmin
            inferred_paramH = np.random.rand(num_infer)*(intmax-intmin) + intmin
            
        inferred_param = toVector(inferred_paramL, inferred_paramH)
        
        while beta >= epsb:
            optim_state = OptimizerState(beta=beta, lambda_const=1/2, smooth=True)
            #print("Beta: %f" % beta)
            def closure(inferred_param, xL, xH):
                interval = create_interval(inferred_param, xL, xH)
                print(interval)
                #print("in: %s" % (str(interval)))
                y = program.propagate(interval, optim_state)
                loss = optim_state.loss 
                #loss.backward()    
                #print("b: %f, Loss: %s,\n in: %s, \n out: %s" % (beta, str(loss.item()), str(interval), str(y)))
                return loss
            result = scipy.optimize.minimize(closure, inferred_param, args=(xL, xH), 
                                    method='Nelder-Mead',
                                    options={'disp':False})#options={'func': None, 'maxiter': None, 'maxfev': None, 'disp': False, 'return_all': False, 'initial_simplex': None, 'xatol': 0.0001, 'fatol': 0.0001, 'adaptive': False})
            inferred_param = result.x
            if not result.success:
                print("Optimization failed")
                exit()
            beta = beta*ki
        # Check answer satisfies requirements
        sat_state = OptimizerState(beta=beta, lambda_const=1/2, smooth=False)
        program.propagate(create_interval(inferred_param, xL, xH), sat_state)
        if sat_state.satisfied:
            satisfies = True
    return create_interval(inferred_param, xL, xH)

def therm_example():
    num_infer = 7
    num_input = 2
    var_list = ['lin', 'ltarget', "i", "isOn", "K", "curL", 'h', 'tOn', 'tOff']
    xL = torch.ones(num_input) * -1.0
    xH = torch.ones(num_input) * 10.0
    diag = torch.diag(torch.ones(len(var_list)))
    var_map = {v: k for (k, v) in enumerate(var_list)}
    var_b = {v: diag[k] for (k, v) in enumerate(var_list)}
    print(var_map)
    program = StatementBlock([
                    # isOn = 0.0;
                    AssignStatement(var_map['isOn'], ConstExpr(0.0)),
                    # K = 0.1;
                    AssignStatement(var_map['K'], ConstExpr(0.1)),
                    # curL = lin;
                    AssignStatement(var_map['curL'], VarExpr(var_map['lin'])),
                    # assert(tOn < tOff) => assert(tOn - tOff <= 0) => assert(tOff - tOn >= 0)
                    AssertStatement(IntervalBool(var_b['tOff'] - var_b['tOn'], 0)),
                    # assert(h >= 0)
                    AssertStatement(IntervalBool(var_b['h'], 0)),
                    # assert(h < 20) => assert(-h + 20 >= 0)
                    AssertStatement(IntervalBool(-var_b['h'], 20.0)),
                    # if isOn >= 0.5 ==> isOn - 0.5 >= 0
                    IfThenElse(IntervalBool(var_b['isOn'], -0.5), 
                               StatementBlock([
                                       # curL = curL + h - K * (curL - lin)
                                       AssignStatement(var_map['curL'], VarExpr(var_map['curL']) + VarExpr(var_map['h']) - VarExpr(var_map['K']) * (VarExpr(var_map['curL']) - VarExpr(var_map['lin']))),
                                       # if (curL - tOff >= 0.0) 
                                       IfThenElse(IntervalBool(var_b['curL'] - var_b['tOff'], 0.0), 
                                                  # isOn = 0.0
                                                  AssignStatement(var_map['isOn'], ConstExpr(0.0)),
                                                  NoOp())]),
                               StatementBlock([
                                       # curL = curL - K * (curL - lin)
                                       AssignStatement(var_map['curL'], VarExpr(var_map['curL']) - VarExpr(var_map['K']) * (VarExpr(var_map['curL']) - VarExpr(var_map['lin']))),
                                       # if (-curL + tOn >= 0)
                                       IfThenElse(IntervalBool(-var_b['curL'] + var_b['tOn'], 0.0),
                                                  # isOn = 1.0
                                                  AssignStatement(var_map['isOn'], ConstExpr(1.0)),
                                                  NoOp())
                                       ])),
                    # assert (-curL + 120 > 0)
                    AssertStatement(IntervalBool(-var_b['curL'], 120.0)),
                    # return curL - ltarget
                    ReturnStatement(VarExpr(var_map['curL']) - VarExpr(var_map['ltarget']))
                    ])
    #infer = torch.tensor([5.0, -5.0, 5.0, 0.0, 0.1, 1.0])
    #print("xL: %s, xH: %s, infer: %s" % (str(xL.size()), str(xH.size()), str(infer.size())))
    print("Arguments are found to be: ")
    print(optimizeNM(xL, xH, program, num_infer))

def repl(program):
    inptL = torch.tensor([-1.0, -1.0])
    inptH = torch.tensor([1.0, 1.0])
    beta = 1.0
    
    while(True):
        cmd = input("> ")
        cmds = cmd.split(" ")
        if len(cmds) == 3 and cmds[1] == '=':
            if cmds[0] == 'xL':
                inptL[0] = float(cmds[2])
            elif cmds[0] == 'xH':
                inptH[0] = float(cmds[2])
            elif cmds[0] == 'cL':
                inptL[1] = float(cmds[2])
            elif cmds[0] == 'cH':
                inptH[1] = float(cmds[2])
            elif cmds[0] == 'beta':
                beta = float(cmds[2])
            else:
                print("Unknown var")
        elif cmd == 'run':
            interval = AbsInterval(inptL, inptH, 1.0)
            optim_state = OptimizerState(beta=beta, lambda_const=1/2, smooth=True)
            print("Output:")
            print(program.propagate(interval, optim_state))
        else:
            print("Usage:")
            print("\tx = v\t x is a name of a variable, v is a value")
            print("Variables are xL, xH, cL, cH, beta")
            print("\trun\t runs the program")

def basic_example():
    # Create random input and output data
    #xL = torch.ones((1, 2), dtype=torch.float16, requires_grad=True)
    #xH = torch.ones((1, 2), dtype=torch.float16, requires_grad=True)
    xL = torch.tensor([float(-10.0)])
    xH = torch.tensor([float(1.0)])
    var_list = ['x', 'c']
    diag = torch.diag(torch.ones(len(var_list)))
    var_map = {v: k for (k, v) in enumerate(var_list)}
    var_b = {v: diag[k] for (k, v) in enumerate(var_list)}
    # x in [-10, 1.0]
    # c approaches 0 but c < 0
    program = StatementBlock([
                    # if c >= 0
                    IfThenElse(IntervalBool(var_b['c'], 0.0), 
                                   # x = 2x + c
                                  AssignStatement(var_map['x'], ConstExpr(2.0) * VarExpr(var_map['x']) + VarExpr(var_map['c'])), 
                                   # x = 700
                                  AssignStatement(var_map['x'], ConstExpr(700.0))),
                    # return x
                    ReturnStatement(VarExpr(var_map['x']))
                    ])
    
    #program = Program([var_map["lin"], var_map["ltarget"]],
    #                  StatementBlock([
    #                          InferAssignStatement(0, )
    #                          ])))
    #repl(program)
    print("Arguments are found to be: ")
    res = optimizeNM(xL, xH, program, 1)
    print("Result:")
    print(res)

if __name__ == '__main__':
    basic_example()
    #therm_example()
    