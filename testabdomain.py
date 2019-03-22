import unittest
import torch
from abdomain import AbsInterval
from program import IntervalBool
from optimizer import OptimizerState
import numpy as np

from hypothesis import given
import hypothesis.strategies as st

t_os = OptimizerState(beta=1, lambda_const=1/2, smooth=False)

class IntervalTest(unittest.TestCase):    
    def test_meet_div0_true(self):
        # [1, 5] meet (0x + 1 > 0) -> [1, 5]
        val = AbsInterval(torch.tensor([1.0]), torch.tensor([5.0]), 1.0)
        self.assertEqual(val.meet(IntervalBool(torch.tensor([0.0]), torch.tensor([1.0])), t_os), 
                         AbsInterval(torch.tensor([1.0]), torch.tensor([5.0]), 1.0))
    def test_meet_div0_false(self):
        # [1, 5] meet (0x - 1 > 0) -> empty
        val = AbsInterval(torch.tensor([1.0]), torch.tensor([5.0]), 1.0)
        self.assertEqual(val.meet(IntervalBool(torch.tensor([0.0]), torch.tensor([-1.0])), t_os), 
                         AbsInterval(torch.tensor([float('inf')]), torch.tensor([float('-inf')]), 0.0))
         # ([-1, 5], [1, 3]) meet 0x1 >= 0 and 0x2 - 1 >= 0 => EMPTY
        val = AbsInterval(torch.tensor([-1.0, 1.0]), torch.tensor([5.0, 3.0]), 1.0)
        self.assertEqual(val.meet(IntervalBool(torch.tensor([0.0, 0.0]), torch.tensor([0.0, -1.0])), t_os), 
                         AbsInterval(torch.tensor([float('inf'), float('inf')]), torch.tensor([float('-inf'), float('-inf')]), 0.0))
        # ([-1, 5], [1, 3]) meet 0x1 >= 0 and 0x2 + 1 >= 0 => ([-1, 5], [1, 3])
        val = AbsInterval(torch.tensor([-1.0, 1.0]), torch.tensor([5.0, 3.0]), 1.0)
        self.assertEqual(val.meet(IntervalBool(torch.tensor([0.0, 0.0]), torch.tensor([0.0, 1.0])), t_os), 
                         val)
    
    
    def test_meet_prog(self):
        x1 = AbsInterval(torch.tensor([-10.0, 0.5]), torch.tensor([1.0, 0.5]), 1.0)
        x2 = AbsInterval(torch.tensor([-10.0, -0.5]), torch.tensor([1.0, -0.5]), 1.0)
        x3 = AbsInterval(torch.tensor([-10.0, -0.5]), torch.tensor([1.0, 0.5]), 1.0)
        b = IntervalBool(torch.tensor([0.0, 1.0]), torch.tensor([0.0, 0.0]))
        self.assertEqual(x1.meet(b, t_os), x1)
        self.assertEqual(x2.meet(b, t_os), AbsInterval(torch.tensor([float('inf'), float('inf')]), torch.tensor([float('-inf'), float('-inf')]), 1.0))
        self.assertEqual(x3.meet(b, t_os), AbsInterval(torch.tensor([-10.0, 0.0]), torch.tensor([1.0, 0.5]), 1.0))
    
    def test_meet_basic(self):
        # [-1, 5] meet x > 0 => [0, 5]
        val = AbsInterval(torch.tensor([-1.0]), torch.tensor([5.0]), 1.0)
        self.assertEqual(val.meet(IntervalBool(torch.tensor([1.0]), torch.tensor([0.0])), t_os), 
                         AbsInterval(torch.tensor([0.0]), torch.tensor([5.0]), 1.0))
        # [-1, 5] meet x - 3 > 0 => [3, 5]
        val = AbsInterval(torch.tensor([-1.0]), torch.tensor([5.0]), 1.0)
        out = val.meet(IntervalBool(torch.tensor([1.0]), torch.tensor([-3.0])), t_os)
        self.assertEqual(out.L, torch.tensor([3.0]))
        self.assertEqual(out.H, torch.tensor([5.0]))
        # [-5, 3] meet 2x + 3 > 0 => [3, 5]
        val = AbsInterval(torch.tensor([-5.0]), torch.tensor([3.0]), 1.0)
        self.assertEqual(val.meet(IntervalBool(torch.tensor([2.0]), torch.tensor([3.0])), t_os), 
                         AbsInterval(torch.tensor([-1.5]), torch.tensor([3.0]), 1.0))
        
        # ([-1, 5], [1, 3]) meet x1 > 0 and x2 < 0 => ([0, 5], [inf, -inf])
        val = AbsInterval(torch.tensor([-1.0, 1.0]), torch.tensor([5.0, 3.0]), 1.0)
        self.assertEqual(val.meet(IntervalBool(torch.tensor([1.0, -1.0]), torch.tensor([0.0, 0.0])), t_os), 
                         AbsInterval(torch.tensor([float('inf'), float('inf')]), torch.tensor([float('-inf'), float('-inf')]), 0.0))

class IntervalSmoothTest(unittest.TestCase):   
    def test_smooth_join_basic(self):
        v1 = AbsInterval(torch.tensor([-1.0, 1.0, 1.0]), torch.tensor([5.0, 3.0, 3.0]), 1.0)
        v2 = AbsInterval(torch.tensor([-1.0, 1.0, 1.0]), torch.tensor([5.0, 3.0, 3.0]), 1.0)
    
        self.assertEqual(AbsInterval.smooth_join([v1, v2]), 
                         AbsInterval(torch.tensor([-1.0, 1.0, 1.0]), torch.tensor([5.0, 3.0, 3.0]), 1.0))
    
    def test_size_decreases_with_alpha(self):
        last_width = torch.tensor([0.0, 0.0])
        for a in np.arange(0.01, 1.5, 0.02):
            # Keep one alpha 1.0, increase the other
            v1 = AbsInterval(torch.tensor([2.0, 1.0]), torch.tensor([4.0, 3.0]), 1.0)
            v2 = AbsInterval(torch.tensor([-1.0, 0.0]), torch.tensor([5.0, 4.0]), a)
        
            joined = AbsInterval.smooth_join([v1, v2])
            hard_joined = v1.join(v2)
            
            # Our interval should get bigger at all iterations
            self.assertTrue((joined.H - joined.L >= last_width).all())
            # Our interval should be inside a hard join
            self.assertTrue((joined.L >= hard_joined.L).all())
            self.assertTrue((joined.H <= hard_joined.H).all())
            # Our interval should be bigger than the interval with alpha=1.0
            self.assertTrue((joined.H >= v1.H).all())
            self.assertTrue((joined.L <= v1.L).all())
            last_width = joined.H -joined.L
        
    def test_smoothjoin_alpha(self):
        a = AbsInterval(torch.tensor([0.0, 1.0]), torch.tensor([5.0, 6.0]), 0.6)
        b = AbsInterval(torch.tensor([-2.0, 0.0]), torch.tensor([-1.0, 1.0]), 0.4)
        
        res = AbsInterval.smooth_join([a, b])
        print(res)
        self.assertEqual(res.alpha, 1.0)
        
        #for a in np.arange(0.01, 1.5, 0.02):
        #    a = AbsInterval(torch.tensor([0.0, 1.0]), torch.tensor([5.0, 6.0]), a)
        #    b = AbsInterval(torch.tensor([-2.0, 0.0]), torch.tensor([-1.0, 1.0]), 1-a)
        #    res = AbsInterval.smooth_join([a, b])
        #    center = res.L + (res.H - res.L)/2
            
    #def test_grad(self):
    #    v1 = AbsInterval(torch.tensor([-1.0, 1.0, 1.0], requires_grad=True), torch.tensor([5.0, 3.0, 3.0], requires_grad=True), 1.0)
    #    v2 = AbsInterval(torch.tensor([-1.0, 1.0, 1.0], requires_grad=True), torch.tensor([5.0, 3.0, 3.0], requires_grad=True), 1.0)
    #
    #    joined = AbsInterval.smooth_join([v1, v2])
    #    print(joined)
    #    joined.L.backward(torch.tensor([1.0, 1.0, 1.0]))
    #    print(v1.H.grad)
        
if __name__ == '__main__':
    unittest.main()
    #suite = unittest.TestSuite()
    #suite.addTest(IntervalTest("test_meet_prog"))
    #runner = unittest.TextTestRunner()
    #runner.run(suite)