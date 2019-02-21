import unittest
import torch
from abdomain import AbsInterval
from program import IntervalBool
from optimizer import OptimizerState
import numpy as np

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
                         AbsInterval(torch.tensor([float('inf')]), torch.tensor([float('-inf')]), 1.0))
         # ([-1, 5], [1, 3]) meet 0x1 > 0 ^ 0x2 - 1 < 0 => ([-1, 5], [inf, -inf])
        val = AbsInterval(torch.tensor([-1.0, 1.0]), torch.tensor([5.0, 3.0]), 1.0)
        self.assertEqual(val.meet(IntervalBool(torch.tensor([0.0, 0.0]), torch.tensor([0.0, -1.0])), t_os), 
                         AbsInterval(torch.tensor([-1.0, float('inf')]), torch.tensor([5.0, float('-inf')]), 1.0))
    
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
        
        # ([-1, 5], [1, 3]) meet x1 > 0 ^ x2 < 0 => ([0, 5], [inf, -inf])
        val = AbsInterval(torch.tensor([-1.0, 1.0]), torch.tensor([5.0, 3.0]), 1.0)
        self.assertEqual(val.meet(IntervalBool(torch.tensor([1.0, -1.0]), torch.tensor([0.0, 0.0])), t_os), 
                         AbsInterval(torch.tensor([0.0, float('inf')]), torch.tensor([5.0, float('-inf')]), 1.0))

class IntervalSmoothTest(unittest.TestCase):   
    #def test_smooth_join_basic(self):
    #    v1 = AbsInterval(torch.tensor([-1.0, 1.0, 1.0]), torch.tensor([5.0, 3.0, 3.0]), 1.0)
    #    v2 = AbsInterval(torch.tensor([-1.0, 1.0, 1.0]), torch.tensor([5.0, 3.0, 3.0]), 1.0)
    
    #    self.assertEqual(AbsInterval.smooth_join([v1, v2]), 
    #                     AbsInterval(torch.tensor([-1.0, 1.0, 1.0]), torch.tensor([5.0, 3.0, 3.0]), 1.0))
    
    def test_size_decreases_with_alpha(self):
        last_width = torch.tensor([6.0, 3.0])
        for a in np.arange(0.9, 0.1, -0.01):
            v1 = AbsInterval(torch.tensor([2.0, 1.0]), torch.tensor([4.0, 3.0]), 1.0)
            v2 = AbsInterval(torch.tensor([-1.0, 1.0]), torch.tensor([5.0, 4.0]), a)
        
            joined = AbsInterval.smooth_join([v1, v2])
            hard_joined = v1.join(v2)
            
            # Our interval should get smaller at all iterations
            self.assertTrue((joined.H - joined.L <= last_width).all())
            # Our interval should be inside a hard join
            self.assertTrue((joined.L >= hard_joined.L).all())
            self.assertTrue((joined.H <= hard_joined.H).all())
            # Our interval should converge to the interval with alpha=1.0
            self.assertTrue((joined.H >= v1.H).all())
            self.assertTrue((joined.L <= v1.L).all())
            last_width = joined.H -joined.L
        
if __name__ == '__main__':
    unittest.main()