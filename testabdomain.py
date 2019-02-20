import unittest
import torch
from abdomain import AbsInterval
from program import IntervalBool


class IntervalTest(unittest.TestCase):    
    def test_meet_div0_true(self):
        # [1, 5] meet (0x + 1 > 0) -> [1, 5]
        val = AbsInterval(torch.tensor([1.0]), torch.tensor([5.0]), 1.0)
        self.assertEqual(val.meet(IntervalBool(torch.tensor([0.0]), torch.tensor([1.0]))), 
                         AbsInterval(torch.tensor([1.0]), torch.tensor([5.0]), 1.0))
    def test_meet_div0_false(self):
        # [1, 5] meet (0x - 1 > 0) -> empty
        val = AbsInterval(torch.tensor([1.0]), torch.tensor([5.0]), 1.0)
        self.assertEqual(val.meet(IntervalBool(torch.tensor([0.0]), torch.tensor([-1.0]))), 
                         AbsInterval(torch.tensor([float('inf')]), torch.tensor([float('-inf')]), 1.0))
         # ([-1, 5], [1, 3]) meet 0x1 > 0 ^ 0x2 - 1 < 0 => ([-1, 5], [inf, -inf])
        val = AbsInterval(torch.tensor([-1.0, 1.0]), torch.tensor([5.0, 3.0]), 1.0)
        self.assertEqual(val.meet(IntervalBool(torch.tensor([0.0, 0.0]), torch.tensor([0.0, -1.0]))), 
                         AbsInterval(torch.tensor([-1.0, float('inf')]), torch.tensor([5.0, float('-inf')]), 1.0))
    
    def test_meet_basic(self):
        # [-1, 5] meet x > 0 => [0, 5]
        val = AbsInterval(torch.tensor([-1.0]), torch.tensor([5.0]), 1.0)
        self.assertEqual(val.meet(IntervalBool(torch.tensor([1.0]), torch.tensor([0.0]))), 
                         AbsInterval(torch.tensor([0.0]), torch.tensor([5.0]), 1.0))
        # [-1, 5] meet x - 3 > 0 => [3, 5]
        val = AbsInterval(torch.tensor([-1.0]), torch.tensor([5.0]), 1.0)
        out = val.meet(IntervalBool(torch.tensor([1.0]), torch.tensor([-3.0])))
        self.assertEqual(out.L, torch.tensor([3.0]))
        self.assertEqual(out.H, torch.tensor([5.0]))
        # [-5, 3] meet 2x + 3 > 0 => [3, 5]
        val = AbsInterval(torch.tensor([-5.0]), torch.tensor([3.0]), 1.0)
        self.assertEqual(val.meet(IntervalBool(torch.tensor([2.0]), torch.tensor([3.0]))), 
                         AbsInterval(torch.tensor([-1.5]), torch.tensor([3.0]), 1.0))
        
        # ([-1, 5], [1, 3]) meet x1 > 0 ^ x2 < 0 => ([0, 5], [inf, -inf])
        val = AbsInterval(torch.tensor([-1.0, 1.0]), torch.tensor([5.0, 3.0]), 1.0)
        self.assertEqual(val.meet(IntervalBool(torch.tensor([1.0, -1.0]), torch.tensor([0.0, 0.0]))), 
                         AbsInterval(torch.tensor([0.0, float('inf')]), torch.tensor([5.0, float('-inf')]), 1.0))
        
        
        
if __name__ == '__main__':
    unittest.main()