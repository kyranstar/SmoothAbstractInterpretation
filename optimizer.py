
class OptimizerState:
    def __init__(self, beta, lambda_const, smooth):
        self.beta = beta
        self.lambda_const = lambda_const
        self.smooth = smooth