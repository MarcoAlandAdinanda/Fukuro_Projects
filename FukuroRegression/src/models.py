"""
    There are two type of regression, 
    linear regression and parabolic regression 
"""

import torch

class FukuroLinearRegression(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = torch.nn.Parameter(data=torch.randn(1, requires_grad=True, dtype=torch.float))
        self.bias = torch.nn.Parameter(data=torch.randn(1, requires_grad=True, dtype=torch.float))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias

class FukuroParabolicRegression(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(data=torch.randn(1, requires_grad=True, dtype=torch.float))
        self.b = torch.nn.Parameter(data=torch.randn(1, requires_grad=True, dtype=torch.float))
        self.c = torch.nn.Parameter(data=torch.randn(1, requires_grad=True, dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.a * (x**2) + self.b * x + self.c