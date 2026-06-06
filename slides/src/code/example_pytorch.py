from torch import nn

class MnistLogistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(784, 10) /
                                    math.sqrt(784)) # W
        self.bias = nn.Parameter(torch.zeros(10)) # b

    def forward(self, xb: torch.Tensor) -> torch.Tensor:
        return xb.matmul(self.weights) + self.bias # xW + b
