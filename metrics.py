import torch
import numpy as np

if torch.cuda.is_available():
    Tensor = torch.cuda.FloatTensor
else:
    Tensor = torch.FloatTensor

class H_Gaussian():

    def __init__(self):

        self.Q = torch.zeros((2048, 2048)).type(Tensor)


        self.sigma = torch.zeros((1, )).type(Tensor)
        self.parameters = [self.Q, self.sigma]
        for p in self.parameters:
            torch.nn.init.normal_(p)

    def __call__(self, x, y):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).type(Tensor)
            y = torch.from_numpy(y).type(Tensor)
        if x.dtype != Tensor.dtype:
            x = x.type(Tensor)
            y = y.type(Tensor)

        if x.dim() == 1:
            x = x.unsqueeze(1)
            y = y.unsqueeze(1)

        d = torch.exp(-(torch.mm(torch.mm((x - y).transpose(1, 0), self.Q), x - y))/(2 * self.sigma ** 2))
        return d
def gaussian_kernel(x, sigma=0.5):

    torch.exp()


