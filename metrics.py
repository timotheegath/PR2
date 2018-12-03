import torch
import numpy as np
from scipy.optimize import minimize as min

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


class bileanar_similarity():

    def __init__(self, bilinear_matrix, cosine=True):

        if isinstance(bilinear_matrix, np.ndarray):
            bilinear_matrix = torch.from_numpy(bilinear_matrix).type(Tensor)

        if bilinear_matrix.dtype != Tensor.dtype:
            bilinear_matrix = bilinear_matrix.type(Tensor)
        self.bilinear_matrix = bilinear_matrix
        self.cosine = cosine

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
        if self.cosine:
            Kcos = torch.mm(x.transpose(), y)
            Kcos /= (torch.norm(x, 2) * torch.norm(y, 2))
            return Kcos
        else:
            Km = torch.mm(torch.mm(x.transpose, self.bilinear_matrix), y)
            return Km

class mahalanobis_distance():

    def __init__(self, A):
        if isinstance(A, np.ndarray):
            A = torch.from_numpy(A).type(Tensor)

        if A.dtype != Tensor.dtype:
            A = A.type(Tensor)

        self.A = A

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

        distances = torch.mm(torch.mm((x - y).transpose, self.A), (x - y))
        return distances

def cross_correlation(x, y):

    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).type(Tensor)
        y = torch.from_numpy(y).type(Tensor)

    cc = torch.sum(torch.mul(x, y))
    return cc



#IN PROGRESS
def optimize(model_matrix, training_features, labels, slack_var=False):

    distances = torch.zeros(training_features.shape[1] - 1, training_features.shape[1])
    for i in range(training_features.shape[1]):
        out_d = mahalanobis_distance(training_features, training_features[i], model_matrix)
        distances[:, i] = out_d[out_d.nonzero]

    if isinstance(model_matrix, np.ndarray):
        model_matrix = torch.from_numpy(model_matrix)/type(Tensor)

    #model_parameters = model_matrix.view(-1)

    #min(mahalanobis_distance())





def gaussian_kernel(x, sigma=0.5):

    torch.exp()



