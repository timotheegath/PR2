import torch
import numpy as np
from scipy.optimize import minimize as min

if torch.cuda.is_available():
    Tensor = torch.cuda.FloatTensor
else:
    Tensor = torch.FloatTensor

class H_Gaussian():

    def __init__(self):

        A = torch.zeros((2048, 2048)).type(Tensor)
        self.exp_distance = Mahalanobis(A)
        self.sigma = torch.zeros((1, )).type(Tensor)
        self.parameters = [self.exp_distance.A, self.sigma]
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

        d = torch.exp((-self.exp_distance(x, y))/(2 * self.sigma ** 2))
        return d

    def update_A(self, new):
        self.exp_distance.A = new

class BilinearSimilarity():

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

class Mahalanobis():

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
    mahalanobis_distance = Mahalanobis(model_matrix)
    for i in range(training_features.shape[1]):
        out_d = mahalanobis_distance(training_features, training_features[i])
        distances[:, i] = out_d[out_d.nonzero]

    if isinstance(model_matrix, np.ndarray):
        model_matrix = torch.from_numpy(model_matrix)/type(Tensor)

    #model_parameters = model_matrix.view(-1)

    #min(mahalanobis_distance())

# Implementing loss on slide 87 of distance metrics
def loss3(distances, labels, metric):
    
    same_person = torch.from_numpy(labels[:, None] == labels[:, None].transpose()).type(torch.LongTensor)
    # different_person = 1 - same_person
    same_distance_score = torch.sum(
        torch.sqrt(distances*same_person))/2  # Dividing by 2 since the matrix is symmetrical ?
    relative_distance = torch.zeros(1).type(Tensor)
    for i in range(distances.shape[0]):
        current_value = distances[i, 0]
        for j in range(distances.shape[1]):
            if i == j:
                break
            relative_distance += (current_value-distances[i, j])

    # Lagrangian trick, Going from maximize to minimize
    distance_difference = -relative_distance + 1
    # How do you enforce positive definite matrices ?
    return same_distance_score + distance_difference
