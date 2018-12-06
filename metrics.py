import torch
import numpy as np
import data_in_out as io
from torch import autograd
from scipy.optimize import minimize as min
from scipy.io import savemat
import evaluation as eval

if torch.cuda.is_available():
    Tensor = torch.cuda.FloatTensor
    torch.set_default_tensor_type(Tensor)
else:
    Tensor = torch.FloatTensor

# class H_Gaussian():
#
#     # def __init__(self):
#     #
#     #     self.L = torch.tril(torch.rand((2048, 2048)).type(Tensor))
#     #     self.A = torch.mm(self.L, self.L.transpose(1, 0))
#     #     self.sigma = torch.full((1,), 2).type(Tensor)
#     #     self.exp_distance = Mahalanobis()
#     #
#     #     self.parameters = [self.L]
#
#
#     def __call__(self, x, y, A):
#         if isinstance(x, np.ndarray):
#             x = torch.from_numpy(x).type(Tensor)
#             y = torch.from_numpy(y).type(Tensor)
#         exponent = (-exp_distance(x, y, A))/(2 * sigma ** 2)
#
#         d = torch.exp(exponent)
#
#         return d
#
#     def update_A(self):
#         self.A = torch.mm(self.L, self.L.transpose(1, 0))
#     def clamp(self):
#         self.sigma = torch.clamp(self.sigma, min=0.01)
#         self.L = torch.clamp(torch.triu(self.L), min=0, max=0)
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

    def __call__(self, x, y, A):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).type(Tensor)
            y = torch.from_numpy(y).type(Tensor)

        if x.dtype != Tensor.dtype:
            x = x.type(Tensor)
            y = y.type(Tensor)
        if x.dim() == 1:
            x = x.unsqueeze(1)
            # x = x.repeat(y.shape[1], 1)
        if y.dim() == 1:
            y = y.unsqueeze(1)

        distances = torch.matmul(torch.matmul((x - y).transpose(1, 0), A), (x - y))

        return distances

def cross_correlation(x, y):

    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).type(Tensor)
        y = torch.from_numpy(y).type(Tensor)

    cc = torch.sum(torch.mul(x, y))
    return cc

def normalise_features(features):

    norms = torch.norm(features, 2, 1, keepdim=True)
    return features/norms

#IN PROGRESS
def optimize(model_matrix, training_features, labels, slack_var=False):

    distances = torch.zeros(training_features.shape[1] - 1, training_features.shape[1])
    mahalanobis_distance = Mahalanobis(model_matrix)
    for i in range(training_features.shape[1]):
        out_d = mahalanobis_distance(training_features, training_features[i])
        distances[:, i] = out_d[out_d.nonzero]

    if isinstance(model_matrix, np.ndarray):
        model_matrix = torch.from_numpy(model_matrix)/type(Tensor)

def new_Maha(x, y, L, sigma):

    x = x.unsqueeze(1)
    x = torch.mm(L, x)
    y = torch.mm(L, y)

    def kernel(X, Y):
        exponent = -torch.norm((X - Y), 2, 0)/(2*sigma**2)
        # print(exponent)
        return torch.exp(exponent)
    # distance = torch.norm(torch.mm(L, x) - torch.mm(L, y), 2, 0)
    distance = kernel(x, x) + kernel(y, y) - 2*kernel(x, y)

    return distance

def optimize_torch(features, training_indexes, ground_truth, iterations, batch_size=7000):
    L = torch.rand((2048, 2048), requires_grad=True)
    lagrangian = torch.full((1, ), 1, requires_grad=True)
    def get_batch():
        np.random.shuffle(training_indexes)
        batch_index = training_indexes[:batch_size]
        batch_features = features[:, batch_index]
        batch_labels = ground_truth[batch_index]

        return batch_index, batch_features, batch_labels
    training_indexes = training_indexes.astype(np.int32)
    with torch.no_grad():
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features).cpu()
        features = normalise_features(features)
    optimizer = torch.optim.Adadelta(params=[L], lr=0.1)
    scores = []
    losses = []
    for n in range(iterations):
        try:
            L_new = torch.tril(L)

            Lagrag_new = torch.clamp(lagrangian, min=10)

            batch_index, batch_features, batch_labels = get_batch()
            batch_features = batch_features.type(Tensor)
            distances = torch.empty((batch_size, batch_size)).type(Tensor)

            for i in range(batch_size):

                distances[i, :] = new_Maha(batch_features[:, i], batch_features, L_new, torch.full((1,), 1))

                # io.loading_bar(i, batch_size)

            if (distances < 0).any():
                print('Negative distances detected')
            loss = loss1(distances, batch_labels, batch_index, Lagrag_new)
            print(loss, lagrangian)

            loss.backward()
            print(L)
            optimizer.step()

            optimizer.zero_grad()
            ranked_idx, _ = eval.rank(10, distances.clone().detach(), batch_index)
            score_by_query, total_score = eval.compute_score(10, ground_truth, ranked_idx, batch_index)
            # io.display_ranklist(training_indexes[:batch_size],ranked_idx, rank=10, N=3)
            print(total_score)
            scores.append([score_by_query, total_score])
            losses.append(loss.clone().detach())

        except KeyboardInterrupt:

            savemat('training_metrics', {'score': scores, 'losses': losses})





# Implementing loss on slide 87 of distance metrics
def loss3(distances, labels, training_index, lagrangian):

    same_person = torch.from_numpy((labels[:, None] == labels[None, :]).astype(np.int32) - np.eye(labels.shape[0])).cuda().byte()

    # Actual objective function to minimize
    same_distance_score = torch.masked_select(distances, same_person)

    same_distance_score = torch.sum(same_distance_score)  # Dividing by 2 since the matrix is symmetrical ?
    relative_distance_constraint = torch.zeros(1).type(Tensor)


    # Lagrangian trick: second part of objective function, relative distance

    for i in range(distances.shape[0]):

        # same_label = same_person[i, :i]
        for j in range(distances.shape[1]):

            if i == j:
                break

            current_value = distances[i, j]

            if same_person[i, j]:
                constraint = 1 + torch.masked_select(distances[i, j:], 1 - same_person[i, j:]) - current_value
            else:
                constraint = 1 - torch.masked_select(distances[i, j:], same_person[i, j:]) + current_value

            relative_distance_constraint += (lagrangian * torch.sum(constraint))

    # Lagrangian trick, Going from maximize to minimize

    # How do you enforce positive definite matrices ?
    loss = same_distance_score + relative_distance_constraint
    # loss = same_distance_score
    return loss

def loss1(distances, labels, training_index, lagrangian):

    same_person = torch.from_numpy((labels[:, None] == labels[None, :]).astype(np.int32) - np.eye(labels.shape[0])).cuda().byte()

    # Actual objective function to minimize
    same_distance_score = torch.masked_select(distances, same_person)

    same_distance_score = lagrangian * (torch.sum(same_distance_score) - 1)  # Dividing by 2 since the matrix is symmetrical ?

    different_distance = -torch.sum(torch.sqrt(torch.masked_select(distances, 1 - same_person)))



    # Lagrangian trick, Going from maximize to minimize

    # How do you enforce positive definite matrices ?
    loss = same_distance_score + different_distance
    # loss = same_distance_score
    return loss
