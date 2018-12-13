import numpy as np
import data_in_out as io
import evaluation as eval
import torch
import metrics as metrics


if torch.cuda.is_available():
    Tensor = torch.cuda.FloatTensor
else:
    Tensor = torch.FloatTensor
torch.set_default_tensor_type(Tensor)

# ----------------------------------------------------------------------------------------------------------------------
class StopCallback():

    def __init__(self, tolerance):
        self.tolerance = tolerance
        self.losses = np.zeros((tolerance,), dtype=np.float32)
        self.iter = 0

    def update_and_analyse(self, newLoss):
        for i in range(self.tolerance-1):
            self.losses[i+1] = self.losses[i]
        self.losses[0] = newLoss

        if self.iter < self.tolerance:
            pass
        elif self.losses[-1] :
            pass


class TrainableMetric():

    def __init__(self, init_params, dims, loss, lagrangian=True, kernel=None):

        self.kernel = kernel

        self.lagrangian = torch.full((1, ), 1, requires_grad=lagrangian)
        self.parameters = {}

        self.parameters['L'] = torch.empty((dims, dims), requires_grad=True)
        self.parameters['L'].data = init_params['L']
        if kernel is 'poly':
            self.parameters['p'] = torch.empty((1, ), requires_grad=True)
            self.parameters['p'].data = init_params['p']
            self.distance = metrics.poly_Maha
        elif kernel is 'RBF':

            self.parameters['sigma'] = torch.empty((1, ), requires_grad=True)
            self.parameters['sigma'].data = init_params['sigma']
            self.distance = metrics.gaussian_Maha
        else:
            self.distance = metrics.mahalanobis_metric
        self.loss = loss

    def __call__(self, features, features_compare=None):

        return self.distance(self.parameters, features, features_compare)

    def objective_function(self, training_features, training_labels):

        distances = self(training_features, training_features)
        loss = self.loss(distances, training_labels, self.lagrangian)

        return loss, distances.clone().detach().cpu().numpy()

    def get_params(self, lagr_lr=0):

        return [{'params': [self.parameters[k] for k in self.parameters.keys()]},
                {'params': self.lagrangian, 'lr': lagr_lr}]


def to_torch(*arrays):
    torch_arrays = []
    for array in arrays:
        if isinstance(array, np.ndarray):
            torch_arrays.append(torch.from_numpy(array).type(Tensor))
        else:
            torch_arrays.append(array)
    if len(arrays) is 1:
        torch_arrays = torch_arrays[0]
    return torch_arrays


def lossA(distances, labels, l):
    distances = distances - torch.diag(distances.diag())
    label_mask = labels.view(1, -1) == labels.view(-1, 1)
    Dw = torch.masked_select(distances, label_mask) - 1
    Db = torch.masked_select(distances, 1 - label_mask)

    objective = l*torch.sum(Dw) - torch.sum(torch.sqrt(Db))
    return objective


def lossC(distances, labels, l):
    distances = distances - torch.diag(distances.diag())
    label_mask = labels.view(1, -1) == labels.view(-1, 1)
    constraint = torch.zeros((1,))
    chosen_rows = np.arange(0, distances.shape[0])
    np.random.shuffle(chosen_rows)
    chosen_rows = chosen_rows[:min(100, distances.shape[0])].tolist()
    for i in chosen_rows:

        same_label_candidates = torch.masked_select(distances[i, :],  label_mask[i, :])
        different_label_candidates = torch.masked_select(distances[i, :],  1 - label_mask[i, :])
        try:
            pair = different_label_candidates[0] - same_label_candidates[0]
            constraint += pair - 1
        except IndexError:

            pass

    # Transform maximisation into minimisation

    same_distances = torch.masked_select(distances, label_mask)

    loss = torch.sum(same_distances) - torch.abs(l)*constraint

    return loss





def create_batch(features, g_t,  train_index, size):
    temp_index = np.arange(0, train_index.shape[0]).astype(np.int32)

    np.random.shuffle(temp_index)
    temp_index = temp_index[:size]
    train_ix = train_index[temp_index].astype(np.int32)
    training_features = to_torch(features[:, train_ix])

    training_labels = g_t[train_ix]

    return to_torch(training_features, training_labels, train_ix)


if __name__ == '__main__':
    BATCHIFY = True
    KERNEL = 'RBF'
    BATCH_SIZE = 2000
    RANK = 10
    SKIP_STEP = 3
    FILENAME = 'test_run'
    NUM_ITER = 1000

    # Feature loading
    features = np.memmap('PR_data/features', mode='r', shape=(14096, 2048), dtype=np.float64).transpose()
    ground_truth = io.get_ground_truth()
    train_ind = io.get_training_indexes()
    query_ind = io.get_query_indexes()
    gallery_ind = io.get_gallery_indexes()

    if not BATCHIFY:
        BATCH_SIZE = train_ind.shape[0]

    init_params = {}

    """Initialise parameters here"""

    matrix = torch.eye(features.shape[0])
    matrix.data = torch.from_numpy(np.linalg.cholesky(np.eye(features.shape[0]))).type(Tensor)
    # matrix = torch.rand((features.shape[0], features.shape[0]), requires_grad=True)

    # matrix.data = (torch.from_numpy(np.linalg.cholesky(np.linalg.inv(np.cov(features[:, train_ind]))).transpose()).type(Tensor))
    # matrix = torch.eye(features.shape[0], requires_grad=True)
    # matrix.data = torch.tril(matrix)

    init_params['L'] = matrix
    # For gaussian kernel
    if KERNEL is 'RBF':

        param2 = torch.rand((1,))
        param2.data = torch.full((1,), 300)
        init_params['sigma'] = param2

    elif KERNEL is 'poly':
        param2 = torch.full((1,), 0.1)
        init_params['p'] = param2

    Metric = TrainableMetric(init_params, features.shape[0], lossC, lagrangian=True, kernel=KERNEL)

    optimizers = torch.optim.ASGD(Metric.get_params(0.001), lr=0.002)
    recorder = io.Recorder('loss', 'test_mAp', 'train_mAp')
    param_recorder = io.ParameterSaver(*Metric.parameters.keys())

    for it in range(NUM_ITER):

        param_recorder.update(FILENAME + '_param_', **Metric.parameters)
        m_loss = 0
        m_score = 0
        for _ in range(SKIP_STEP):

            training_features, training_labels, temp_indices = create_batch(features, ground_truth, train_ind, BATCH_SIZE)

            loss, train_distances = Metric.objective_function(training_features, training_labels)
            m_loss += loss.clone().detach().cpu().numpy()/SKIP_STEP

            _, score = eval.evaluate(RANK, train_distances, temp_indices, temp_indices)
            m_score += score/SKIP_STEP
            torch.cuda.empty_cache()
            loss.backward()

        if not it == 0: # Don't optimise on first iteration
            optimizers.step()
            optimizers.zero_grad()
            print('Optimized')

        with torch.no_grad():

            test_distances = Metric(features[:, query_ind], features[:, gallery_ind])

        ranked_winners, test_score = eval.evaluate(RANK, test_distances, query_ind, gallery_ind)
        recorder.update(FILENAME, loss=m_loss, test_mAp=test_score, train_mAp=m_score)

        print(m_loss)

        print(m_score, test_score)