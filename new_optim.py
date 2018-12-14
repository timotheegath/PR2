import numpy as np
import data_in_out as io
from scipy.io import loadmat
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


class TrainableMetric:

    def __init__(self, init_params, dims, loss, lagrangian=True, kernel=None, similarity=False):

        self.kernel = kernel
        self.lagrangian_bool = lagrangian
        self.lagrangian = torch.full((1, ), 1, requires_grad=lagrangian)
        self.parameters = {}

        if similarity is False:
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

        else:
            self.parameters['A'] = torch.empty((dims, dims), requires_grad=True)
            self.parameters['A'].data = init_params['A']
            if kernel is 'poly':
                print('Not implemented')
            elif kernel is 'RBF':
                print('Not implemented')
            else:

                self.distance = metrics.BilinearSimilarity
        self.loss = loss
        self.transgressors = None

    def __call__(self, features, features_compare=None):

        return self.distance(self.parameters, features, features_compare)

    def objective_function(self, features, labels, temp_indices):

        full_size = temp_indices.shape[0]
        batch_size = full_size // 6
        indices = range(0, full_size, batch_size)
        distances = torch.zeros((features.shape[1], features.shape[1]))
        training_features = features[:, temp_indices]
        training_labels = labels[temp_indices]
        # for i in indices:
        #     end = min(full_size, i + batch_size)
        #     distances[i:end, :] = self(training_features[:, i:end], training_features)
        temp_dist = self(training_features, training_features)
        to_ignore = torch.zeros((features.shape[1], features.shape[1])).cpu()
        for i in range(training_features.shape[1]):
            distances[temp_indices[i], temp_indices] = temp_dist[i]
            to_ignore[temp_indices[i], temp_indices] += 1

        loss = self.loss(distances, training_labels, self.lagrangian, self.transgressors, to_ignore)

        return loss, distances.clone().detach().cpu().numpy()

    def get_params(self, lagr_lr=0.00001):

        return [{'params': [self.parameters[k] for k in self.parameters.keys()]},
                {'params': self.lagrangian, 'lr': lagr_lr}]

    def initial_run(self, training_features, training_labels):
        training_features, training_labels = to_torch(training_features, training_labels)
        full_size = training_features.shape[1]
        batch_size = full_size // 6
        indices = range(0, full_size, batch_size)
        distances = torch.empty((training_features.shape[1], training_features.shape[1]))
        for i in indices:
            end = min(full_size, i + batch_size)
            distances[i:end, :] = self(training_features[:, i:end], training_features)
        self.transgressors = self.loss(distances, training_labels, 1, self.transgressors, 0)

        self.transgressors = to_torch(np.array(list(self.transgressors))).type(torch.LongTensor)
        self.lagrangian = torch.full((1, self.transgressors.shape[0]), 1, requires_grad=self.lagrangian_bool)





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


def lossA(distances, labels, l, *args):
    distances = distances - torch.diag(distances.diag())
    label_mask = labels.view(1, -1) == labels.view(-1, 1)
    Dw = torch.masked_select(distances, label_mask) - 1
    Db = torch.masked_select(distances, 1 - label_mask)

    objective = l*torch.sum(Dw) - torch.sum(torch.sqrt(Db))
    return objective


def lossC(distances, labels, l, transgressors, to_ignore):
    margin = 1
    if KERNEL is not None:
        margin = 0.1
    distances -= torch.diag(distances.diag())
    # distances = distances/torch.max(distances.view(-1))*2
    label_mask = labels.view(1, -1) == labels.view(-1, 1)
    # constraint = torch.zeros((1,))
    # chosen_rows = np.arange(0, distances.shape[0])
    # np.random.shuffle(chosen_rows)
    # chosen_rows = chosen_rows[:min(100, distances.shape[0])].tolist()
    if transgressors is not None:
        # for i in chosen_rows:
        #
        #     same_label_candidates = torch.masked_select(distances[i, :],  label_mask[i, :])
        #     different_label_candidates = torch.masked_select(distances[i, :],  1 - label_mask[i, :])
        #     try:
        #         pair = different_label_candidates[0] - same_label_candidates[0]
        #         constraint += pair - 1
        #     except IndexError:
        #
        #         pass
        # same_distances = torch.masked_select(distances, label_mask)
        #
        # loss = torch.sum(same_distances) - torch.abs(l) * constraint
        # return loss
        if torch.cuda.is_available():
            transgressors = transgressors.cuda()
        to_ignore = to_ignore.cuda()
        constraint = torch.sum((l * (distances[transgressors[:, 0], transgressors[:, 1]] - \
                     distances[transgressors[:, 0], transgressors[:, 2]] - margin*transgressors.shape[0])) *
                               (1 - to_ignore)[transgressors[:, 0], transgressors[:, 1]]*
                               (1 - to_ignore)[transgressors[:, 0], transgressors[:, 2]])

        distances = distances.cpu()
        label_mask = label_mask.cpu()
        same_distances = torch.masked_select(distances, (label_mask.cpu()>0) & (to_ignore==0))

        loss = torch.sum(same_distances) - constraint.cpu()
        return loss

    else:
        transgressors = set()

        for i in range(distances.shape[0]):

            same_label_candidates = torch.masked_select(distances[i, :i], label_mask[i, :i])
            same_label_indices = label_mask[i, :].nonzero().flatten()
            different_label_candidates = torch.masked_select(distances[i, :i], 1 - label_mask[i, :i])
            different_label_indices = (1 - label_mask[i, :]).nonzero().flatten()

            if not ((different_label_candidates.shape[0] == 0) or (same_label_candidates.shape[0] == 0)):
                try:

                    pair = different_label_candidates[:, None] - same_label_candidates[None, :]
                    mask = pair < margin
                    if mask.any():
                        where_true = mask.nonzero()
                        diff_ids = different_label_indices[where_true[:, 0]].cpu().numpy().flatten()
                        same_ids = same_label_indices[where_true[:, 1]].cpu().numpy().flatten()
                        # print(different_label_indices.shape)
                        # print(pair.shape)
                        for k in range(where_true.shape[0]):
                            update = tuple([i, diff_ids[k], same_ids[k]])
                            transgressors.add(update)

                except IndexError:

                    pass
            io.loading_bar(i, distances.shape[0])
        return transgressors


    # Transform maximisation into minimisation


def create_batch(features, g_t,  train_index, size):
    temp_index = np.arange(0, train_index.shape[0]).astype(np.int32)
    if RANDOMIZE:
        np.random.shuffle(temp_index)
    temp_index = temp_index[:size]
    train_ix = train_index[temp_index].astype(np.int32)
    training_features = to_torch(features[:, train_ix])

    training_labels = g_t[train_ix]

    return to_torch(training_features, training_labels, train_ix)

def initialise(mode):

    if mode is 'I':
        param = torch.eye(features.shape[0])
    elif mode is 'cov':
        param = torch.empty((features.shape[0], features.shape[0]), requires_grad=True)

        param.data = (
            torch.from_numpy(np.linalg.cholesky(np.linalg.inv(np.cov(features[:, train_ind]))).transpose()).type(
                Tensor))
    elif mode is 'restore':
        param = loadmat('Results/' + FILENAME + '_param_')['L'][0]

        param = to_torch(param)
    return param


if __name__ == '__main__':
    BATCHIFY = False
    RANDOMIZE = False
    SIM = False
    KERNEL = 'RBF'
    BATCH_SIZE = 2000
    RANK = 10
    SKIP_STEP = 1
    FILENAME = 'kernel_RBF_Maha_I'
    NUM_ITER = 1000
    INIT_MODES = ['cov', 'I', 'restore']

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
    if SIM == False:
        init_params['L'] = initialise(INIT_MODES[1])
        lr = 0.0000001
    else:
        init_params['A'] = initialise(INIT_MODES[1])
        lr = 0.0000001

    # For gaussian kernel
    if KERNEL is 'RBF':

        param2 = torch.full((1,), 30)
        init_params['sigma'] = param2
        lr = 0.01

    elif KERNEL is 'poly':

        param2 = torch.full((1,), 0.1)
        init_params['p'] = param2
        lr = 0.01

    Metric = TrainableMetric(init_params, features.shape[0], lossC, lagrangian=True, kernel=KERNEL, similarity=SIM)
    with torch.no_grad():
        Metric.initial_run(features[:, train_ind], ground_truth[train_ind])
    optimizers = torch.optim.ASGD(Metric.get_params(), lr=lr)
    recorder = io.Recorder('loss', 'test_mAp', 'train_mAp')
    param_recorder = io.ParameterSaver(*Metric.parameters.keys())

    for it in range(NUM_ITER):

        param_recorder.update(FILENAME + '_param_', **Metric.parameters)
        m_loss = 0
        m_score = 0
        for _ in range(SKIP_STEP):

            # training_features, training_labels, temp_indices = create_batch(features, ground_truth, train_ind, BATCH_SIZE)
            np.random.shuffle(train_ind)
            temp_indices = train_ind[:2000]
            loss, train_distances = Metric.objective_function(features, ground_truth, temp_indices)
            m_loss += loss.clone().detach().cpu().numpy()/SKIP_STEP

            _, score = eval.evaluate(RANK, train_distances, temp_indices, temp_indices)
            m_score += score/SKIP_STEP
            torch.cuda.empty_cache()
            loss.backward()

        if not it == 0: # Don't optimise on first iteration
            optimizers.step()
            optimizers.zero_grad()
            print('Optimized')
        # Test phase
        with torch.no_grad():

            test_distances = Metric(features[:, query_ind], features[:, gallery_ind])

        ranked_winners, test_score = eval.evaluate(RANK, test_distances, query_ind, gallery_ind)
        print('I/O operation, do not interrupt')
        recorder.update(FILENAME, loss=m_loss, test_mAp=test_score, train_mAp=m_score)
        print('Done')

        print(m_loss)

        print(m_score, test_score)
