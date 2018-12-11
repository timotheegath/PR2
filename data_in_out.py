import numpy as np
import json
from scipy.io import loadmat, savemat
import cv2
import os.path
import torch
import collections

NUMBER_PEOPLE = 1360
IMAGE_SIZE = (293, 100)
class Recorder():

    def __init__(self, *args):
        self.memory = {}
        self.keys = []
        for a in args:
            self.keys.append(a)
            self.memory[a] = []

    def update(self, **kwargs):
        for key in kwargs:
            if isinstance(kwargs[key], collections.Iterable):
                tosave = []
                for p in kwargs[key]:
                    if isinstance(p, torch.Tensor):
                        tosave.append(kwargs[key].clone().detach().cpu().numpy())
                    else:
                        tosave.append(p)
            if isinstance(kwargs[key], torch.Tensor):
                tosave = kwargs[key].clone().detach().cpu().numpy()
            else:
                tosave = kwargs[key]
            self.memory[key].append(tosave)
            if key not in self.keys:
                print('Recorder: Key not recognised')


    def save(self, name):
        savemat('Results/' + name, self.memory)





def load_features():

    with open('PR_data/feature_data.json', 'r') as f:
        features = json.load(f)

    return np.array(features)


#   Function to easily interact with our image bank. Choose one out of three ways to get image(s): by filename, by index 
#   Or by label (returns all images of same label). The function returns the images, their ground_truth and their cam_id
#   The arguments can be single integers or a list. If opening by filename, specify if the image is part of query or
#   training. If images are not needed, return_im=False (saves time). how_many will be used when needing to get e.g 7
#   images per label. If all are wanted, leave to None
def get_im_info(filename=None, index=None, phase='training', labels=None, how_many=None, return_im=True):
    global IMAGE_SIZE
    IMAGE_SIZE_inv = (IMAGE_SIZE[1], IMAGE_SIZE[0])
    chosen_way = 0
    all_g_t = loadmat('PR_data/cuhk03_new_protocol_config_labeled.mat', variable_names=['labels'])['labels'].flatten()
    filenames = loadmat('PR_data/cuhk03_new_protocol_config_labeled.mat', variable_names=['filelist'])['filelist'].flatten()
    cam_ids = loadmat('PR_data/cuhk03_new_protocol_config_labeled.mat', variable_names=['camId'])['camId'].flatten()
    if phase == 'query':
        desired_idxs = loadmat('PR_data/cuhk03_new_protocol_config_labeled.mat', variable_names=['query_idx'])[
            'query_idx'].flatten() - 1
    else:
        desired_idxs = loadmat('PR_data/cuhk03_new_protocol_config_labeled.mat', variable_names=['train_idx'])[
            'train_idx'].flatten() - 1
    # Make all relevant arguments lists
    # And identify which way of indexing has been selected
    if filename is not None:
        if not isinstance(filename, list):
            filename = [filename]
        chosen_way = 1
    elif index is not None:
        if not isinstance(index, list):
            index = [index]
        chosen_way = 2
    elif labels is not None:
        if not isinstance(labels, list):
            desired_labels = [labels]
        desired_labels = labels
        chosen_way = 3
    elif phase == 'validation':
        chosen_way = 4

    if chosen_way is 0:
        return

    elif chosen_way is 1:
        images = []
        g_t = []
        cam_id = []
        for f in filename:
            idx = filenames.index(f)
            if return_im:
                images.append(cv2.imread(os.path.join('PR_data/images_cuhk03/', filename)))
            g_t.append(all_g_t[idx])
            cam_id.append(cam_ids[idx])
        return images, g_t, cam_id, []
    elif chosen_way is 2:
        images = []
        cam_id = []
        g_t = []

        for i in index:

            cam_id.append(cam_ids[i])
            filename = np.array2string(filenames[i])[2:-2]
            if return_im:
                im = cv2.imread('PR_data/images_cuhk03/' + filename)
                im = cv2.resize(im, dsize=IMAGE_SIZE_inv, interpolation=cv2.INTER_LINEAR)
                images.append(im)
            g_t.append(all_g_t[i])

        return images, g_t, cam_id, []
    elif chosen_way is 3:
        # These lists store all examples of all labels
        images = []
        g_t = []
        cam_id = []
        save_indexes = []
        # Only return images in the desired partition training or query
        all_g_t = all_g_t[desired_idxs]

        for l in desired_labels:
            idx = np.argwhere(all_g_t == l).reshape(-1)
            if how_many is not None:
                np.random.shuffle(idx)
                idx = idx[:how_many]
            idx = desired_idxs[idx]

            idx = idx.tolist()
            save_indexes.append(idx)

            # These lists store the info about all examples of the same label
            this_image = []
            this_g_t = []
            this_cam_id = []
            for i in idx:

                this_cam_id.append(cam_ids[i])
                filename = np.array2string(filenames[i])[2:-2]
                if return_im:
                    im = cv2.imread('PR_data/images_cuhk03/' + filename)
                    im = cv2.resize(im, dsize=IMAGE_SIZE_inv, interpolation=cv2.INTER_LINEAR)
                    this_image.append(im)
                this_g_t.append(l)
            images.append(this_image)
            g_t.append(this_g_t)
            cam_id.append(this_cam_id)
        # Warning: lists are now 2d lists
        return images, g_t, cam_id, save_indexes

    elif chosen_way is 4:

        available_labels = np.unique(all_g_t[desired_idxs])
        np.random.shuffle(available_labels)
        available_labels = available_labels[:100].tolist()

        images = []
        g_t = []
        cam_id = []
        save_indexes = []

        for l in available_labels:

            idx = np.argwhere(all_g_t == l).reshape(-1)

            idx = idx.tolist()
            save_indexes.append(idx)
            # These lists store the info about all examples of the same label
            this_image = []
            this_g_t = []
            this_cam_id = []
            for i in idx:

                this_cam_id.append(cam_ids[i])
                filename = np.array2string(filenames[i])[2:-2]
                if return_im:
                    this_image.append(cv2.imread('PR_data/images_cuhk03/' + filename))
                this_g_t.append(l)
            images.append(this_image)
            g_t.append(this_g_t)
            cam_id.append(this_cam_id)
            # Warning: lists are now 2d lists
        return images, g_t, cam_id, save_indexes






#   Get the index of the images/features to be used for training
def get_training_indexes():
    train_idxs = loadmat('PR_data/cuhk03_new_protocol_config_labeled.mat', variable_names=['train_idx'])[
        'train_idx'].flatten()


    return train_idxs - 1


def get_validation_indexes(number=100):

    _, g_t, _, ix = get_im_info(phase='validation', return_im=False)

    return np.array(ix).flatten() - 1, g_t




#   Get the index of the images/features for testing
def get_query_indexes():
    query_idxs= loadmat('PR_data/cuhk03_new_protocol_config_labeled.mat', variable_names=['query_idx'])['query_idx'].flatten()

    return query_idxs - 1


def get_gallery_indexes():
    gal_idxs= loadmat('PR_data/cuhk03_new_protocol_config_labeled.mat', variable_names=['gallery_idx'])[
        'gallery_idx'].flatten()

    return gal_idxs - 1


def get_cam_ids():
    cam_ids= loadmat('PR_data/cuhk03_new_protocol_config_labeled.mat', variable_names=['camId'])[
        'camId'].flatten()

    return cam_ids


def get_ground_truth():

    all_g_t = loadmat('PR_data/cuhk03_new_protocol_config_labeled.mat', variable_names=['labels'])['labels'].flatten()
    return all_g_t


def interpret_rank(ranked_results, gallery_indexes, query_indexes):
    # indexes have to be indexes for the whole dataset, not the specific partition
    g_images, g_g_t, g_cam_id, _ = get_im_info(index=gallery_indexes)
    q_images, q_g_t, q_cam_id, _ = get_im_info(index=query_indexes)
    # true_ranked_results = gallery_indexes[ranked_results]
    labelled_ranked_results = g_g_t[ranked_results]
    positive = labelled_ranked_results == q_g_t[:, None]


    return positive


def loading_bar(current, max):

    if current/max*100 % 10 == 0:
        print('Completion', int(current/max*100), '%')


# Override choice allows you to select the query you'd like to display instead of randomly selecting them
def display_ranklist(query_indexes, winner_indexes, rank, N, override_choice=None):

    # Parameters for later
    EDGE_THICKNESS = 10
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    GRAY = (50, 50, 50)
    WHITE = (255, 255, 255)

    def pad_frame(picture, color):

        interm = np.pad(picture,
                        pad_width=((EDGE_THICKNESS//2, EDGE_THICKNESS//2), (EDGE_THICKNESS//2, EDGE_THICKNESS//2), (0,0)),
                               mode='constant', constant_values=((color, color), (color, color), (0, 0)))

        final = np.pad(interm,
                        pad_width=((EDGE_THICKNESS//2, EDGE_THICKNESS//2), (EDGE_THICKNESS//2, EDGE_THICKNESS//2), (0,0)),
                               mode='constant', constant_values=((WHITE, WHITE), (WHITE, WHITE), (0, 0)))
        return final

    # Array that contains the total image
    display = np.zeros(((IMAGE_SIZE[0]+2*EDGE_THICKNESS)*N,
                        (IMAGE_SIZE[1]+2*EDGE_THICKNESS)*(rank+1), 3), dtype=np.uint8)

    if override_choice is None: # Find which query to display randomly
        choice = np.arange(0, query_indexes.shape[0])
        np.random.shuffle(choice)
        choice = choice[:N]
        chosen_query_ix = query_indexes[choice]
        chosen_winner_ix = winner_indexes[choice, :]
    else: # Use the user-specified indices
        chosen_query_ix = query_indexes[override_choice]
        chosen_winner_ix = winner_indexes[override_choice, :]
        N = override_choice.shape[0]
    # Get all query images and their ground_truth
    query_images, q_g_t, _ , _ = get_im_info(index=chosen_query_ix.tolist())
    # Goes through the rows
    for n in range(N):
        # Define where on the image you're working
        low_i = (EDGE_THICKNESS * 2 + IMAGE_SIZE[0]) * n
        high_i = (EDGE_THICKNESS * 2 + IMAGE_SIZE[0]) * (n + 1)
        low_j = 0
        high_j = (EDGE_THICKNESS * 2 + IMAGE_SIZE[1])

        # Add the query images on the left
        display[low_i:high_i, low_j:high_j] = pad_frame(query_images[n], GRAY)

        # Get all the winning images and their ground-truth
        winner_images, w_g_t, _, _ = get_im_info(index=chosen_winner_ix[n].tolist())
        # Get positive matches
        positives = (np.array(w_g_t) == np.array(q_g_t[n]))

        # Goes through the columns
        for i, im in enumerate(winner_images):

            low_j = (EDGE_THICKNESS * 2 + IMAGE_SIZE[1]) * (i + 1)
            high_j = (EDGE_THICKNESS * 2 + IMAGE_SIZE[1]) * (i + 2)

            if positives[i]:
                block = pad_frame(im, GREEN)
            else:
                block = pad_frame(im, RED)
            display[low_i:high_i, low_j:high_j] = block
    cv2.imshow('test', display)
    cv2.waitKey()
    cv2.imwrite('Results/rank_list.png', display)

def big_color_map():

    color_map = np.zeros((255*3, 3))
    for i in range(255*3):
        color_map[i] = [min(i, 255), max(i-255, 0), max(i-510, 0)]
    return color_map



