import numpy as np
import json
from scipy.io import loadmat
import cv2
import os.path

NUMBER_PEOPLE = 1360
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
                images.append(cv2.imread('PR_data/images_cuhk03/' + filename))
            g_t.append(all_g_t[index])

        return images, g_t, cam_id
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
                    this_image.append(cv2.imread('PR_data/images_cuhk03/' + filename))
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

    return train_idxs

def get_validation_indexes(number=100):

    _, g_t, _, ix = get_im_info(phase='validation', return_im=False)
    return np.array(ix).flatten(), g_t



#   Get the index of the images/features for testing
def get_query_indexes():
    query_idxs, g_t = loadmat('PR_data/cuhk03_new_protocol_config_labeled.mat', variable_names=['query_idx'])[
        'query_idx'].flatten()

    return query_idxs, g_t


def get_gallery_indexes():
    gal_idxs, g_t = loadmat('PR_data/cuhk03_new_protocol_config_labeled.mat', variable_names=['gallery_idx'])[
        'gallery_idx'].flatten()

    return gal_idxs, g_t


def get_ground_truth(indexes):

    all_g_t = loadmat('PR_data/cuhk03_new_protocol_config_labeled.mat', variable_names=['labels'])['labels'].flatten()
    return all_g_t[indexes]


def interpret_rank(ranked_results, gallery_indexes, query_indexes):
    # indexes have to be indexes for the whole dataset, not the specific partition
    g_images, g_g_t, g_cam_id, _ = get_im_info(index=gallery_indexes)
    q_images, q_g_t, q_cam_id, _ = get_im_info(index=query_indexes)
    # true_ranked_results = gallery_indexes[ranked_results]
    labelled_ranked_results = g_g_t[ranked_results]
    positive = labelled_ranked_results == q_g_t[:, None]


    return positive

