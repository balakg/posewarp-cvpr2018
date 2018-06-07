import tensorflow as tf
import numpy as np
import sys
import os
import data_generation
import scipy.io as sio
import param
import util
import truncated_vgg
from keras.backend.tensorflow_backend import set_session

"""
Saves mean and std. dev. for each channel of each layer of 
vgg network over training data.
"""


def main(gpu_id):
    params = param.get_general_params()

    train_feed = data_generation.create_feed(params, params['data_dir'])

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    vgg_model = truncated_vgg.vgg_norm()

    n_layers = len(vgg_model.outputs)
    n_batches = 2000

    # First, calculate mean activation of each channel
    mean_response = []
    num_elements = []
    for batch in range(n_batches):
        print batch
        x, y = next(train_feed)
        pred_batch = vgg_model.predict(util.vgg_preprocess(x[0]))

        for i in range(n_layers):
            sum_i = np.sum(pred_batch[i], axis=(0, 1, 2))
            n_elt = np.prod(pred_batch[i].shape[0:3])

            if batch == 0:
                mean_response.append(sum_i)
                num_elements.append(n_elt)
            else:
                mean_response[i] += sum_i
                num_elements[i] += n_elt

    for i in range(n_layers):
        mean_response[i] /= (1.0 * num_elements[i])

    # Now calculate std. dev. of each channel
    std_response = []
    for batch in range(n_batches):
        print batch
        x, y = next(train_feed)
        pred_batch = vgg_model.predict(util.vgg_preprocess(x[0]))

        for i in xrange(len(pred_batch)):
            mean_response_i = np.reshape(mean_response[i], (1, 1, 1, -1))
            mean_response_i = np.tile(mean_response_i, (pred_batch[i].shape[0:3]) + (1,))

            d = np.sum((pred_batch[i] - mean_response_i) ** 2, axis=(0, 1, 2))
            if batch == 0:
                std_response.append(d)
            else:
                std_response[i] += d

    for i in range(n_layers):
        std_response[i] = np.sqrt(std_response[i] / (num_elements[i] - 1.0))

    responses = {}
    for i in range(n_layers):
        responses[str(i)] = (mean_response[i], std_response[i])

    sio.savemat('vgg_train_statistics.mat', responses)


if __name__ == "__main__":
    main(sys.argv[1])
