import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.optimizers import Adam
import os
import scipy.io as sio
import numpy as np
import sys
sys.path.append('../')
import data_generation
import networks
import param
import truncated_vgg

def run(gpu_id):
    params = param.get_general_params()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    vgg_model = truncated_vgg.vgg_norm()
    networks.make_trainable(vgg_model, False)
    response_weights = sio.loadmat('../../data/vgg_activation_distribution_train.mat')
    model = networks.network_posewarp(params)
    model.compile(optimizer=Adam(), loss=[networks.vgg_loss(vgg_model, response_weights, 12)])
    iterations = range(1000, 185001, 1000)

    n_batches = 25
    losses = []
    for i in iterations:
        print(i)
        model.load_weights('../../models/posewarp_vgg2/' + str(i) + '.h5')

        np.random.seed(11)
        feed = data_generation.create_feed(params, params['data_dir'], 'test')

        loss = 0
        for batch in range(n_batches):
            x, y = next(feed)
            loss += model.evaluate(x, y, verbose=False)
        loss /= (n_batches*1.0)
        losses.append(loss)
        sio.savemat('losses_by_iter2.mat', {'losses': losses, 'iterations': iterations})


if __name__ == "__main__":
    run(sys.argv[1])
