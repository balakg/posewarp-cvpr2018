import tensorflow as tf
import os
import numpy as np
import sys
import cv2
import datareader
import data_generation
import paDataReader
import networks
import scipy.io as sio
import param
from keras.models import load_model,Model
import h5py
import util
import truncated_vgg

def train(gpu_id):	

	params = param.get_general_params()
	gpu = '/gpu:' + str(gpu_id)

	test_params = param.getDatasetParams('test-aux')

	_,test = datareader.makeWarpExampleList(test_params,0,200,2,0)

	feed = data_generation.warp_example_generator(test, params, do_augment=False, draw_skeleton=False, skel_color=(0, 0, 255))
	

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.allow_soft_placement = True
	
	with tf.Session(config=config) as sess:

		sess.run(tf.global_variables_initializer())
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)

		with tf.device(gpu):
			vgg_model = truncated_vgg.vgg_norm()
			networks.make_trainable(vgg_model,False)
			response_weights = sio.loadmat('mean_response.mat')
			fgbg = networks.network_posewarp(params, vgg_model, response_weights, True)
	
		np.random.seed(17)


		n_batches = 50
		for i in xrange(146000,1452000,2000):
			fgbg.load_weights('../results/networks/fgbg_boundary/' + str(i) + '.h5')
			loss = 0
			for j in xrange(n_batches):	
				X,Y = next(feed)		
				loss += fgbg.test_on_batch(X,Y)
			
			loss /= n_batches
			print loss
			sys.stdout.flush()

if __name__ == "__main__":
	train(sys.argv[1])
