import tensorflow as tf
import os
import numpy as np
import sys
import cv2
import datareader
import datageneration
import paDataReader
import networks
import scipy.io as sio
import param
from keras.models import load_model,Model
from keras.optimizers import Adam
import h5py
import util
import truncated_vgg

def train(dataset,gpu_id):	

	params = param.getGeneralParams()
	gpu = '/gpu:' + str(gpu_id)

	lift_params = param.getDatasetParams('weightlifting')
	golf_params = param.getDatasetParams('golfswinghd')
	workout_params = param.getDatasetParams('workout')
	tennis_params = param.getDatasetParams('tennis')
	aux_params = param.getDatasetParams('test-aux')

	_,lift_test = datareader.makeWarpExampleList(lift_params,0,2000,2,1)
	_,golf_test = datareader.makeWarpExampleList(golf_params,0,5000,2,2)
	_,workout_test = datareader.makeWarpExampleList(workout_params,0,2000,2,3)
	_,tennis_test = datareader.makeWarpExampleList(tennis_params,0,2000,2,4)
	_,aux_test = datareader.makeWarpExampleList(aux_params,0,2000,2,5)

	test = lift_test + golf_test+workout_test + tennis_test + aux_test
	feed = datageneration.warp_example_generator(test, params, do_augment=False, draw_skeleton=False, skel_color=(0, 0, 255),
                                                 return_pose_vectors=True)
	

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


			gen = networks.network_posewarp(params, vgg_model, response_weights, True, loss='vgg')
			disc = networks.discriminator(params)
			gan = networks.gan(gen,disc,params,vgg_model,response_weights,0.01,1e-4)
			gan.load_weights('../results/networks/gan/10000.h5')
		
	
		np.random.seed(17)
		n_batches = 25
		for j in xrange(n_batches):	
			print j
			X,Y = next(feed)		
			loss = gen.evaluate(X[0:-2],Y)
			pred = gen.predict(X[0:-2])

			sio.savemat('results/outputs/' + str(j) + '.mat',{'X': X[0],'Y': Y,'pred':pred,'loss':loss,
							'src_pose': X[-2], 'tgt_pose': X[-1]})


if __name__ == "__main__":
	train('golfswinghd',sys.argv[1])
