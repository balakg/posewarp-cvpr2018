import tensorflow as tf
import os
import numpy as np
import sys
import cv2
import datareader
import datageneration
import networks
import scipy.io as sio
import param
from keras.models import load_model,Model
import h5py
import util
import myVGG
import random

def train(dataset,gpu_id):	

	params = param.getGeneralParams()
	gpu = '/gpu:' + str(gpu_id)

	lift_params = param.getDatasetParams('weightlifting')
	golf_params = param.getDatasetParams('golfswinghd')
	workout_params = param.getDatasetParams('workout')
	tennis_params = param.getDatasetParams('tennis')
	aux_params = param.getDatasetParams('test-aux')

	_,golf_test = datareader.makeWarpExampleList(golf_params,0,5000,2,2)
	_,workout_test = datareader.makeWarpExampleList(workout_params,0,2000,2,3)
	_,tennis_test = datareader.makeWarpExampleList(tennis_params,0,2000,2,4)
	_,aux_test = datareader.makeWarpExampleList(aux_params,0,2000,2,5)

	test = golf_test + workout_test + tennis_test + aux_test

	feed = datageneration.transferExampleGenerator(test,test,params)	

	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.allow_soft_placement = True
	
	with tf.Session(config=config) as sess:

		sess.run(tf.global_variables_initializer())
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)

		with tf.device(gpu):
			vgg_model = myVGG.vgg_norm()
			networks.make_trainable(vgg_model,False)
			response_weights = sio.loadmat('mean_response.mat')


			fgbg = networks.network_fgbg(params,vgg_model,response_weights,loss='vgg')
			#fgbg.load_weights('../results/networks/fgbg/170000.h5')
			disc = networks.discriminator(params)
			gan = networks.gan(fgbg,disc,params,vgg_model,response_weights,0.01,1e-4)
			gan.load_weights('../results/networks/wgan/1000.h5')
		
	
		np.random.seed(17)
		n_batches = 25
		for j in xrange(n_batches):	
			print j
			X,Y = next(feed)		
			pred = fgbg.predict(X)

			sio.savemat('results/transfer3/' + str(j) + '.mat',{'X': X[0],'Y': Y, 'pred':pred}) 

if __name__ == "__main__":
	train('golfswinghd',sys.argv[1])
