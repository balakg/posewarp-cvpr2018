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
from keras.optimizers import Adam
from keras.applications.vgg19 import VGG19
import h5py
import util

def train(dataset,gpu_id):	

	params = param.get_general_params()
	gpu = '/gpu:' + str(gpu_id)

	lift_params = param.getDatasetParams('weightlifting')
	golf_params = param.getDatasetParams('golfswinghd')
	yoga_params = param.getDatasetParams('yoga')

	lift_train,lift_test = datareader.makeWarpExampleList(lift_params,20000,5000,2,1)
	golf_train,golf_test = datareader.makeWarpExampleList(golf_params,20000,5000,2,2)
	yoga_train,yoga_test = datareader.makeWarpExampleList(yoga_params,20000,5000,2,3)

	test_all = lift_test + golf_test + yoga_test
	train_all = lift_train + golf_train + yoga_train

	test_feed = data_generation.warp_example_generator(test_all, params, drawSkeleton=False, doAugment=False)
	train_feed = data_generation.warp_example_generator(test_all, params, drawSkeleton=False, doAugment=False)
	
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.allow_soft_placement = True
	
	with tf.Session(config=config) as sess:

		sess.run(tf.global_variables_initializer())
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)

		with tf.device(gpu):
			vgg_model = VGG19(weights='imagenet',include_top=False,input_shape=(256,256,3))
			networks.make_trainable(vgg_model,False)
			generator1 = networks.network_warpclass(params,vgg_model)
			generator1.load_weights('../results/networks/all_l2_vgg/22500.h5')
		
		n_batches = 1000

		data = np.zeros((n_batches*4,57))
		
		for j in xrange(n_batches):	
			print j
			X,Y = next(feed)			
			pred = generator1.test_on_bath(X[0:6],Y)[1]
			
			data[X

			sio.savemat('results/skeloverlay_red/' + str(j) + '.mat',{'X': X[0],'pred': pred, 'Y': Y})	

if __name__ == "__main__":
	train('golfswinghd',sys.argv[1])
