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
from keras.applications.vgg19 import VGG19
import h5py
import util

def train(dataset,gpu_id):	

	params = param.getGeneralParams()
	gpu = '/gpu:' + str(gpu_id)

	lift_params = param.getDatasetParams('weightlifting')
	golf_params = param.getDatasetParams('golfswinghd')
	yoga_params = param.getDatasetParams('yoga')

	_,lift_test = datareader.makeWarpExampleList(lift_params,0,5000,2,1)
	_,golf_test = datareader.makeWarpExampleList(golf_params,0,5000,2,2)
	_,yoga_test = datareader.makeWarpExampleList(yoga_params,0,5000,2,3)

	test = lift_test + golf_test + yoga_test
	feed = datageneration.warpExampleGenerator(test,params,do_augment=False,draw_skeleton=False,skel_color=(0,0,255))
	
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
			full = networks.network_warp(params,vgg_model)
			full.load_weights('../results/networks/l2+vgg/20500.h5')
			#warp_model = Model(full.inputs,full.get_layer('warped_stack').output)
			
			#l2 = networks.network_warp(params)
			#l2.load_weights('../results/networks/l2/23500.h5')

			#unet = networks.network_ed(params,vgg_model)
			#unet.load_weights('../results/networks/ed/45000.h5')			


		n_batches = 50
		for j in xrange(n_batches):	
			print j
			X,Y = next(feed)			
			#pred = l2.predict(X[0:-1])
			#sio.savemat('results/l2/' + str(j) + '.mat',{'X': X[0],'pred': pred, 'Y': Y})	

			pred = full.predict(X[0:-2])[0]
			#warp = warp_model.predict(X[0:-1])

			sio.savemat('results/poses/' + str(j) + '.mat',{'X': X[0],'pred': pred, 'Y': Y, 
				'pose': X[-2],'class': X[-1]})	

			#pred = unet.predict(X[0:2])[0]
			#sio.savemat('results/unet/' + str(j) + '.mat',{'X': X[0],'pred': pred, 'Y': Y})	

if __name__ == "__main__":
	train('golfswinghd',sys.argv[1])
