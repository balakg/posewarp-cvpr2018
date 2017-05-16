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

	_,lift_test = datareader.makeWarpExampleList(lift_params,0,5000,2,1)
	_,golf_test = datareader.makeWarpExampleList(golf_params,0,5000,2,2)

	feed = datageneration.transferExampleGenerator(golf_test,lift_test,params)

	#test = pa_test + lift_test + golf_test
	#feed = datageneration.warpExampleGenerator(test,params)
	
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
			#generator2 = networks.network_warp(params,vgg_model)
			#generator3 = networks.network_warp(params,vgg_model)


			generator1.load_weights('../results/networks/golf+lifting_class/2500.h5')
			#generator2.load_weights('../results/networks/golf+lifting_feat/5000.h5')
			#generator3.load_weights('../results/networks/golf+lifting_mpii/5000.h5')			

			#generator.summary()
			#discriminator = networks.discriminator(params)
			#gan = networks.gan(generator,discriminator,params)
			#gan.load_weights('../results/networks/gan/5000.h5')
			#discriminator.compile(loss='binary_crossentropy',optimizer=Adam(lr=1e-4))
			#generator.load_weights('../results/networks/L2+VGG_0.001/100000.h5')
			#generator.load_weights('../results/networks/centered/15000.h5')
			#mask = Model(inputs=generator.inputs,outputs=generator.get_layer('mask').output)
		
		n_batches = 10
		for j in xrange(n_batches):	

			X,Y = next(feed)			

			pred = generator1.predict(X)[0]
			#pred_feat = generator2.predict(X)[0]
			#pred_mpii = generator3.predict(X)[0]

			#mask = mask.predict(X)

			print j
			sio.savemat('results/test' + str(j) + '.mat',{'X': X[0],'pred': pred, 'Y': Y})	
			'''
if __name__ == "__main__":
	train('golfswinghd',sys.argv[1])
