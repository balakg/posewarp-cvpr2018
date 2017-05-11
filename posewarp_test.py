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
from keras.optimizers import Adam
from keras.applications.vgg19 import VGG19
import h5py
import util

def train(dataset,gpu_id):	

	params = param.getGeneralParams()
	gpu = '/gpu:' + str(gpu_id)


	dataset1_params = param.getDatasetParams('weightlifting',1000,1000)
	dataset2_params = param.getDatasetParams('golfswinghd',1000,100)

	_,ex_test1 = datareader.makeWarpExampleList(dataset1_params)
	_,ex_test2 = datareader.makeWarpExampleList(dataset2_params)


	feed = datageneration.warpTransferExampleGenerator(ex_test2,ex_test1,params)

	#feed = datageneration.warpExampleGenerator(ex_test1,params)
	
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
			generator = networks.network_warp(params,vgg_model)
		
			#generator.summary()
			#discriminator = networks.discriminator(params)
			#gan = networks.gan(generator,discriminator,params)
			#gan.load_weights('../results/networks/gan/5000.h5')
			#discriminator.compile(loss='binary_crossentropy',optimizer=Adam(lr=1e-4))
			#generator.load_weights('../results/networks/L2+VGG_0.001/100000.h5')
			#generator.load_weights('../results/networks/lifting/50000.h5')
			generator.load_weights('../results/networks/golf+lifting/17500.h5')
			#mask = Model(inputs=generator.inputs,outputs=generator.get_layer('mask').output)
		
		#generator.summary()
		#X,Y = next(feed)

		#sio.savemat('0.mat',{'X': X[0], 'Y': Y, 'pose': X[1]}), 
		
		n_batches = 5
		for j in xrange(n_batches):	

			X = next(feed)			

			pred = generator.predict(X)[0]
			#mask = mask.predict(X)
			
			sio.savemat('results/golf2lifting_combined/' + str(j) + '.mat', 
					{'X': X[0], 'pose': X[1], 'pred': pred})


			#sio.savemat('test.mat',{'X': X[0],'Y': Y, 'mask': mask})	


if __name__ == "__main__":
	train('golfswinghd',sys.argv[1])
