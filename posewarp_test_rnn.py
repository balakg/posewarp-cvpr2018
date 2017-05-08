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

	params = param.getParam(dataset)
	gpu = '/gpu:' + str(gpu_id)

	ex_train,ex_test = datareader.makeWarpExampleList(params)
	test_feed = datageneration.warpExampleGenerator(ex_test,params)
	
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.allow_soft_placement = True
	
	with tf.Session(config=config) as sess:

		sess.run(tf.global_variables_initializer())
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)

		with tf.device(gpu):
			single_net,rnn_net,_ = networks.make_rnn_from_single(params,
								'../results/networks/L2+VGG_0.001/90000.h5', 
								'../results/networks/rnn_L2+VGG_0.001_90000/10000.h5')

		batch_size = params['batch_size']
		seq_len = params['seq_len']

		#src = np.zeros((batch_size,seq_len-1,128,128,3))
		#tgt = np.zeros((batch_size,seq_len-1,128,128,3))
		#pred = np.zeros((batch_size,seq_len-1,128,128,3))

		for t in xrange(seq_len-1):
			print t	
			X,Y = next(test_feed)			
			
			if(t == 0):
				I = X[0]

			with tf.device(gpu):
				out = single_net.predict([I,X[1],X[2],X[3]])			

			for idx in xrange(len(out)):
				out[idx] = np.expand_dims(out[idx],1)

			I = rnn_net.predict(out)[0]
			I = I[:,0,:,:,:]

			sio.savemat('rnn_output/' + str(t) + '.mat', {'X': X[0], 'Y': Y, 'pred': I})

		rnn_net.reset_states()


if __name__ == "__main__":
	train('golfswinghd',sys.argv[1])
