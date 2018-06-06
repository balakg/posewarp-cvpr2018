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
import h5py
import util
import truncated_vgg
from keras.models import load_model,Model
from keras.backend.tensorflow_backend import set_session


def train(dataset,gpu_id):	

	params = param.getGeneralParams()
	gpu = '/gpu:' + str(gpu_id)

	np.random.seed(17)
	feed = datageneration.create_feed(params, 'test_vids.txt', 5000, False, False, False, True)
	
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.allow_soft_placement = True
	set_session(tf.Session(config=config))

	with tf.device(gpu):
		vgg_model = truncated_vgg.vgg_norm()
		networks.make_trainable(vgg_model,False)
		response_weights = sio.loadmat('mean_response.mat')
		fgbg = networks.network_fgbg(params,vgg_model,response_weights)
		fgbg.load_weights('../results/networks/fgbg_vgg/184000.h5')
		#disc = networks.discriminator(params)
		#gan = networks.gan(fgbg,disc,params,vgg_model,response_weights,0.01,1e-4)
		#gan.load_weights('../results/networks/fgbg_gan/7000.h5')
	
	n_batches = 200
	for j in xrange(n_batches):	
		print j
		X,Y = next(feed)		
		pred = fgbg.predict(X)	
		sio.savemat('results/transfer_vgg/' + str(j) + '.mat',{'X': X[0],'Y': Y, 'pred': pred})


if __name__ == "__main__":
	train('golfswinghd',sys.argv[1])
