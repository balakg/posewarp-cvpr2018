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
import myVGG
from keras.models import load_model,Model
from keras.backend.tensorflow_backend import set_session


def train(dataset,gpu_id):	

	params = param.getGeneralParams()
	gpu = '/gpu:' + str(gpu_id)

	np.random.seed(17)
	feed = datageneration.createFeed(params,'test_vids.txt',False,True)
	
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.allow_soft_placement = True
	set_session(tf.Session(config=config))

	with tf.device(gpu):
		vgg_model = myVGG.vgg_norm()
		networks.make_trainable(vgg_model,False)
		response_weights = sio.loadmat('mean_response.mat')
		fgbg = networks.network_fgbg(params)
		fgbg.load_weights('../results/networks/fgbg_vgg/20000.h5')
		#disc = networks.discriminator(params)
		#gan = networks.gan(fgbg,disc,params,vgg_model,response_weights,0.01,1e-4)
		#gan.load_weights('../results/networks/fgbg_gan/7000.h5')

		outputs = [fgbg.outputs[0]]
		outputs.append(fgbg.get_layer('mask_src').output)
		outputs.append(fgbg.get_layer('fg_stack').output)
		outputs.append(fgbg.get_layer('bg_src').output)
		outputs.append(fgbg.get_layer('bg_tgt').output)
		outputs.append(fgbg.get_layer('fg_tgt').output)
		outputs.append(fgbg.get_layer('fg_mask_tgt').output)
		model = Model(fgbg.inputs, outputs)
	
	n_batches = 10
	for j in xrange(n_batches):	
		print j
		X,Y = next(feed)		
		pred = model.predict(X[:-2])
	
		sio.savemat('results/fgbg_vgg/' + str(j) + '.mat',{'X': X[0],'Y': Y, 'pred': pred[0], 'mask_src': pred[1],
					'fg_stack': pred[2], 'bg_src': pred[3], 'bg_tgt': pred[4], 'fg_tgt': pred[5], 'fg_mask_tgt': pred[6], 
					'prior': X[3], 'pose_src': X[-2], 'pose_tgt': X[-1]})	


if __name__ == "__main__":
	train('golfswinghd',sys.argv[1])
