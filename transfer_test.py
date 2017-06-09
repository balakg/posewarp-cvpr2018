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
import myVGG
import h5py
import util

def train(dataset,gpu_id):	

	params = param.getGeneralParams()
	gpu = '/gpu:' + str(gpu_id)

	lift_params = param.getDatasetParams('weightlifting')
	golf_params = param.getDatasetParams('golfswinghd')
	yoga_params = param.getDatasetParams('yoga')
	famous_params = param.getDatasetParams('famous')

	_,lift_test = datareader.makeWarpExampleList(lift_params,0,5000,2,1)
	_,golf_test = datareader.makeWarpExampleList(golf_params,0,5000,2,2)
	_,yoga_test = datareader.makeWarpExampleList(yoga_params,0,5000,2,3)
	#_,famous_test = datareader.makeWarpExampleList(famous_params,0,1000,2,4)

	test = lift_test + golf_test + yoga_test
	
	feed = datageneration.transferExampleGenerator(test,test,params,do_augment=False)
	
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
			response_weights = sio.loadmat('mean_response4.mat')
			fgbg = networks.network_fgbg(params,vgg_model,response_weights)
			fgbg.load_weights('../results/networks/fgbg_vgg_4/14000.h5')

			'''
			outputs = [fgbg.outputs[0]]
			outputs.append(fgbg.get_layer('mask_src').output)
			outputs.append(fgbg.get_layer('fg_stack').output)
			outputs.append(fgbg.get_layer('bg_src').output)
			outputs.append(fgbg.get_layer('bg_tgt').output)
			outputs.append(fgbg.get_layer('fg_tgt').output)
			outputs.append(fgbg.get_layer('fg_mask_tgt').output)
			model = Model(fgbg.inputs, outputs)
			'''

			#warp_model = Model(inputs=generator.inputs,outputs=generator.get_layer('warped_stack').output)

		np.random.seed(17)	
		n_batches = 10
		for j in xrange(n_batches):	
			print j
			X,Y = next(feed)			
			pred = fgbg.predict(X)
			sio.savemat('results/transfer/' + str(j) + '.mat', {'X': X[0], 'Y': Y, 'pred': pred})
			#warp = warp_model.predict(X)
			#sio.savemat('results/transfer/' + str(j) + '.mat',{'X': X[0],'Y': Y, 'pred': pred[0], 'mask_src': pred[1],
			#			'fg_stack': pred[2], 'bg_src': pred[3], 'bg_tgt': pred[4], 'fg_tgt': pred[5], 'fg_mask_tgt': pred[6], 'prior': X[2][:,:,:,0]})	

if __name__ == "__main__":
	train('golfswinghd',sys.argv[1])
