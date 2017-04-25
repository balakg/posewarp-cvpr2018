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
import h5py

def train(dataset,gpu_id):	

	params = param.getParam(dataset)
	gpu = '/gpu:' + str(gpu_id)

	ex_train,ex_test = datareader.makeTransferExampleList(params)
	test_feed = datageneration.transferExampleGenerator(ex_test,params)
	
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.allow_soft_placement = True
	
	f = h5py.File('../results/networks/gan100/20000.h5', 'r+')
	del f['optimizer_weights']
	f.close()

	with tf.Session(config=config) as sess:

		sess.run(tf.global_variables_initializer())
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)

		with tf.device(gpu):
			model_warp = load_model('../results/networks/warp_spotlight_tanh/40000.h5')
			model_mask = Model(model_warp.input, model_warp.get_layer('mask').output)
			model_gan = load_model('../results/networks/gan100/20000.h5')
		
			n_batches = 8
			for j in xrange(n_batches):	
				print j
				X_src,X_tgt,X_pose,X_mask = next(test_feed)			
				I_warp = model_warp.predict([X_src,X_pose,X_mask])
				I_gan = model_gan.predict([X_src,X_pose,X_mask])
				I_mask = model_mask.predict([X_src,X_pose,X_mask])	

				sio.savemat('tests/' + str(j) + '.mat',
         		{'X_src': X_src,'X_tgt': X_tgt, 'mask0': X_mask, 'mask1': I_mask, 
					'I_warp': I_warp, 'I_gan': I_gan[0]})	

if __name__ == "__main__":
	train('golfswinghd',sys.argv[1])
