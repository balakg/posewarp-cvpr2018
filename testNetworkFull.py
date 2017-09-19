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

def train(dataset,gpu_id):	

	params = param.getGeneralParams()
	gpu = '/gpu:' + str(gpu_id)

	golf_params = param.getDatasetParams('golfswinghd')
	workout_params = param.getDatasetParams('workout')
	tennis_params = param.getDatasetParams('tennis')
	aux_params = param.getDatasetParams('test-aux')

	_,golf_test = datareader.makeWarpExampleList(golf_params,0,2200,False,2)
	_,workout_test = datareader.makeWarpExampleList(workout_params,0,1200,False,3)
	_,tennis_test = datareader.makeWarpExampleList(tennis_params,0,1000,False,4)
	_,aux_test = datareader.makeWarpExampleList(aux_params,0,1000,False,5)

	test = workout_test + tennis_test + aux_test
	feed = datageneration.warpExampleGenerator(test,params,do_augment=False,return_pose_vectors=True)
	

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
			fgbg = networks.network_fgbg(params,vgg_model,response_weights)
			fgbg.load_weights('../results/networks/fgbg_vgg/140000.h5')
			#disc = networks.discriminator(params)
			#gan = networks.gan(fgbg,disc,params,vgg_model,response_weights,0.01,1e-4)
			#gan.load_weights('../results/networks/fgbg_gan2/2000.h5')

			outputs = [fgbg.outputs[0]]
			outputs.append(fgbg.get_layer('mask_src').output)
			outputs.append(fgbg.get_layer('fg_stack').output)
			outputs.append(fgbg.get_layer('bg_src').output)
			outputs.append(fgbg.get_layer('bg_tgt').output)
			outputs.append(fgbg.get_layer('fg_tgt').output)
			outputs.append(fgbg.get_layer('fg_mask_tgt').output)
			model = Model(fgbg.inputs, outputs)

		model.summary()
	
		np.random.seed(17)
		n_batches = 20
		for j in xrange(n_batches):	
			print j
			X,Y = next(feed)		
			pred = model.predict(X[:-2])
	
			sio.savemat('results/fgbg140/' + str(j) + '.mat',{'X': X[0],'Y': Y, 'pred': pred[0], 'mask_src': pred[1],
						'fg_stack': pred[2], 'bg_src': pred[3], 'bg_tgt': pred[4], 'fg_tgt': pred[5], 'fg_mask_tgt': pred[6], 
						'prior': X[3], 'pose_src': X[-2], 'pose_tgt': X[-1]})	


if __name__ == "__main__":
	train('golfswinghd',sys.argv[1])
