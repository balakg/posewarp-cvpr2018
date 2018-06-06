import tensorflow as tf
import os
import numpy as np
import sys
import datareader
import datageneration
import networks
import scipy.io as sio
import param
from keras.models import load_model,Model
import util
import myVGG
from keras.backend.tensorflow_backend import set_session

def convert(I):
	I = (I + 1.0)/2.0
	J = np.stack([I[:,:,2],I[:,:,1],I[:,:,0]],2)
	return J

def train(dataset,gpu_id):	

	params = param.getGeneralParams()
	gpu = '/gpu:' + str(gpu_id)
	
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.allow_soft_placement = True
	set_session(tf.Session(config=config))

	with tf.device(gpu):
		vgg_model = myVGG.vgg_norm()
		networks.make_trainable(vgg_model,False)
		response_weights = sio.loadmat('mean_response.mat')
		fgbg = networks.network_fgbg(params,vgg_model,response_weights)
		#fgbg.load_weights('../results/networks/fgbg_vgg/140000.h5')
		disc = networks.discriminator(params)
		gan = networks.gan(fgbg,disc,params,vgg_model,response_weights,0.01,1e-4)
		gan.load_weights('../results/networks/fgbg_gan/2000.h5')

		outputs = [fgbg.outputs[0]]
		#outputs.append(fgbg.get_layer('mask_src').output)
		#outputs.append(fgbg.get_layer('fg_stack').output)
		#outputs.append(fgbg.get_layer('bg_src').output)
		#outputs.append(fgbg.get_layer('bg_tgt').output)
		#outputs.append(fgbg.get_layer('fg_tgt').output)
		outputs.append(fgbg.get_layer('fg_mask_tgt').output)
		model = Model(fgbg.inputs, outputs)


	test = datareader.makeActionExampleList('test_vids.txt',1)
	feed = datageneration.warp_example_generator(test, params, do_augment=False, return_pose_vectors=True)

	
	n_frames = len(test)

	true_action = np.zeros((256,256,3,n_frames))
	pred_action = np.zeros((256,256,3,n_frames))
	mask = np.zeros((256,256,1,n_frames))

	for i in xrange(n_frames):
		print i
		X,Y = next(feed)		
		pred = model.predict(X[:-2])
		true_action[:,:,:,i] = convert(np.reshape(Y,(256,256,3)))		
		pred_action[:,:,:,i] = convert(np.reshape(pred[0],(256,256,3)))
		mask[:,:,:,i] = pred[1]		

	sio.savemat('results/action/1_gan.mat',{'true': true_action, 'pred': pred_action, 'mask': mask})


if __name__ == "__main__":
	train('golfswinghd',sys.argv[1])
