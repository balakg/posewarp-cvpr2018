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

	golf_params = param.getDatasetParams('golfswinghd')

	golf_src,golf_pose = datareader.makeActionExampleList(golf_params,src_frame=0)
	
	feed = datageneration.actionExampleGenerator(golf_src,golf_pose,params)
	
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
			fgbg = networks.network_fgbg(params,vgg_model)
			fgbg.load_weights('../results/networks/fgbg/40000.h5')
	
			'''
			outputs = [fgbg.outputs[0]]
			outputs.append(fgbg.get_layer('mask_src').output)
			outputs.append(fgbg.get_layer('fg_stack').output)
			outputs.append(fgbg.get_layer('bg_src').output)
			outputs.append(fgbg.get_layer('bg_tgt').output)
			outputs.append(fgbg.get_layer('fg_tgt').output)
			outputs.append(fgbg.get_layer('fg_mask_tgt').output)
			'''

			#model = Model(fgbg.inputs, outputs)

		for j in xrange(len(golf_pose)):	
			print j
			X,Y = next(feed)			
			pred = fgbg.predict(X[0:-1])[0]

			sio.savemat('results/action2/' + str(j) + '.mat', {'pred': pred,'X':X[0], 'Y':Y, 'joints': X[-1]});

				

if __name__ == "__main__":
	train('golfswinghd',sys.argv[1])
