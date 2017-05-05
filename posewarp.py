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
import util
from keras.models import load_model,Model
from keras.optimizers import Adam
from keras.applications.vgg19 import VGG19

def train(dataset,model_name,gpu_id):	

	params = param.getParam(dataset)
	gpu = '/gpu:' + str(gpu_id)

	output_dir = params['project_dir'] + '/results/outputs/' + model_name
	network_dir = params['project_dir'] + '/results/networks/' + model_name

	if not os.path.isdir(output_dir):
		os.mkdir(output_dir)

	if not os.path.isdir(network_dir):
		os.mkdir(network_dir)

	ex_train,ex_test = datareader.makeTransferExampleList(params)

	train_feed = datageneration.transferExampleGenerator(ex_train,params)
	test_feed = datageneration.transferExampleGenerator(ex_test,params)
	
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.allow_soft_placement = True
	
	with tf.Session(config=config) as sess:

		sess.run(tf.global_variables_initializer())
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)

		with tf.device(gpu):
			vgg_model = VGG19(weights='imagenet',include_top=False,
						input_shape=(128,128,3))
			networks.make_trainable(vgg_model,False)
			model = networks.network_warp(params,vgg_model)
			model.compile(optimizer=Adam(lr=1e-4),loss=['mse','mse'],
						loss_weights=[1.0,0.001])

		step = 0	
		while(True):
			X_src,X_tgt,X_pose,X_mask,X_trans = next(train_feed)			

			with tf.device(gpu):
				X_feat = vgg_model.predict(util.vgg_preprocess(X_tgt))
				inputs = [X_src,X_pose,X_mask,X_trans]
				outputs = [X_tgt,X_feat]
				train_loss = model.train_on_batch(inputs,outputs)

			util.printProgress(step,0,train_loss)

			if(step % params['test_interval'] == 0):
				n_batches = 8
				test_loss = [0.0,0.0,0.0]
				for j in xrange(n_batches):	
					X_src,X_tgt,X_pose,X_mask,X_trans = next(test_feed)			
					X_feat = vgg_model.predict(util.vgg_preprocess(X_tgt))
					inputs = [X_src,X_pose,X_mask,X_trans]
					outputs = [X_tgt,X_feat]
					test_loss += np.array(model.test_on_batch(inputs,outputs))

				test_loss /= (n_batches)
				util.printProgress(step,1,test_loss)

			'''
			if(step % params['test_save_interval']==0):
				X_src,X_tgt,X_pose,X_mask,X_trans = next(test_feed)			
				inputs = [X_src,X_pose,X_mask,X_trans]
				pred = model.predict(inputs)
	
				sio.savemat(output_dir + '/' + str(step) + '.mat',
         		{'X_src': X_src,'X_tgt': X_tgt, 'X_mask': X_mask, 'pred': pred[0]})	
	
			if(step % params['model_save_interval']==0):
				model.save(network_dir + '/' + str(step) + '.h5')			
			'''		

			step += 1	

if __name__ == "__main__":
	if(len(sys.argv) != 3):
		print "Need model name and gpu id as command line arguments."
	else:
		train('golfswinghd',sys.argv[1],sys.argv[2])
