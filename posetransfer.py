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

def train(dataset,model_name,gpu_id):	

	params = param.getParam(dataset)
	gpu = '/gpu:' + str(gpu_id)

	output_dir = '/afs/csail.mit.edu/u/b/balakg/pose/pose2image/results/outputs/' + model_name
	network_dir = '/afs/csail.mit.edu/u/b/balakg/pose/pose2image/results/networks/' + model_name

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
			model = networks.network_warp(params)
			#model = load_model('../results/networks/warp3/30000.h5')
			model.compile(optimizer=Adam(lr=1e-4), loss='mse')
		
		#X_src,X_tgt,X_pose,X_mask = next(train_feed)			
		#train_loss = model.train_on_batch([X_src,X_pose,X_mask],X_tgt)
		#pred = model.predict([X_src,X_pose,X_warp])
		#sio.savemat('test.mat', {'X_src': X_src, 'X_tgt': X_tgt, 'X_pose': X_pose, 'X_mask': X_mask})
		#return

		step = 0	
		while(True):

			X_src,X_tgt,X_pose,X_mask = next(train_feed)			

			with tf.device(gpu):
				train_loss = model.train_on_batch([X_src,X_pose,X_mask],[X_tgt])

			print str(step) + ",0," + str(train_loss)
			sys.stdout.flush()	

			if(step % params['test_interval'] == 0):
				n_batches = 8
				test_loss = 0
				for j in xrange(n_batches):	
					X_src,X_tgt,X_pose,X_mask = next(test_feed)			
					test_loss += model.test_on_batch([X_src,X_pose,X_mask], [X_tgt])

				test_loss /= (n_batches)
				print str(step) + ",1," + str(test_loss)
				sys.stdout.flush()

			if(step % params['test_save_interval']==0):
				X_src,X_tgt,X_pose,X_mask = next(test_feed)			
				pred_val = model.predict([X_src,X_pose,X_mask])
		
				sio.savemat(output_dir + '/' + str(step) + '.mat',
         		{'X_src': X_src,'X_tgt': X_tgt, 'X_mask': X_mask, 'pred': pred_val})	

				
			if(step % params['model_save_interval']==0):
				model.save(network_dir + '/' + str(step) + '.h5')			
			
			step += 1	

if __name__ == "__main__":
	if(len(sys.argv) != 3):
		print "Need model name and gpu id as command line arguments."
	else:
		train('golfswinghd',sys.argv[1],sys.argv[2])
