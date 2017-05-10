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
from keras.layers import TimeDistributed

def train(dataset,model_name,gpu_id):	

	params = param.getParam(dataset)
	gpu = '/gpu:' + str(gpu_id)

	output_dir = params['project_dir'] + '/results/outputs/' + model_name
	network_dir = params['project_dir'] + '/results/networks/' + model_name

	if not os.path.isdir(output_dir):
		os.mkdir(output_dir)

	if not os.path.isdir(network_dir):
		os.mkdir(network_dir)

	ex_train,ex_test = datareader.makeWarpExampleList(params)

	train_feed = datageneration.warpExampleGenerator(ex_train,params)
	test_feed = datageneration.warpExampleGenerator(ex_test,params)
	
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.allow_soft_placement = True

	batch_size = params['batch_size']
	seq_len = params['seq_len']
	
	with tf.Session(config=config) as sess:

		sess.run(tf.global_variables_initializer())
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)

		with tf.device(gpu):
			single_net,rnn_net,vgg_model = networks.make_rnn_from_single(params,
							'../results/networks/L2+VGG_0.001/90000.h5')
			rnn_net.compile(optimizer=Adam(lr=(1e-4)),loss=['mse','mse'],loss_weights=[1.0,0.001])

		step = 0	
		while(True):
			X,Y = next(train_feed)			

			train_loss = np.zeros(3)
			for t in xrange(seq_len-1):
				Xt = []
				for j in xrange(len(X)):
					Xt.append(X[j][:,t,:,:,:])
	
				Yt = Y[:,t,:,:,:]

				with tf.device(gpu):
					out = single_net.predict(Xt)			
					Yt_vgg = vgg_model.predict(util.vgg_preprocess(Yt))

				for idx in xrange(len(out)):
					out[idx] = np.expand_dims(out[idx],1)
			
				Yt = np.expand_dims(Yt,1)
				Yt_vgg = np.expand_dims(Yt_vgg,1)			

				train_loss += np.array(rnn_net.train_on_batch(out,[Yt,Yt_vgg]))

			rnn_net.reset_states()

			util.printProgress(step,0,train_loss*1.0/(seq_len-1))

			if(step % params['test_interval'] == 0):
				n_batches = 8
				test_loss = np.zeros(3)
				for batch in xrange(n_batches):	
					X,Y = next(test_feed)			
					
					for t in xrange(seq_len-1):
						Xt = []
						for j in xrange(len(X)):
							Xt.append(X[j][:,t,:,:,:])
	
						Yt = Y[:,t,:,:,:]

						with tf.device(gpu):
							out = single_net.predict(Xt)			
							Yt_vgg = vgg_model.predict(util.vgg_preprocess(Yt))

						for idx in xrange(len(out)):
							out[idx] = np.expand_dims(out[idx],1)
			
						Yt = np.expand_dims(Yt,1)
						Yt_vgg = np.expand_dims(Yt_vgg,1)			

						test_loss += np.array(rnn_net.test_on_batch(out,[Yt,Yt_vgg]))

					rnn_net.reset_states()

				test_loss /= (n_batches*1.0*(seq_len-1))
				util.printProgress(step,1,test_loss)

			'''
			if(step % params['test_save_interval']==0):
				X,Y = next(test_feed)			
				pred = model.predict(X)[0]
	
				sio.savemat(output_dir + '/' + str(step) + '.mat',
         		{'X_src': X[0],'Y': Y, 'pred': pred})	
			'''	


			if(step % params['model_save_interval']==0):
				rnn_net.save(network_dir + '/' + str(step) + '.h5')			

			step += 1	

if __name__ == "__main__":
	if(len(sys.argv) != 3):
		print "Need model name and gpu id as command line arguments."
	else:
		train('golfswinghd',sys.argv[1],sys.argv[2])
