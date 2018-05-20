import tensorflow as tf
import os
import numpy as np
import sys
import datageneration
import networks
import scipy.io as sio
import param
import util
import myVGG
from keras.models import load_model,Model
from keras.backend.tensorflow_backend import set_session
from keras.optimizers import Adam,RMSprop
import time

def train(model_name,gpu_id):	

	params = param.getGeneralParams()

	gpu = '/gpu:' + str(gpu_id)

	network_dir = params['project_dir'] + '/results/networks/' + model_name

	if not os.path.isdir(network_dir):
		os.mkdir(network_dir)

	train_feed=datageneration.createFeed(params,"train_vids.txt")
	#test_feed=datageneration.createFeed(params,"test_vids.txt")


	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.allow_soft_placement = True
	set_session(tf.Session(config=config))

	with tf.device(gpu):
		vgg_model = myVGG.vgg_norm()
		networks.make_trainable(vgg_model,False)
		response_weights = sio.loadmat('mean_response.mat')
		model = networks.network_fgbg(params)
		#model.load_weights('../results/networks/fgbg_vgg/60000.h5')
		model.compile(optimizer=Adam(lr=1e-4),loss=[networks.vggLoss(vgg_model,response_weights)])

	#model.summary()

	for step in xrange(0,250000):
		start = time.time()
		
		#X,Y = next(test_feed)			
		#sio.savemat('data/data' + str(step) + '.mat',{'X':X[0],'Y':Y, 'ps': X[1], 'pt': X[2], 'mask': X[3]})
		#return

		X,Y = next(train_feed)			

		with tf.device(gpu):
			train_loss = model.train_on_batch(X,Y)

		end = time.time()

		util.printProgress(step,0,train_loss,end-start)

		'''
		if(step % params['test_interval'] == 0):
			n_batches = 8
			test_loss = 0
			for j in xrange(n_batches):	
				X,Y = next(test_feed)			
				test_loss += np.array(model.test_on_batch(X,Y))
			
			test_loss /= (n_batches)
			util.printProgress(step,1,test_loss,0)
		'''

		if(step > 0 and step % params['model_save_interval']==0):
			model.save(network_dir + '/' + str(step) + '.h5')			
	

if __name__ == "__main__":
	if(len(sys.argv) != 3):
		print "Need model name and gpu id as command line arguments."
	else:
		train(sys.argv[1],sys.argv[2])
