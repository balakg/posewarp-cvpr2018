import tensorflow as tf
import os
import numpy as np
import sys
import datareader
import datageneration
import networks
import scipy.io as sio
import param
import util
import myVGG
from keras.models import load_model,Model
from keras.backend.tensorflow_backend import set_session

def train(model_name,gpu_id):	

	params = param.getGeneralParams()

	gpu = '/gpu:' + str(gpu_id)

	network_dir = params['project_dir'] + '/results/networks/' + model_name

	if not os.path.isdir(network_dir):
		os.mkdir(network_dir)

	train = datareader.makeWarpExampleList('train_vids.txt',50000)
	test = datareader.makeWarpExampleList('test_vids.txt',5000)

	train_feed = datageneration.warpExampleGenerator(train,params,return_pose_vectors=False)
	test_feed = datageneration.warpExampleGenerator(test,params,return_pose_vectors=False)
	
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.allow_soft_placement = True
	set_session(tf.Session(config=config))

	with tf.device(gpu):
		vgg_model = myVGG.vgg_norm()
		networks.make_trainable(vgg_model,False)
		response_weights = sio.loadmat('mean_response_new.mat')
		model = networks.network_fgbg(params,vgg_model,response_weights,loss='vgg')
		#model = networks.network_pix2pix(params,vgg_model,response_weights,loss='l1')
		#model.load_weights('../results/networks/fgbg_vgg_new/80000.h5')

	#model.summary()

	for step in xrange(0,200000):
		X,Y = next(train_feed)			
		
		with tf.device(gpu):
			train_loss = model.train_on_batch(X,Y)
		util.printProgress(step,0,train_loss)
	
		if(step % params['test_interval'] == 0):
			n_batches = 16
			test_loss = 0
			for j in xrange(n_batches):	
				X,Y = next(test_feed)			
				test_loss += np.array(model.test_on_batch(X,Y))
			
			test_loss /= (n_batches)
			util.printProgress(step,1,test_loss)

		if(step % params['model_save_interval']==0):
			model.save(network_dir + '/' + str(step) + '.h5')			

if __name__ == "__main__":
	if(len(sys.argv) != 3):
		print "Need model name and gpu id as command line arguments."
	else:
		train(sys.argv[1],sys.argv[2])
