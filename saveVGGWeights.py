import tensorflow as tf
import os
import numpy as np
import sys
import datareader
import datageneration
import scipy.io as sio
import param
import util
from keras.models import load_model,Model
import myVGG
from keras.backend.tensorflow_backend import set_session

def main(gpu_id):	

	params = param.getGeneralParams()
	gpu = '/gpu:' + str(gpu_id)

	n_steps = 2000
	train = datareader.makeWarpExampleList('train_vids.txt',n_steps)
	train_feed = datageneration.warpExampleGenerator(train,params,return_pose_vectors=False)
	
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.allow_soft_placement = True
	set_session(tf.Session(config=config))
	
	with tf.device(gpu):
		vgg_model = myVGG.vgg_norm()

	mean_response = []
	num_elements = []
	for step in xrange(n_steps):
		print step
		X,Y = next(train_feed)			
		pred_step = vgg_model.predict(util.vgg_preprocess(X[0]))
		for i in xrange(len(pred_step)):
			sum_i = np.sum(pred_step[i],axis=(0,1,2))
			n_elt = np.prod(pred_step[i].shape[0:3]) 
			if(step == 0):
				mean_response.append(sum_i)
				num_elements.append(n_elt)
			else:
				mean_response[i] += sum_i
				num_elements[i] += n_elt
		
	for i in xrange(len(mean_response)):
		mean_response[i]/= (1.0*num_elements[i])

	std_response = []
	for step in xrange(n_steps):
		print step
		X,Y = next(train_feed)			
		pred_step = vgg_model.predict(util.vgg_preprocess(X[0]))

		for i in xrange(len(pred_step)):
			mean_response_i = np.reshape(mean_response[i], (1,1,1,-1))						
			mean_response_i = np.tile(mean_response_i, (pred_step[i].shape[0:3])+(1,))

			d = np.sum((pred_step[i] - mean_response_i)**2,axis=(0,1,2))	
			if(step == 0):
				std_response.append(d)
			else:
				std_response[i] += d

	for i in xrange(12):
		std_response[i] = np.sqrt(std_response[i]/(num_elements[i]-1.0))

	responses = {}
	for i in xrange(12):
		responses[str(i)] = (mean_response[i],std_response[i])

	sio.savemat('mean_response_new.mat', responses)


if __name__ == "__main__":
	main(sys.argv[1])
