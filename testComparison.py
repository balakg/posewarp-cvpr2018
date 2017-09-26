import tensorflow as tf
import os
import numpy as np
import sys
import datageneration
import networks
import scipy.io as sio
import param
from keras.models import load_model,Model
from keras.optimizers import Adam
import h5py
import util
import myVGG
import skimage
from skimage.measure import compare_ssim
from keras.backend.tensorflow_backend import set_session

def l1Error(pred,true):
	pred = (pred+1.0)/2.0
	true = (true+1.0)/2.0
	return np.mean(np.abs(pred-true))

def ssimError(pred,true):

	true = (true+1.0)/2.0
	true = true.astype('float32')
	pred = (pred+1.0)/2.0

	ssim = 0
	for i in xrange(pred.shape[0]):
		ssim += compare_ssim(pred[i,:,:,:],true[i,:,:,:],multichannel=True)

	return (ssim/(1.0*pred.shape[0]))

def vggError(pred_vgg,true_vgg,feat_weights):
	loss = 0
	for j in xrange(12):
		std = feat_weights[str(j)][1]+0.1
		std = np.expand_dims(np.expand_dims(np.expand_dims(std,0),0),0)
		d = true_vgg[j] - pred_vgg[j]
		loss += np.mean(np.abs(d/std))

	return loss/12.0


def train(dataset,gpu_id):	

	params = param.getGeneralParams()
	gpu = '/gpu:' + str(gpu_id)

	np.random.seed(17)
	feed = datageneration.createFeed(params,'test_vids.txt',5000,False,True,True)
	
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.allow_soft_placement = True
	set_session(tf.Session(config=config))
	
	with tf.device(gpu):
		vgg_model = myVGG.vgg_norm()
		networks.make_trainable(vgg_model,False)
		response_weights = sio.loadmat('mean_response.mat')

		fgbg_vgg = networks.network_fgbg(params,vgg_model,response_weights)
		fgbg_vgg.load_weights('../results/networks/fgbg_vgg/184000.h5')	

		gen = networks.network_fgbg(params,vgg_model,response_weights)
		disc = networks.discriminator(params)
		gan = networks.gan(gen,disc,params,vgg_model,response_weights,0.1,1e-4)
		gan.load_weights('../results/networks/fgbg_gan/7000.h5')
			
		fgbg_l1 = networks.network_fgbg(params,vgg_model,response_weights,loss='l1')
		fgbg_l1.load_weights('../results/networks/fgbg_l1/100000.h5')	
	
		#mask_model = Model(fgbg_vgg.inputs,fgbg_vgg.get_layer('fg_mask_tgt').output)
	
	n_examples = 500
	
	metrics = np.zeros((n_examples,9))
	poses = np.zeros((n_examples,28*2))
	classes = np.zeros(n_examples)

	for j in xrange(n_examples):	
		print j
		X,Y = next(feed)		

		pred_l1 = fgbg_l1.predict(X[:-3])
		pred_vgg = fgbg_vgg.predict(X[:-3])
		pred_gan = gen.predict(X[:-3])

		'''
		#mask = mask_model.predict(X[:-3])
		pred_l1_fg = pred_l1 * mask
		pred_vgg_fg = pred_vgg * mask
		pred_gan_fg = pred_gan * mask

		pred_l1_bg = pred_l1 * (1-mask)
		pred_vgg_bg = pred_vgg * (1-mask)
		pred_gan_bg = pred_gan * (1-mask)

		Y_fg = Y * mask
		Y_bg = Y * (1-mask)
		'''


		#,pred_l1_fg,pred_vgg_fg,pred_gan_fg,pred_l1_bg,pred_vgg_bg,pred_gan_bg]
		#,Y_fg,Y_fg,Y_fg,Y_bg,Y_bg,Y_bg]	
		preds = [pred_l1,pred_vgg,pred_gan] 
		targets = [Y,Y,Y] 
	
		metrics[j,0:3] = [l1Error(preds[i],targets[i]) for i in xrange(len(preds))]
		metrics[j,3:6] = [vggError(vgg_model.predict(util.vgg_preprocess(preds[i])),
									vgg_model.predict(util.vgg_preprocess(targets[i])),response_weights) for i in xrange(len(preds))]
		metrics[j,6:] = [ssimError(preds[i],targets[i]) for i in xrange(len(preds))]
		poses[j,0:28] = X[-3]
		poses[j,28:] = X[-2]
		classes[j] = int(X[-1])
		sio.savemat('results/comparison/' + str(j) + '.mat', {'X': X[0], 'Y': Y, 'pred_l1': pred_l1, 'pred_vgg': pred_vgg, 'pred_gan': pred_gan})
		sio.savemat('results/comparison_fgbg.mat',{'metrics': metrics, 'poses': poses, 'classes': classes})


if __name__ == "__main__":
	train('golfswinghd',sys.argv[1])
