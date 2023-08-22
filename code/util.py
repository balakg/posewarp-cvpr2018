import sys
import numpy as np

def vgg_preprocess(x):
	x = 255.0 * (x + 1.0)/2.0

	x[:,:,:,0] -= 103.939
	x[:,:,:,1] -= 116.779
	x[:,:,:,2] -= 123.68

	return x

def printProgress(step,test,train_loss,time=None):
	s = str(step) + "," + str(test)

	if(isinstance(train_loss,list) or isinstance(train_loss,np.ndarray)):
		s += "," + ",".join([str(t) for t in train_loss])
	else:
		s += "," + str(train_loss)

	if(time is not None):
		s += "," + str(time)

	print(s)
	sys.stdout.flush()
