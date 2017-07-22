
#Compute the probability of each pixel in a tensor to be from background or foreground
import os
import pickle
import numpy as np
from PIL import Image, ImageDraw

import theano
import theano.tensor as T
from theano.tensor.nnet.conv import conv2d
from theano.tensor.shared_randomstreams import RandomStreams

class BGsubstract(object):

	#Input :
	#	- x : input tensor, should be activation volume coming out of AlexNet CNN
	#	- y : desired segmentation if we want to do training
	def __init__(self, x_activ, pretrained = False):


		#predefined number of activation (should be same as x.shape[1])
		nb_activations = 4227

		#Reshape our tensors to 2D matrix with samples in first dim and channels activation in second dim
		x_activ_flat = x_activ.dimshuffle(0,2,3,1).reshape((x_activ.shape[0]*x_activ.shape[2]*x_activ.shape[3],x_activ.shape[1]))

		#will do a standart NN with one hidden layer, set size here
		nb_hiddens = 50
		nb_hiddens2 = 50


		# define model: neural network
		def floatX(x):
		    return np.asarray(x, dtype=theano.config.floatX)

		def init_weights(shape):
		    return theano.shared(floatX(np.random.randn(*shape) * 1e-3))

		def dropout(x, p=0.0):
			if p > 0:
				retain_prob = 1 - p
				#x has smaples in first dim and features in second dim, what we want is get only some features => binomial on second dim
				srng = RandomStreams()
				x *= srng.binomial(n=1, size=(x.shape[1],), p=retain_prob, dtype=theano.config.floatX).dimshuffle('x',0)
				x /= retain_prob
			return x



		def model(x, w_h, b_h, w_h2, b_h2, w_o, b_o, p=0.0):
		    #h = T.maximum(0, T.dot(x, w_h) + b_h)
		    h = T.nnet.sigmoid(T.dot(x, w_h) + b_h)
		    #h2 = T.maximum(0, T.dot(h, w_h2) + b_h2)
		    h2 = T.nnet.sigmoid(T.dot(h, w_h2) + b_h2)
		    h2_d = dropout(h2, p)
		    p_fb = T.nnet.softmax(T.dot(h2_d, w_o) + b_o)
		    return p_fb

		if pretrained == False :
			w_h = init_weights((nb_activations, nb_hiddens))
			b_h = init_weights((nb_hiddens,))
			w_h2 = init_weights((nb_hiddens, nb_hiddens2))
			b_h2 = init_weights((nb_hiddens2,))
			w_o = init_weights((nb_hiddens2, 2))
			b_o = init_weights((2,))
		else:
			if len(os.path.dirname(__file__)) >0:
				paramsValues = pickle.load(open(os.path.dirname(__file__)+"/models/paramsBG.pickle","rb"))
			else:
				paramsValues = pickle.load(open("./models/paramsBG.pickle","rb"))

			w_h = theano.shared(paramsValues[0])
			b_h = theano.shared(paramsValues[1])
			w_h2 = theano.shared(paramsValues[2])
			b_h2 = theano.shared(paramsValues[3])
			w_o = theano.shared(paramsValues[4])
			b_o = theano.shared(paramsValues[5])



		self.params = [w_h, b_h, w_h2, b_h2, w_o, b_o]

		#get foreground background prob flatten
		self.p_fb_flat_test = model(x_activ_flat, *self.params, p=0.0)
		self.p_fb_flat_train = model(x_activ_flat, *self.params, p=0.0)
		#get foreground background proba reshaped as tensor, that s all we want for inference
		self.p_fb = self.p_fb_flat_test.reshape((x_activ.shape[0],x_activ.shape[2],x_activ.shape[3],2)).dimshuffle(0,3,1,2)

		


	def getParams(self):
	    params_values = []
	    for p in range(len(self.params)):
	        params_values.append(self.params[p].get_value())
	    
	    return params_values

	def setParams(self, params_values):
	    for p in range(len(params_values)):
	        self.params[p].set_value(params_values[p])



