
#Compute the density estimation of each pixel from a tensor without CNN
import os
import pickle
import numpy as np
from PIL import Image, ImageDraw

import theano
import theano.tensor as T
from theano.tensor.nnet.conv import conv2d
from theano.tensor.shared_randomstreams import RandomStreams

import NetFcts 


class VGG(object):

	#Input :
	#	- x : input tensor, should be activation volume coming out of AlexNet CNN
	def __init__(self, x):

		#predefined depth of input, here rgb image
		nb_activations_in = 3
		#for output
		self.nb_activations = 4227
		self.VGGout_resize = 16

		#apply first layers of VGG
		def modelVGG(x,w_c11,b_c11,w_c12,b_c12,  w_c21,b_c21,w_c22,b_c22,w_c31,b_c31,w_c32,b_c32,w_c33,b_c33,w_c41,b_c41,w_c42,b_c42,w_c43,b_c43, w_c51,b_c51,w_c52,b_c52,w_c53,b_c53):
            
			#first, we need to cut x to have a image size which is multiple of 16
			batch,in_channels,H,W = T.shape(x)
			#set image in BGR format
			xt = x[:,::-1,:,:]

			#remove mean
			xt = T.set_subtensor(xt[:,0,:,:], xt[:,0,:,:] - 103.939)
			xt = T.set_subtensor(xt[:,1,:,:], xt[:,1,:,:] - 116.779)
			xt = T.set_subtensor(xt[:,2,:,:], xt[:,2,:,:] - 123.68)
			xt = xt[:,:,0:self.VGGout_resize*(H/self.VGGout_resize),0:self.VGGout_resize*(W/self.VGGout_resize) ]
            
			c11 = T.maximum(0, NetFcts.convSameSize(xt, w_c11) + b_c11.dimshuffle('x', 0, 'x', 'x'))
			c12 = T.maximum(0, NetFcts.convSameSize(c11, w_c12) + b_c12.dimshuffle('x', 0, 'x', 'x'))
			#p1 = theano.tensor.signal.downsample.max_pool_2d(c12, ds = (2,2), st = (2,2), ignore_border=True)
			p1 = theano.tensor.signal.pool.pool_2d(c12, ds = (2,2), st = (2,2), ignore_border=True)

			c21 = T.maximum(0, NetFcts.convSameSize(p1, w_c21) + b_c21.dimshuffle('x', 0, 'x', 'x'))
			c22 = T.maximum(0, NetFcts.convSameSize(c21, w_c22) + b_c22.dimshuffle('x', 0, 'x', 'x'))
			p2 = theano.tensor.signal.pool.pool_2d(c22, ds = (2,2), st = (2,2), ignore_border=True)

			c31 = T.maximum(0, NetFcts.convSameSize(p2, w_c31) + b_c31.dimshuffle('x', 0, 'x', 'x'))
			c32 = T.maximum(0, NetFcts.convSameSize(c31, w_c32) + b_c32.dimshuffle('x', 0, 'x', 'x'))
			c33 = T.maximum(0, NetFcts.convSameSize(c32, w_c33) + b_c33.dimshuffle('x', 0, 'x', 'x'))
			p3 = theano.tensor.signal.pool.pool_2d(c33, ds = (2,2), st = (2,2), ignore_border=True)

			c41 = T.maximum(0, NetFcts.convSameSize(p3, w_c41) + b_c41.dimshuffle('x', 0, 'x', 'x'))
			c42 = T.maximum(0, NetFcts.convSameSize(c41, w_c42) + b_c42.dimshuffle('x', 0, 'x', 'x'))
			c43 = T.maximum(0, NetFcts.convSameSize(c42, w_c43) + b_c43.dimshuffle('x', 0, 'x', 'x'))
			p4 = theano.tensor.signal.pool.pool_2d(c43, ds = (2,2), st = (2,2), ignore_border=True)

			c51 = T.maximum(0, NetFcts.convSameSize(p4, w_c51) + b_c51.dimshuffle('x', 0, 'x', 'x'))
			c52 = T.maximum(0, NetFcts.convSameSize(c51, w_c52) + b_c52.dimshuffle('x', 0, 'x', 'x'))
			c53 = T.maximum(0, NetFcts.convSameSize(c52, w_c53) + b_c53.dimshuffle('x', 0, 'x', 'x'))
			#p5 = theano.tensor.signal.downsample.max_pool_2d(c53, ds = (2,2), st = (2,2), ignore_border=True)
            
            
            #We want 1/4 of the original size -> max pooling when too big, repeat whent too small
            #Do upsampling with repeat
			x_r = theano.tensor.signal.pool.pool_2d(xt, ds = (4,4), st = (4,4), ignore_border=True)
			c11_r = theano.tensor.signal.pool.pool_2d(c11, ds = (4,4), st = (4,4), ignore_border=True)
			c12_r = theano.tensor.signal.pool.pool_2d(p1, ds = (2,2), st = (2,2), ignore_border=True)
                
			c21_r = theano.tensor.signal.pool.pool_2d(c21, ds = (2,2), st = (2,2), ignore_border=True)
			c22_r = p2
            
			c31_r = c31
			c32_r = c32
			c33_r = c33
            
			c41_r = T.extra_ops.repeat(T.extra_ops.repeat(c41,2,axis = 2),2,axis = 3)
			c42_r = T.extra_ops.repeat(T.extra_ops.repeat(c42,2,axis = 2),2,axis = 3)
			c43_r = T.extra_ops.repeat(T.extra_ops.repeat(c43,2,axis = 2),2,axis = 3)
                
			c51_r = T.extra_ops.repeat(T.extra_ops.repeat(c51,4,axis = 2),4,axis = 3)
			c52_r = T.extra_ops.repeat(T.extra_ops.repeat(c52,4,axis = 2),4,axis = 3)
			c53_r = T.extra_ops.repeat(T.extra_ops.repeat(c53,4,axis = 2),4,axis = 3)
                
			stacked_features = T.concatenate([x_r,c11_r,c12_r,c21_r,c22_r,c31_r,c32_r,c33_r,c41_r,c42_r,c43_r,c51_r,c52_r,c53_r],axis =1)
            
			self.activation_volume = stacked_features
			self.c53_r  = c53_r 
            
			#return c53 #resolution should be 16 times smaller that input

		#load VGG params
		if len(os.path.dirname(__file__)) >0:
			paramsValuesVGG = pickle.load(open(os.path.dirname(__file__)+"/models/paramsVGG.pickle","rb"))
		else:
			paramsValuesVGG = pickle.load(open("./models/paramsVGG.pickle","rb"))
		w_c11 = theano.shared(paramsValuesVGG[0])
		b_c11 = theano.shared(paramsValuesVGG[1])
		w_c12 = theano.shared(paramsValuesVGG[2])
		b_c12 = theano.shared(paramsValuesVGG[3])

		w_c21 = theano.shared(paramsValuesVGG[4])
		b_c21 = theano.shared(paramsValuesVGG[5])
		w_c22 = theano.shared(paramsValuesVGG[6])
		b_c22 = theano.shared(paramsValuesVGG[7])

		w_c31 = theano.shared(paramsValuesVGG[8])
		b_c31 = theano.shared(paramsValuesVGG[9])
		w_c32 = theano.shared(paramsValuesVGG[10])
		b_c32 = theano.shared(paramsValuesVGG[11])
		w_c33 = theano.shared(paramsValuesVGG[12])
		b_c33 = theano.shared(paramsValuesVGG[13])

		w_c41 = theano.shared(paramsValuesVGG[14])
		b_c41 = theano.shared(paramsValuesVGG[15])
		w_c42 = theano.shared(paramsValuesVGG[16])
		b_c42 = theano.shared(paramsValuesVGG[17])
		w_c43 = theano.shared(paramsValuesVGG[18])
		b_c43 = theano.shared(paramsValuesVGG[19])

		w_c51 = theano.shared(paramsValuesVGG[20])
		b_c51 = theano.shared(paramsValuesVGG[21])
		w_c52 = theano.shared(paramsValuesVGG[22])
		b_c52 = theano.shared(paramsValuesVGG[23])
		w_c53 = theano.shared(paramsValuesVGG[24])
		b_c53 = theano.shared(paramsValuesVGG[25])

		paramsVGG = [w_c11,b_c11,w_c12,b_c12,  
        w_c21,b_c21,w_c22,b_c22,
        w_c31,b_c31,w_c32,b_c32,w_c33,b_c33,
        w_c41,b_c41,w_c42,b_c42,w_c43,b_c43,
        w_c51,b_c51,w_c52,b_c52,w_c53,b_c53,
        ]
		self.paramsVGG = paramsVGG
		self.outVGG = modelVGG(x, *paramsVGG)


	def reshapeInputImageToActivationVol(self,y):
		bacth,in_channels,H,W = T.shape(y)
		y = y[:,:,0:self.VGGout_resize*(H/self.VGGout_resize),0:self.VGGout_resize*(W/self.VGGout_resize) ]         
		return y[:,:,::4,::4]
    
	def getParams(self):
	    params_values = []
	    for p in range(len(self.paramsVGG)):
	        params_values.append(self.paramsVGG[p].get_value())
	    
	    return params_values

	def setParams(self, params_values):
	    for p in range(len(params_values)):
	        self.paramsVGG[p].set_value(params_values[p])


