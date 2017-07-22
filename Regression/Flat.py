
#Compute convolutions like in AlexNet
#define graph from input to activation volume
import os
import numpy as np
from PIL import Image, ImageDraw

import theano
import theano.tensor as T
from theano.tensor.nnet.conv import conv2d
 
import RegFcts
RegFcts = reload(RegFcts)
from AlexNet import CNNet as CNNet
from AlexNet import BGsubstract as BGsubstract


class Flat(object):

	def __init__(self, mNet,y_activ_regression,mBGsub,depth):
        

		n_leaves = 2**depth
        
		x_activ = mNet.activation_volume
		size_last_convolution = mNet.nb_activations
		H_ds,W_ds = x_activ.shape[2], x_activ.shape[3]
		batch_size = x_activ.shape[0]
        
        
        #get foreground/background proba
		p_fb = mBGsub.p_fb
		p_foreground = p_fb[:,0,:,:].reshape((batch_size,1,H_ds,W_ds))


        #Decision Flat 2 layers

		self.params_regression = RegFcts.init_FC_params(size_last_convolution,n_leaves,hidden_decision_size = 80)

		p_fin=RegFcts.decision_node_flat(x_activ,self.params_regression[0],
                                                    self.params_regression[1],self.params_regression[2],self.params_regression[3],batch_size)

		p_fin = p_fin*T.repeat(p_foreground,n_leaves,axis = 1)
        
        # Put Output in the leaves


		p_leaves =[]
		for l in range(0,n_leaves):
			p_leaves.append(p_fin[:,l:l+1,:,:])



        #Define objects

		self.p_leaves = p_leaves
		self.n_leaves = n_leaves
		self.size_last_convolution = size_last_convolution


	def init_regression_params(self):
		self.params_regression = RegFcts.init_FC_params(self.size_last_convolution,self.n_leaves)
		return self.params_regression
    
	def load_regression_params(self,params_list):
        
		self.params_regression[0].set_value(params_list[0])
		self.params_regression[1].set_value(params_list[1])
		self.params_regression[2].set_value(params_list[2])
		self.params_regression[3].set_value(params_list[3])
        
		return

	def save_regression_params(self):       
		params_FC_save = []

		params_FC_save.append(self.params_regression[0].get_value())
		params_FC_save.append(self.params_regression[1].get_value())
		params_FC_save.append(self.params_regression[2].get_value())
		params_FC_save.append(self.params_regression[3].get_value())
    
		return params_FC_save
