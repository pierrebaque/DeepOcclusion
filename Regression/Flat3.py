
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


class Flat3(object):

	def __init__(self, mNet,y_activ_regression,mBGsub,n_leaves,p_drop = 0.0):
        
        
		x_activ = mNet.activation_volume
		size_last_convolution = mNet.nb_activations
		H_ds,W_ds = x_activ.shape[2], x_activ.shape[3]
		batch_size = x_activ.shape[0]
        
        
        #get foreground/background proba
		p_fb = mBGsub.p_fb
		p_foreground = p_fb[:,0,:,:].reshape((batch_size,1,H_ds,W_ds))


        #Decision Flat 2 hidden layers

		self.params_regression = RegFcts.init_FC3_params(size_last_convolution,n_leaves,hidden_0_size = 300,hidden_1_size = 80)

		p_fin=RegFcts.decision_node_flat3(x_activ,self.params_regression[0],
                                                    self.params_regression[1],self.params_regression[2],self.params_regression[3],self.params_regression[4],self.params_regression[5],batch_size,p_drop = p_drop)

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
		self.params_regression = RegFcts.init_FC3_params(self.size_last_convolution,self.n_leaves,hidden_0_size = 300,hidden_1_size = 80)
		return self.params_regression
    
	def get_random_regression_params(self):
		random_params_regression = RegFcts.random_FC3_params(self.size_last_convolution,self.n_leaves,hidden_0_size = 300,hidden_1_size = 80)
		return random_params_regression
    
	def load_regression_params(self,params_list):
        
		self.params_regression[0].set_value(params_list[0])
		self.params_regression[1].set_value(params_list[1])
		self.params_regression[2].set_value(params_list[2])
		self.params_regression[3].set_value(params_list[3])
		self.params_regression[4].set_value(params_list[4])
		self.params_regression[5].set_value(params_list[5])
       
		return
    
	def load_regression_params_fromshared(self,params_list):
        
		self.params_regression[0].set_value(params_list[0].get_value())
		self.params_regression[1].set_value(params_list[1].get_value())
		self.params_regression[2].set_value(params_list[2].get_value())
		self.params_regression[3].set_value(params_list[3].get_value())
		self.params_regression[4].set_value(params_list[4].get_value())
		self.params_regression[5].set_value(params_list[5].get_value())
       
		return


	def save_regression_params(self):       
		params_FC_save = []

		params_FC_save.append(self.params_regression[0].get_value())
		params_FC_save.append(self.params_regression[1].get_value())
		params_FC_save.append(self.params_regression[2].get_value())
		params_FC_save.append(self.params_regression[3].get_value())
		params_FC_save.append(self.params_regression[4].get_value())
		params_FC_save.append(self.params_regression[5].get_value())
    
		return params_FC_save
