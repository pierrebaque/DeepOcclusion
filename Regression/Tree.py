
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


class Tree(object):

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

    #Decision tree
		self.params_regression =RegFcts.init_all_tree_params(depth,size_last_convolution)#[w_1,b_1]


    #defining all decision nodes, breadth first, storing all the probas tensors in a list
		tree_probas_list = RegFcts.create_tree_probas_list(x_activ,p_foreground,self.params_regression,depth,batch_size)
        
    # Put Output in the leaves

		p_leaves =[]
		for l in range(0,n_leaves):
			p_leaves.append(tree_probas_list[pow(2,depth) - 1 +l])


        #Define objects

		self.p_leaves = p_leaves
		self.n_leaves = n_leaves
		self.size_last_convolution = size_last_convolution
		self.depth = depth


	def init_regression_params(self):
		self.params_regression = RegFcts.init_all_tree_params(self.depth,self.size_last_convolution)
		return self.params_regression
    
	def load_regression_params(self,params_list):
        
		number_nodes = len(params_list)/4
		for node in range(0,number_nodes):
			self.params_regression[4*node].set_value(params_list[4*node])
			self.params_regression[4*node+1].set_value(params_list[4*node+1])
			self.params_regression[4*node+2].set_value(params_list[4*node+2])
			self.params_regression[4*node+3].set_value(params_list[4*node+3])
        
		return

        

	def save_regression_params(self):  
        
		params_tree_save = []
		number_nodes = len(self.params_regression)/4
		for node in range(0,number_nodes):
			params_tree_save.append(self.params_regression[4*node].get_value())
			params_tree_save.append(self.params_regression[4*node+1].get_value())
			params_tree_save.append(self.params_regression[4*node+2].get_value())
			params_tree_save.append(self.params_regression[4*node+3].get_value())
    
		return params_tree_save
        
