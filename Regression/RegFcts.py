from theano import *
import theano.tensor as T
theano.__version__
from theano.sandbox.cuda import dnn

import theano
import pandas as pd
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
import cPickle, gzip, numpy
from theano.tensor.nnet.conv import conv2d

from random import randint





def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def init_weights_np(shape):
    return floatX(np.random.randn(*shape) * 0.01)

def decision_node(last_CNN_layer,p_in,w0,b0,w1,b1,batch_size):
    linear_0 = conv2d(last_CNN_layer, w0)
    hidden = T.nnet.sigmoid(linear_0+T.repeat(b0,batch_size,axis=0).dimshuffle(0, 'x', 'x', 'x'))
    linear_1 = conv2d(hidden, w1)
    p_out = p_in*T.nnet.sigmoid(linear_1+T.repeat(b1,batch_size,axis=0).dimshuffle(0, 'x', 'x', 'x'))
    
    return p_out, p_in-p_out

def decision_node_flat(last_CNN_layer,w0,b0,w1,b1,batch_size):
    linear_0 = conv2d(last_CNN_layer, w0)
    hidden = T.nnet.sigmoid(linear_0+T.repeat(b0,batch_size,axis=0).dimshuffle(0, 1, 'x', 'x'))
    linear_1 = conv2d(hidden, w1)
    x_out = T.nnet.sigmoid(linear_1+T.repeat(b1,batch_size,axis=0).dimshuffle(0, 1, 'x', 'x'))
    p_out = x_out / T.sum(x_out, axis = 1, keepdims = True)
    
    return p_out

def decision_node_flat3(last_CNN_layer,w0,b0,w1,b1,w2,b2,batch_size,p_drop = 0.0):
    #first layer
    last_CNN_layer_drop = dropout(last_CNN_layer,p_drop)
    linear_0 = conv2d(last_CNN_layer_drop, w0)
    hidden_0 = T.nnet.sigmoid(linear_0+T.repeat(b0,batch_size,axis=0).dimshuffle(0, 1, 'x', 'x'))
    hidden_0_drop = dropout(hidden_0,p_drop)
    #second layer
    linear_1 = conv2d(hidden_0_drop, w1)
    hidden_1 = T.nnet.sigmoid(linear_1+T.repeat(b1,batch_size,axis=0).dimshuffle(0, 1, 'x', 'x'))
    #third layer
    linear_2 = conv2d(hidden_1, w2)
    x_out = T.nnet.sigmoid(linear_2+T.repeat(b2,batch_size,axis=0).dimshuffle(0, 1, 'x', 'x'))
    p_out = x_out / T.sum(x_out, axis = 1, keepdims = True)
    
    return p_out

def dropout(x, p=0.0):
    #if p > 0:
    retain_prob = 1 - p
    #x has smaples in first dim and features in second dim, what we want is get only some features => binomial on second dim
    srng = RandomStreams()
    x *= srng.binomial(n=1, size=(x.shape[1],), p=retain_prob, dtype=theano.config.floatX).dimshuffle('x',0,'x','x')
    x /= retain_prob
    return x


def init_all_tree_params(tree_depth,size_last_convolution,hidden_decision_size = 30):
    all_params_tree = []  
    for depth in range(0,tree_depth):
        n_nodes_at_level = pow(2,depth)
        for i in range(0,n_nodes_at_level):
            #first layer
            all_params_tree.append(init_weights((hidden_decision_size,size_last_convolution,1,1)))
            all_params_tree.append(shared(np.zeros((1,),dtype=theano.config.floatX), borrow=True,name = 'b_%d_%d'%(depth,i)))
            #second layer
            all_params_tree.append(init_weights((1,hidden_decision_size,1,1)))
            all_params_tree.append(shared(np.zeros((1,),dtype=theano.config.floatX), borrow=True,name = 'b_%d_%d'%(depth,i)))

            
    return all_params_tree

# def init_FC_params(size_last_convolution,n_leaves,hidden_decision_size = 80):
#     all_FC_params = []  

#     #first layer
#     all_FC_params.append(init_weights((hidden_decision_size,size_last_convolution,1,1)))
#     all_FC_params.append(shared(np.zeros((1,hidden_decision_size),dtype=theano.config.floatX), borrow=True,name = 'b_hidden'))
#     #second layer
#     all_FC_params.append(init_weights((n_leaves,hidden_decision_size,1,1)))
#     all_FC_params.append(shared(np.zeros((1,n_leaves),dtype=theano.config.floatX), borrow=True,name = 'b_last'))

            
#     return all_FC_params

def init_FC3_params(size_last_convolution,n_leaves,hidden_0_size = 300, hidden_1_size =80 ):
    all_FC_params = []  
    
    #first layer
    all_FC_params.append(init_weights((hidden_0_size,size_last_convolution,1,1)))
    all_FC_params.append(shared(np.zeros((1,hidden_0_size),dtype=theano.config.floatX), borrow=True,name = 'b_hidden0'))

    #second layer
    all_FC_params.append(init_weights((hidden_1_size,hidden_0_size,1,1)))
    all_FC_params.append(shared(np.zeros((1,hidden_1_size),dtype=theano.config.floatX), borrow=True,name = 'b_hidden1'))
    #third layer
    all_FC_params.append(init_weights((n_leaves,hidden_1_size,1,1)))
    all_FC_params.append(shared(np.zeros((1,n_leaves),dtype=theano.config.floatX), borrow=True,name = 'b_last'))

            
    return all_FC_params

def random_FC3_params(size_last_convolution,n_leaves,hidden_0_size = 300, hidden_1_size =80 ):
    '''
    Same as init_FC3_params, except that all params are in numpy form, it is used to re-initialise params
    '''
    all_FC_params = []  
    
    #first layer
    all_FC_params.append(init_weights_np((hidden_0_size,size_last_convolution,1,1)))
    all_FC_params.append(np.zeros((1,hidden_0_size),dtype=theano.config.floatX))

    #second layer
    all_FC_params.append(init_weights_np((hidden_1_size,hidden_0_size,1,1)))
    all_FC_params.append(np.zeros((1,hidden_1_size),dtype=theano.config.floatX))
    #third layer
    all_FC_params.append(init_weights_np((n_leaves,hidden_1_size,1,1)))
    all_FC_params.append(np.zeros((1,n_leaves),dtype=theano.config.floatX))

            
    return all_FC_params



def create_tree_probas_list(x_activ, p_foreground,params_tree,tree_depth,batch_size):
    tree_probas_list = [p_foreground]
    idx=0 # it is the n-th node in the breadth first order
    for depth in range(0,tree_depth):
        for node_index in range(0,pow(2,depth)):
            p = tree_probas_list[idx]
            w0 =params_tree[4*idx]
            b0 =params_tree[4*idx+1]
            w1 =params_tree[4*idx+2]
            b1 = params_tree[4*idx+3]
            p_1,p_2 = decision_node(x_activ,p,w0,b0,w1,b1,batch_size)
            idx=idx+1
            tree_probas_list.append(p_1)
            tree_probas_list.append(p_2)
    
    return tree_probas_list



##saving parameters of the tree into the list params tree save
def save_tree_params():
    params_tree_save = []
    number_nodes = len(params_tree)/4
    for node in range(0,number_nodes):
        params_tree_save.append(params_tree[4*node].get_value())
        params_tree_save.append(params_tree[4*node+1].get_value())
        params_tree_save.append(params_tree[4*node+2].get_value())
        params_tree_save.append(params_tree[4*node+3].get_value())
    
    return params_tree_save

##Loading the parameters of the tree from params tree load
def load_tree_params(params_tree_load):
    number_nodes = len(params_tree_load)/4
    for node in range(0,number_nodes):
        params_tree[4*node].set_value(params_tree_load[4*node])
        params_tree[4*node+1].set_value(params_tree_load[4*node+1])
        params_tree[4*node+2].set_value(params_tree_load[4*node+2])
        params_tree[4*node+3].set_value(params_tree_load[4*node+3])
        
    return


##
def save_FC_params(params_FC):
    params_FC_save = []

    params_FC_save.append(params_FC[0].get_value())
    params_FC_save.append(params_FC[1].get_value())
    params_FC_save.append(params_FC[2].get_value())
    params_FC_save.append(params_FC[3].get_value())
    
    return params_FC_save

##Loading the parameters of the tree from params tree load
def load_FC_params(params_FC,params_FC_load):

    params_FC[0].set_value(params_FC_load[0])
    params_FC[1].set_value(params_FC_load[1])
    params_FC[2].set_value(params_FC_load[2])
    params_FC[3].set_value(params_FC_load[3])
        
    return



            
