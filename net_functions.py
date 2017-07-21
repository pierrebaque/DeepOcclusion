#Putting data to regression format

#This time, we learn the middle and the bottom right corner

import numpy as np
from random import randint

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
#from theano.compile.nanguardmode import NanGuardMode

def to_regression_format(assignment_rect,H,W):
    #creating matrices to get relative bounding box coordinates
    X_matrix = np.repeat(np.reshape(np.arange(H),(H,1)),W,axis =1)
    Y_matrix = np.repeat(np.reshape(np.arange(W),(1,W)),H,axis =0)
    
    out = np.zeros((H,W,4))

    out[:,:,0] = (assignment_rect[:,:,0,0] + assignment_rect[:,:,2,0])/2 - X_matrix
    out[:,:,1] = (assignment_rect[:,:,0,1]+assignment_rect[:,:,2,1])/2 - Y_matrix
    out[:,:,2] = assignment_rect[:,:,2,0] - X_matrix
    out[:,:,3] = assignment_rect[:,:,2,1] - Y_matrix

    return out
            
                        

    

def prepare_regression_gt(assignment_rect_gt,assignment_binary_gt):
 
    n_gt_images,H,W = np.shape(assignment_binary_gt)
    #Have to compute target size from the structure of the CNN

    gt_regression = np.zeros((n_gt_images,H,W,4))
    gt_binary = np.zeros((n_gt_images,H,W))

    for fid in range(0,n_gt_images):

        #put to format for each pixel x0,y0,x3,y3
        assignment_rect_regression = to_regression_format(assignment_rect_gt[fid],H,W)
        #add to the table
        gt_regression[fid] = assignment_rect_regression
        gt_binary[fid] = assignment_binary_gt[fid]
    
    return gt_regression,gt_binary


#Functions for the gaussian voting scheme, learning etc...

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape,name = None,scale = 0.01):
    return theano.shared(floatX(np.random.randn(*shape) * scale),name)

def init_gaussian(dim):
    alpha = np.ones(dim)
    sigma = 1000 * np.ones(dim)
    for i in range(dim):
        alpha[i] = randint(-500,500)
    
    return theano.shared(floatX(alpha)) , theano.shared(floatX(sigma))

def rectify(X):
    return T.maximum(X, 0.)

def softmax(X):
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

def stab_logsoftmax(X):
    #To implement
    x_norm = X - X.max(axis=1).dimshuffle(0, 'x')
    return x_norm - T.log(T.sum(T.exp(x_norm),axis = 1)).dimshuffle(0, 'x')

def dropout(X, p=0.):
    retain_prob = 1 - p
    srng = RandomStreams()
    X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
    X /= retain_prob
    return X

def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates




def gaussian(y,a,s):
    a_broad =  a.dimshuffle('x',0,'x', 'x')
    s_broad = s.dimshuffle('x',0,'x', 'x')
    
    G = 1/T.sqrt(2*np.pi)*1/s_broad*T.exp(-(y - a_broad)**2/(2*s_broad**2))
    G_prod = T.prod(G,axis =1)
    return G_prod.dimshuffle(0,'x',1,2)  


def Adam(cost, params, lr=0.002, b1=0.1, b2=0.001, e=1e-8):
    updates = []
    grads = T.grad(cost, params)
    i = theano.shared(np.float32(0.))
    i_t = i + 1.
    fix1 = 1. - (1. - b1)**i_t
    fix2 = 1. - (1. - b2)**i_t
    lr_t = lr * (T.sqrt(fix2) / fix1)
    for p, g in zip(params, grads):
        m = theano.shared(np.float32(p.get_value() * 0.))
        v = theano.shared(np.float32(p.get_value() * 0.))
        m_t = (b1 * g) + ((1. - b1) * m)
        v_t = (b2 * T.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (T.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))
    return updates

def Adam_fromgrad(grads, lr=0.002, b1=0.1, b2=0.001, e=1e-8):
    updates = []
    i = theano.shared(np.float32(0.))
    i_t = i + 1.
    fix1 = 1. - (1. - b1)**i_t
    fix2 = 1. - (1. - b2)**i_t
    lr_t = lr * (T.sqrt(fix2) / fix1)
    for p, g in zip(params, grads):
        m = theano.shared(np.float32(p.get_value() * 0.))
        v = theano.shared(np.float32(p.get_value() * 0.))
        m_t = (b1 * g) + ((1. - b1) * m)
        v_t = (b2 * T.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (T.sqrt(v_t) + e)
        p_t = p - (lr_t * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))
    updates.append((i, i_t))
    return updates


def SGD(cost, params, lr=0.002):
    updates = []
    grads = T.grad(cost, params)

    for p, g in zip(params, grads):
        p_t = p - (lr * g)
        updates.append((p, p_t))

    return updates

def gaussian_maximisation(p,G,P_T,Y_in_regression,Y_in_binary,params_gaussian,sums_gaussian,numerical_normalisation,epsilon):
    sigma_epsilon_regulariser = 1.0
    updates =[]
    n_leaves = len(p)
    #repeat binary for compatibility in multiplicaiton also
    Y_in_binary_repeat = T.repeat(Y_in_binary,4,axis =1)
    #loop through leaves
    for l in range(0,n_leaves):
        #update alpha
        new_alpha = sums_gaussian[3*l]/sums_gaussian[3*l+2]
        updates.append((params_gaussian[2*l],new_alpha))
        
        #update sigma 
        new_sigma = T.sqrt(sums_gaussian[3*l+1]/sums_gaussian[3*l+2] + sigma_epsilon_regulariser)
        updates.append((params_gaussian[2*l+1],new_sigma))
    return updates

        
def update_sums(p,G,P_T,Y_in_regression,Y_in_binary,sums_gaussian,numerical_normalisation,epsilon):

    updates =[]
    n_leaves = len(p)
    #repeat binary for compatibility in multiplicaiton also
    Y_in_binary_repeat = T.repeat(Y_in_binary,4,axis =1)
    #loop through leaves
    for l in range(0,n_leaves):
        #Define the EM auxiliary probability xi. Put it to the right format for compatibility in the multiplication
        xi_l = (numerical_normalisation*p[l]*G[l])/(P_T+epsilon)
        xi_l_repeat = T.repeat(xi_l,4,axis = 1)
        
        #update sum of denominator
        new_denominator_sum = sums_gaussian[3*l+2]+T.sum(xi_l*Y_in_binary)
        updates.append((sums_gaussian[3*l+2],new_denominator_sum))
        
        #update alpha sum
        new_alpha_sum = sums_gaussian[3*l]+T.sum(xi_l_repeat*Y_in_binary_repeat*Y_in_regression,axis =[0,2,3])
        updates.append((sums_gaussian[3*l],new_alpha_sum))
        
        #update sigma sum
        alpha_extended =(new_alpha_sum/new_denominator_sum).dimshuffle('x',0,'x','x') #we extend alpha for the calc below
        new_sigma_sum = sums_gaussian[3*l+1] + T.sum(xi_l_repeat*Y_in_binary_repeat*(Y_in_regression -alpha_extended)**2 ,axis =[0,2,3])
        
        updates.append((sums_gaussian[3*l+1],new_sigma_sum))
    return updates
    

def init_all_gaussian_params(n_leaves):
    all_params_gaussian = []  
    for i in range(0,n_leaves):
        alpha,sigma = init_gaussian(4)
        all_params_gaussian = all_params_gaussian + [alpha,sigma]
        
    return all_params_gaussian

def init_all_gaussian_sums(n_leaves):
    all_sums_gaussian = []  
    for i in range(0,n_leaves):
        alpha_sums= theano.shared(floatX(np.zeros(4)))
        sigma_sums = theano.shared(floatX(np.zeros(4)))
        numerator_sums = theano.shared(floatX(0.0))
        all_sums_gaussian = all_sums_gaussian + [alpha_sums,sigma_sums,numerator_sums]
        
    return all_sums_gaussian

def update_sums_to_zero(n_leaves,all_sums_gaussian):
    zero_sums_update = []  
    for l in range(0,n_leaves):
        zero_sums_update.append( (all_sums_gaussian[3*l],theano.shared(floatX(np.zeros(4)))))
        zero_sums_update.append( (all_sums_gaussian[3*l+1],theano.shared(floatX(np.zeros(4)))))
        zero_sums_update.append( (all_sums_gaussian[3*l+2],theano.shared(floatX(0.0))))
        
    return zero_sums_update




##saving gaussian parameters from the leaf into the list gaussian params save
def save_gaussian_params(params_gaussian):
    gaussian_params_save = []
    number_leaves = len(params_gaussian)/2
    for l in range(0,number_leaves):
        gaussian_params_save.append(params_gaussian[2*l].get_value())
        gaussian_params_save.append(params_gaussian[2*l+1].get_value())
    
    return gaussian_params_save

##Loading the gaussian parameters from the leaves 
def load_gaussian_params(params_gaussian,gaussian_params_load):
    number_leaves = len(gaussian_params_load)/2
    for l in range(0,number_leaves):
        params_gaussian[2*l].set_value(gaussian_params_load[2*l])
        params_gaussian[2*l+1].set_value(gaussian_params_load[2*l+1])
    
    return

##Loading the gaussian parameters from the leaves 
def load_gaussian_params_fromshared(params_gaussian,gaussian_params_load):
    number_leaves = len(gaussian_params_load)/2
    for l in range(0,number_leaves):
        params_gaussian[2*l].set_value(gaussian_params_load[2*l].get_value())
        params_gaussian[2*l+1].set_value(gaussian_params_load[2*l+1].get_value())
    
    return
