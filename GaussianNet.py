import numpy as np
import matplotlib.pyplot as plt
import cv2
import re, os, glob, pickle, shutil
from shutil import *
import time

from theano import *
import theano.tensor as T
theano.__version__
from theano.sandbox.cuda import dnn

import theano
import pandas as pd
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.nnet.conv import conv2d

from random import randint
from theano.compile.nanguardmode import NanGuardMode

import VGG.VGGNet as VGGNet
import VGG.BGsubstract as BGsubstractVGG

from net_functions import *
import Regression.Flat3 as Flat3

from PIL import Image
import copy

import Config

#################

class gaussianNet:
    def __init__(self):
        
        #Choose paramters of training
        struct = 'Flat3'
        CNN_name = 'VGG'
        n_leaves = Config.n_parts - 1 
        self.n_leaves = n_leaves
        
        epsilon = 1e-5
        numerical_normalisation = 1e7

        # X is input matrix and Y is output (full image)
        X = T.ftensor4('X')
        Y_in= T.ftensor4('Y_in')
        batch_size = X.shape[0]
        p_drop = T.scalar('dropout',dtype = 'float32')

        # Building net
        ## Convnet 

        mNet = VGGNet.VGG(X)
        self.mNet = mNet

        x_activ = mNet.activation_volume
        size_last_convolution =mNet.nb_activations
        H_ds,W_ds = x_activ.shape[2], x_activ.shape[3]

        #For VGG, y is already 1/4 of X, but we need to remove margins as it is done in the network
        Y_in_crop = Y_in[:,:,0:H_ds,0:W_ds]


        y_bg= Y_in_crop[:,0:1] > 0
        y_inside= Y_in_crop[:,1:2] > -500
        y_activ_regression= Y_in_crop[:,1:5]

        ## Handling GT
        # y_activ_regression = mNet.reshapeInputImageToActivationVol(Y_in_regression)
        # # y_activ_binary = mNet.reshapeInputImageToActivationVol(Y_in_binary)

        ## Background substraction


        mBGsub = BGsubstractVGG.BGsubstract(x_activ)
        self.mBGsub = mBGsub

        p_fb = mBGsub.p_fb
        p_foreground = p_fb[:,0,:,:].reshape((batch_size,1,H_ds,W_ds))

        ## Regression Network to produce probabilities
        ##Change here to switch between tree and flat

        self.regression_net = Flat3.Flat3(mNet,y_activ_regression, mBGsub,n_leaves,p_drop = p_drop)

        p_leaves = self.regression_net.p_leaves

        ## Gaussian leaves

        params_gaussian =init_all_gaussian_params(n_leaves)#[a_1,s_1,a_2, s_2]
        self.params_gaussian = params_gaussian
        sums_gaussian = init_all_gaussian_sums(n_leaves)

        G=[]
        for l in range(0,n_leaves):
            G.append(gaussian(y_activ_regression,params_gaussian[2*l],
                              params_gaussian[2*l+1])) # outputs a (batch_size,h_ds,w_ds) tensor )


        P_T = 0

        for l in range(0,n_leaves):
            P_T = P_T + G[l]*p_leaves[l]*numerical_normalisation


        #Objective functions

        ## Regression
        #regression_cost =-T.sum((T.log(P_T[:,:,:,:]*y_inside + epsilon)*y_inside))/(T.sum(y_inside))
        regression_cost =-T.sum((T.log(P_T[:,:,:,:]*y_bg + epsilon)*y_bg))/(T.sum(y_bg))

        ## Background
        bg_cost = (T.nnet.binary_crossentropy(p_foreground, y_bg)*(3*y_bg +1*(1-y_bg))).mean()


        # Updates for decision parameter
        ## For regression tree/Flat
        updates_decision = Adam(regression_cost,self.regression_net.params_regression,lr=Config.Regrate)
        updates_bg = Adam(bg_cost,mBGsub.params,lr=Config.BGrate)

        ## Updates for gaussian parameters gaussian_maximisation
        updates_sums = update_sums(p_leaves,G,P_T,y_activ_regression,y_inside,sums_gaussian,numerical_normalisation,epsilon)
        updates_zero_sums = update_sums_to_zero(n_leaves,sums_gaussian)

        ## Updates for gaussian parameters gaussian_maximisation
        updates_gaussian = gaussian_maximisation(p_leaves,G,P_T,y_activ_regression,y_inside,
                                                 params_gaussian,sums_gaussian,numerical_normalisation,epsilon)

        ## Prepare outputs
        all_p = T.concatenate(p_leaves,axis =1)
        all_gaussian_parameters=T.concatenate(params_gaussian)



        #Training functions

        ## For decision tree
        self.train_decision_func = theano.function(inputs=[X,Y_in,In(p_drop, value=0.3)], outputs=[regression_cost],
                                         updates=updates_decision, allow_input_downcast=True,
                                         on_unused_input='warn')
        
        self.train_bg_func = theano.function(inputs=[X,Y_in,In(p_drop, value=0.0)], outputs=[bg_cost],
                                   updates=updates_bg, allow_input_downcast=True,on_unused_input='warn')

        ## For updating gaussians
        self.go_zero_sum_func = theano.function(inputs=[],outputs=[],updates =updates_zero_sums)
        self.train_sums_func = theano.function(inputs=[X,Y_in,In(p_drop, value=0.0)], outputs=[],
                                     updates=updates_sums, allow_input_downcast=True,
                                     on_unused_input='warn')
        self.train_gaussians = theano.function(inputs=[X,Y_in,In(p_drop, value=0.0)], outputs=[],
                                          updates=updates_gaussian, allow_input_downcast=True,
                                          on_unused_input='warn')

        ## Test function
        self.test_function = theano.function(inputs=[X,Y_in,In(p_drop, value=0.0)], outputs=[regression_cost],
                                        updates=[], allow_input_downcast=True,on_unused_input='warn')
        self.run_function = theano.function(inputs=[X,In(p_drop, value=0.0)],
                                       outputs=[p_foreground,all_p,all_gaussian_parameters],
                                       updates=[], allow_input_downcast=True,on_unused_input='warn')
        
        
    ########
        
    #Load batch data for the backpropagation part
    
    def set_data_path(self,em_it):
        self.data_path = Config.labels_folder%em_it + 'trainImg/img%08d.png'
        self.labels_path = Config.labels_folder%em_it + 'trainLabels/labels%08d.txt'


    def load_batch(self,local_training_set_indices,train = True,from_generated = False):
        batch_size = len(local_training_set_indices)

        rgb_list = []
        labels_list = []
        for fid in local_training_set_indices:

            rgb = np.asarray(Image.open(self.data_path%fid))[:,:,0:3]
   
            H,W = np.shape(rgb)[0:2]
            rgb_theano = rgb.transpose((2,0,1))
            rgb_theano = rgb_theano.reshape((1,3,H,W))

            rgb_list.append(rgb_theano)


            if train:
                labels = np.clip(np.loadtxt(self.labels_path%fid),-1000,1000)
 
            H_lab,W_lab = H/Config.CNN_factor,W/Config.CNN_factor
            #print labels.shape
            labels = labels.reshape(H_lab,W_lab,5)
            labels = labels.transpose((2,0,1))
            labels = labels.reshape(1,5,H_lab,W_lab)

            labels_list.append(labels)



        x_in = np.concatenate(rgb_list,axis = 0 )
        y_in = np.concatenate(labels_list,axis = 0 )

        return x_in,y_in

    def optimize_gaussians_online(self,all_indices,gaussian_minibatch_size = 4,from_generated = False):
        print 'optimise online gaussian'
        number_of_minibatches = len(all_indices)/gaussian_minibatch_size
        #TODO make x-free train_gaussians
        print 'go_zero_sum'
        self.go_zero_sum_func()
        #we are computing the sums that will be used to update the gaussians
        for b in range(0,number_of_minibatches):
            local_indices =  all_indices[b*gaussian_minibatch_size:(b+1)*gaussian_minibatch_size]
            #print 'gaussian minibatch',b
            x_in,y_in = self.load_batch(local_indices,train = True,from_generated = from_generated)
            self.train_sums_func(x_in,y_in)

        print 'train gaussians'
        self.train_gaussians(x_in[0:2],y_in[0:2]) 
        return
    
##################

    def train_bg(self,em_it,resume_training_round_BG = 0):
        #learning bg params
        
        self.set_data_path(em_it)
        
        if not os.path.exists(Config.labels_folder%em_it):
            os.mkdir(Config.labels_folder%em_it)

        batch_size = 4
        generated_training_set_size = len(Config.data_augmentation_proportions)*len(Config.cameras_list)*len(Config.img_index_list) 
        train_bg_logs_path = Config.logs_path + 'train_bg_%d.txt'%em_it
        epoch_set_size = 400 #Number of images per epoch
        #first, initialize all trees   

            #initialize params
        if resume_training_round_BG==0:
            f_logs = open(train_bg_logs_path,  'w')
            f_logs.close() 
            if em_it > 1:
                params_bg = pickle.load(open(Config.net_params_path + 'EM%d/params_BG.pickle'%(em_it - 1)))
                self.mBGsub.setParams(params_bg)

        else:
            params_bg = pickle.load(open(Config.net_params_path + 'temp/params_BG_%d.pickle'%(resume_training_round_BG)))
            self.mBGsub.setParams(params_bg)



        #learning regression params

        for r in range(resume_training_round_BG,Config.n_epochs):
            print 'epoch',r
            generated_training_set_order = np.random.permutation(np.arange(0,generated_training_set_size))
            #Train
            av_cost = 0
            for batch in range(0,epoch_set_size/batch_size):
                local_training_set_indices = generated_training_set_order[batch*batch_size:(batch+1)*batch_size]
                x_in,y_in = self.load_batch(local_training_set_indices,train = True,from_generated = True)
                
                print 'x_in shape', x_in.shape
                start = time.time()
                bg_cost = self.train_bg_func(x_in,y_in)[0]
                end = time.time()
                print 'bg cost %f,computed in %f'%(bg_cost,end - start)
                av_cost+=bg_cost


            av_cost = av_cost / (epoch_set_size/batch_size)

            print '###### average train : cost %f'%(av_cost)
            f_logs = open(train_bg_logs_path,  'a')
            f_logs.write('%f'%(av_cost) + '\n')
            f_logs.close()

            #Save Params
            if r%2 ==0:
                #save everything in path
                params_BG  = self.mBGsub.getParams()
                pickle.dump(params_BG,open(Config.net_params_path + 'temp/params_BG_%d.pickle'%r,'wb'))
                
                #Run small test
                self.run_test(em_it,reload_params = False,name = 'test_bg_%d_%d_'%(em_it,r))

                
                
        if not os.path.exists(Config.net_params_path + 'EM%d/'%em_it):
            os.mkdir(Config.net_params_path + 'EM%d/'%em_it)
        params_BG  = self.mBGsub.getParams()
        pickle.dump(params_BG,open(Config.net_params_path + 'EM%d/'%em_it + 'params_BG.pickle','wb'))



    def train_parts(self,em_it,resume_training_round = 0,params_scratch = False,bg_pretrained = True):
        # regression learning params
        
        self.set_data_path(em_it)
        
        batch_size = 4
        generated_training_set_size = (len(Config.data_augmentation_proportions)
        *len(Config.cameras_list)*len(Config.img_index_list))
        epoch_set_size = 400
        update_gaussian_every = 100
        gaussian_fitting_size = 500
        
#         epoch_set_size = 10
#         update_gaussian_every = 5
#         gaussian_fitting_size = 40
        
        train_logs_path = Config.logs_path + 'train_%d.txt'%em_it
        #first, initialize all trees 
        
        #Load BG parameters
        if bg_pretrained:
            '''
            Load BG parameters computed in previous step
            '''
            params_bg = pickle.load(open('./VGG/models/params_BG.pickle'))
            self.mBGsub.setParams(params_bg)

            
        else:
            '''
            Load BG parameters computed in previous step
            '''
            params_bg = pickle.load(open(Config.net_params_path + 'EM%d/params_BG.pickle'%(em_it)))
            self.mBGsub.setParams(params_bg)


            #initialize params
        if resume_training_round==0:
            f_logs = open(train_logs_path,  'w')
            f_logs.close()

            
        if params_scratch:
            init_gaussian_params =init_all_gaussian_params(self.n_leaves)
            load_gaussian_params_fromshared(self.params_gaussian,init_gaussian_params)
            random_reg_params = self.regression_net.get_random_regression_params()
            self.regression_net.load_regression_params(random_reg_params)
            
        else:
        
            if resume_training_round==0:
                if em_it > 1:
                    '''
                    Load parameters from previous EM step
                    '''
                    params_regression= pickle.load(open(Config.net_params_path + 'EM%d/params_regression.pickle'%(em_it - 1)))
                    self.regression_net.load_regression_params(params_regression)
                    gaussian_params = pickle.load(open(Config.net_params_path + 'EM%d/params_gaussian.pickle'%(em_it - 1)))
                    load_gaussian_params(self.params_gaussian,gaussian_params)


            else:
                '''
                Load parameters from previous iteration
                '''
                params_regression= pickle.load(open(Config.net_params_path 
                                                    + 'temp/params_regression_%d.pickle'%(resume_training_round)))
                self.regression_net.load_regression_params(params_regression)
                gaussian_params = pickle.load(open(Config.net_params_path 
                                                   + 'temp/params_gaussian_%d.pickle'%(resume_training_round)))
                load_gaussian_params(self.params_gaussian,gaussian_params)

        #learning regression params

        for r in range(resume_training_round,Config.n_epochs):
            print 'parts epoch',r
            #reinitialize params to the previous value
            generated_training_set_order = np.random.permutation(np.arange(0,generated_training_set_size))
            #Train
            av_cost = 0
            
            for batch in range(0,epoch_set_size/batch_size):

                local_training_set_indices = generated_training_set_order[batch*batch_size:(batch+1)*batch_size]
                x_in,y_in = self.load_batch(local_training_set_indices,train = True,from_generated = True)
                start = time.time()
                cost = self.train_decision_func(x_in,y_in)[0]
                end = time.time()

                av_cost+=cost
                #Optimise gaussian
                local_training_set_indices = generated_training_set_order[batch*batch_size:
                                                                          batch*batch_size+gaussian_fitting_size]

                if batch%update_gaussian_every ==update_gaussian_every - 1:
                    self.optimize_gaussians_online(local_training_set_indices,from_generated=True)


            av_cost = av_cost / (epoch_set_size/batch_size)

            f_logs = open(train_logs_path,  'a')
            f_logs.write('%f'%(av_cost) + '\n')
            f_logs.close()

            #Save Params
            if r%2 ==0:
                
                #save everything in path
                params_regression= self.regression_net.save_regression_params()
                gaussian_params = save_gaussian_params(self.params_gaussian)
                with open(Config.net_params_path  + 'temp/params_regression_%d.pickle'%r,'wb') as a:
                    pickle.dump(params_regression,a)
                with open(Config.net_params_path  + 'temp/params_gaussian_%d.pickle'%r,'wb') as a:
                    pickle.dump(gaussian_params,a)
                    
                #Run small test
                self.run_test(em_it,reload_params = False,name = 'test_%d_%d_'%(em_it,r))

            #Compute test loss

        params_regression= self.regression_net.save_regression_params()
        gaussian_params = save_gaussian_params(self.params_gaussian)
        
        if not os.path.exists(Config.net_params_path  + 'EM%d/'%em_it):
            os.mkdir(Config.net_params_path  + 'EM%d/'%em_it)
            
        with open(Config.net_params_path  + 'EM%d/params_regression.pickle'%em_it,'wb') as a:
            pickle.dump(params_regression,a)
        with open(Config.net_params_path  + 'EM%d/params_gaussian.pickle'%em_it,'wb') as a:
            pickle.dump(gaussian_params,a)

            

#Functions to visualize

    @staticmethod
    def load_batch_run(fid_indices,cam,path = './'):
        batch_size = len(fid_indices)

        rgb_list = []
        labels_list = []
        for fid in fid_indices:
        #load rgb

            rgb = np.asarray(Image.open(Config.rgb_name_list[cam]%fid))[:,:,0:3]

            H,W = np.shape(rgb)[0:2]
            rgb_theano = rgb.transpose((2,0,1))
            rgb_theano = rgb_theano.reshape((1,3,H,W))

            rgb_list.append(rgb_theano)



        x_in = np.concatenate(rgb_list,axis = 0 )

        return x_in


    def run_test(self,em_it,epoch = -1,reload_params = True,fid_indices = Config.img_index_list[0:1],cam = 0,name = 'test',bg_pretrained = True):
        
        if reload_params:
        
            if epoch == -1:
                
                if bg_pretrained:
                    '''
                    Load default BG parameters
                    '''
                    params_bg = pickle.load(open('./VGG/models/params_BG.pickle'))
                    self.mBGsub.setParams(params_bg)
                else:
                    params_bg = pickle.load(open(Config.net_params_path + 'EM%d/params_BG.pickle'%(em_it)))
                    self.mBGsub.setParams(params_bg)
                    
                params_regression= pickle.load(open(Config.net_params_path + 'EM%d/params_regression.pickle'%(em_it)))
                self.regression_net.load_regression_params(params_regression)
                gaussian_params = pickle.load(open(Config.net_params_path + 'EM%d/params_gaussian.pickle'%(em_it)))
                load_gaussian_params(self.params_gaussian,gaussian_params)


            else:
                
                if bg_pretrained:
                    '''
                    Load default BG parameters
                    '''
                    params_bg = pickle.load(open('./VGG/models/params_BG.pickle'))
                    self.mBGsub.setParams(params_bg)

                else:
                    params_bg = pickle.load(open(Config.net_params_path + 'temp/params_BG_%d.pickle'%(epoch)))
                    self.mBGsub.setParams(params_bg)
                    
                params_regression= pickle.load(open(Config.net_params_path + 'temp/params_regression_%d.pickle'%(epoch)))
                self.regression_net.load_regression_params(params_regression)
                gaussian_params = pickle.load(open(Config.net_params_path + 'temp/params_gaussian_%d.pickle'%(epoch)))
                load_gaussian_params(params_gaussian,gaussian_params)
                
        #Load data and run
        x_in =  gaussianNet.load_batch_run(fid_indices,cam)
        print x_in.shape
        p_foreground,all_p,all_gaussian_parameters = self.run_function(x_in)
        

        #Display on top of original image and save
        p_bin_np = np.asarray(all_p[0]).transpose(1,2,0)
        bin_np_resize = p_bin_np.repeat(Config.CNN_factor,axis =0).repeat(Config.CNN_factor,axis = 1)
        img = copy.copy(x_in[0].transpose((1,2,0)))

        p_foreground_np = np.asarray(p_foreground[0]).transpose(1,2,0)
        foreground= p_foreground_np[:,:,0].repeat(4,axis =0).repeat(4,axis = 1)

        H_re,W_re = bin_np_resize.shape[0:2] # should only correspond to the small crop on the side

        #Horizontal
        for i in range(0,3):
            #bool_crit = np.logical_and((np.argmax(bin_np_resize[:,:,0:n_cats],axis =2) == i),(foreground>0.2))
            #img[0:H_re,0:W_re,i] = img[0:H_re,0:W_re,i]*0.5 + (bool_crit)*100
            img[0:H_re,0:W_re,i] = img[0:H_re,0:W_re,i]*0.5 + (bin_np_resize[:,:,i]>0.15)*100

        plt.imsave(Config.img_logs + name + '0.png', img)

        img = copy.copy(x_in[0].transpose((1,2,0)))
        for i in range(0,3):
            img[0:H_re,0:W_re,i] = img[0:H_re,0:W_re,i]*0.5 + (bin_np_resize[:,:,i+3]>0.15)*100

        plt.imsave(Config.img_logs + name + '1.png', img)

        img = copy.copy(x_in[0].transpose((1,2,0)))
        for i in range(0,3):
            img[0:H_re,0:W_re,0] = img[0:H_re,0:W_re,i]*0.5 + (foreground)*100

        plt.imsave(Config.img_logs + name + '2.png', img)
        
        return p_bin_np
                 
                 
    def run_inference(self,em_it = -1,bg_pretrained = True,regression_pretrained = False,params_scratch = False,verbose = False):
        
        if bg_pretrained:
            '''
            Load default BG parameters
            '''
            params_bg = pickle.load(open('./VGG/models/params_BG.pickle'))
            self.mBGsub.setParams(params_bg)
        else:
            '''
            Load BG parameters computed in previous step
            '''
            params_bg = pickle.load(open(Config.net_params_path + 'EM%d/params_BG.pickle'%(em_it)))
            self.mBGsub.setParams(params_bg)


        
        # self.data_path = Config.labels_folder%em_it + 'trainImg/img%08d.png'
        # self.labels_path = Config.labels_folder%em_it + 'trainLabels/labels%08d.txt'
        if params_scratch:
            init_gaussian_params =init_all_gaussian_params(self.n_leaves)
            load_gaussian_params_fromshared(self.params_gaussian,init_gaussian_params)
            random_reg_params = self.regression_net.get_random_regression_params()
            self.regression_net.load_regression_params(random_reg_params)
        else:
            
            if regression_pretrained:
                params_regression= pickle.load(open(Config.net_params_path + 'params_regression.pickle'))
                self.regression_net.load_regression_params(params_regression)
                gaussian_params = pickle.load(open(Config.net_params_path + 'params_gaussian.pickle'))
                load_gaussian_params(self.params_gaussian,gaussian_params)

                
            else:
                params_regression= pickle.load(open(Config.net_params_path + 'EM%d/params_regression.pickle'%(em_it)))
                self.regression_net.load_regression_params(params_regression)
                gaussian_params = pickle.load(open(Config.net_params_path + 'EM%d/params_gaussian.pickle'%(em_it)))
                load_gaussian_params(self.params_gaussian,gaussian_params)
                 
        #Prepare output folders
        if em_it > -1 :
            emit_parts_root = Config.parts_root_folder%(em_it+1)
        else:
            emit_parts_root = Config.parts_root_folder
            
        if not os.path.exists(emit_parts_root):
            os.mkdir(emit_parts_root)


        for cam in Config.cameras_list:
            if not os.path.exists(emit_parts_root + 'c%d/'%cam):
                os.mkdir(emit_parts_root + 'c%d/'%cam)

            for fid in Config.img_index_list:
                if verbose:
                    print "Running inference for cam %d, fid %d:"%(cam,fid)

                x_in =  gaussianNet.load_batch_run([fid ],cam)
                p_foreground,all_p,all_gaussian_parameters = self.run_function(x_in)
                p_bin_np = np.asarray(all_p[0]).transpose(1,2,0)
                
                if bg_pretrained:
                    p_foreground_np = np.asarray(p_foreground[0]).transpose(1,2,0)
                    parts_out = np.concatenate([p_bin_np>0.25,p_foreground_np>0.2],axis =2)
                    
                else:
                    #Load background subtraction
                    #Output is going to be weighted average of probability predicted by network and background-sub
                    bkg = cv2.imread(Config.bkg_path%(cam,fid))[:,:,0]>0
                    bkg_soft = 0.65*bkg + 0.35*(1-bkg)
                    bkg_factor = bkg_soft/(1-bkg_soft)
                    p_foreground_np = np.asarray(p_foreground[0]).transpose(1,2,0)*bkg_factor[:,:,np.newaxis]
                    parts_out = np.concatenate([p_bin_np*bkg_factor[:,:,np.newaxis]>0.3,p_foreground_np>0.2],axis =2)

                    

                        
                np.save(emit_parts_root+ 'c%d/%d.npy'%(cam,fid),parts_out)

        np.savetxt(emit_parts_root + 'gaussian_params.txt',np.int32(all_gaussian_parameters),fmt='%d')
        
        
                 


