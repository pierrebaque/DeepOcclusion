import numpy as np
import matplotlib.pyplot as plt
import cv2
import re, os, glob, pickle, shutil,sys, random,copy
from shutil import *

sys.path.append('../roi_pooling/theano-roi-pooling/')
sys.path.append('./POM')


import hickle as hkl
from theano import *
import theano.tensor as T
theano.__version__
from theano.sandbox.cuda import dnn
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d
from theano.compile.nanguardmode import NanGuardMode
config.allow_gc =False

from random import randint

import cPickle, gzip
import time

from PIL import Image

#from pom_funcs import *
from pom_room import POM_room
from pom_evaluator import POM_evaluator

import Config
import VGG.VGGNet as VGGNet
from roi_pooling import ROIPoolingOp
from net_functions import *



class unariesNet:
    def __init__(self,load_pretrained = True):
        #Path save params
        self.path_save_params = './Unaries/trainedModels/'

        #logs
        self.train_logs_path = 'Unaries/train_unaries.txt'
        self.test_logs_path = 'Unaries/test_unaries.txt'
        
        #Oputput
        self.unaries_out_path = Config.unaries_path
        
        print "Preparing room"
        #Prepare room and evaluator
        #Create room
        self.room = POM_room(Config.parts_root_folder,with_templates= True)
        #Prepare evaluator which will let us load GT
        self.evaluator = POM_evaluator(self.room,GT_labels_path_json = '../NDF_peds/data/ETH/labels_json/%08d.json')
        
        
        print "Initializing Unaries Network"
        #DEFINE NETWORK
        
        '''
        Remark, when using ROIPooling, y axis first then x axis for ROI pooling
        '''
        p_h,p_w = 3,3 #"size of extracted features vector"

        epsilon = 1e-7
        X = T.ftensor4('X')
        Ybb= T.fvector('Ybb')
        batch_size = X.shape[0]
        p_drop = T.scalar('dropout',dtype = 'float32')
        t_rois = T.fmatrix()

        # Building net
        ## Convnet 

        mNet = VGGNet.VGG(X)

        c53_r = mNet.c53_r

        op = ROIPoolingOp(pooled_h=p_h, pooled_w=p_w, spatial_scale=1.0)


        roi_features = op(c53_r, t_rois)[0]#T.concatenate(op(c53, t_rois),axis = 0)

        #Initialize weights
        w0_u = init_weights((512*p_h*p_w,1024),name = 'w0_unaries')
        b0_u = init_weights((1024,),name = 'b0_unaries',scale = 0)
        w1_u = init_weights((1024,1024),name = 'w1_unaries')
        b1_u = init_weights((1024,),name = 'b1_unaries',scale = 0)
        w2_u = init_weights((1024,2),name = 'w2_unaries')
        b2_u = init_weights((2,),name = 'b2_unaries',scale = 0)

        paramsUnaries = [w0_u,b0_u,w1_u,b1_u,w2_u,b2_u]


        # #New network
        features_flat = roi_features.reshape((-1,512*p_h*p_w))
        x1 = T.clip(T.dot(features_flat,w0_u) + b0_u,0,100000)
        x1_drop = dropout(x1,p_drop)
        x2 = T.clip(T.dot(x1_drop,w1_u) + b1_u,0,100000)
        x2_drop = dropout(x2,p_drop)
        p_out = softmax(T.dot(x2_drop,w2_u) + b2_u)
        log_p_out = stab_logsoftmax(T.dot(x2_drop,w2_u) + b2_u)

        ## Classification
        #loss = (T.nnet.binary_crossentropy(p_out[:,0], Ybb)).mean()
        loss = -(log_p_out[:,0]*Ybb + log_p_out[:,1]*(1-Ybb)).mean()


        # Updates for decision parameter
        ## For regression tree/Flat
        updates_loss = Adam(loss,paramsUnaries,lr=2e-4)
        updates_loss_VGG = Adam(loss,paramsUnaries+mNet.paramsVGG,lr=1e-6)

        self.train_func = theano.function(inputs=[X,t_rois,Ybb,In(p_drop, value=0.5)], 
                                     outputs=[T.exp(log_p_out),loss], updates=updates_loss_VGG,
                                     allow_input_downcast=True,on_unused_input='warn')
        
        self.test_func = theano.function(inputs=[X,t_rois,Ybb,In(p_drop, value=0.0)],
                                    outputs=[T.exp(log_p_out),loss], updates=[],
                                    allow_input_downcast=True,on_unused_input='warn')
        
        self.run_func = theano.function(inputs=[X,t_rois,In(p_drop, value=0.0)],
                                   outputs=T.exp(log_p_out), updates=[],
                                   allow_input_downcast=True,on_unused_input='warn')
        
        self.play_func = theano.function(inputs=[X,t_rois,In(p_drop, value=0.0)],
                                    outputs=roi_features, updates=[],
                                    allow_input_downcast=True,on_unused_input='warn')
        
        self.features_func = theano.function(inputs=[X,t_rois,In(p_drop, value=0.0)],
                                   outputs=x2, updates=[],
                                   allow_input_downcast=True,on_unused_input='warn')
       
        
        #Define self objects
        self.paramsUnaries = paramsUnaries
        self.mNet = mNet
        
        #Load pretrained params
        if load_pretrained:
            print "loading pretrained params"
            params_to_load = pickle.load(open('./VGG/models/paramsUnaries.pickle'))
            self.setParams(params_to_load)
            params_VGG= pickle.load(open('./VGG/models/paramsVGGUnaries.pickle'))
            mNet.setParams(params_VGG)

            
        
        
    def getParams(self):
        params_values = []
        for p in range(len(self.paramsUnaries)):
            params_values.append(self.paramsUnaries[p].get_value())

        return params_values

    def setParams(self,params_values):
        for p in range(len(params_values)):
            self.paramsUnaries[p].set_value(params_values[p])
            
    
    def train(self,resume_epoch = 0,fine_tune = True):

        test_fid = -1

        if resume_epoch ==0:
            f_logs = open(self.train_logs_path,  'w')
            f_logs.close()
            f_logs = open(self.test_logs_path,  'w')
            f_logs.close()

        else:
            params_to_load = pickle.load(open(self.path_save_params + 'params_Unaries_%d.pickle'%resume_epoch))
            self.setParams(params_to_load)
            if fine_tune:
                params_VGG= pickle.load(open(self.path_save_params + 'params_VGG_%d.pickle'%(resume_epoch)))
                self.mNet.setParams(params_VGG)





        for epoch in range(resume_epoch,80):
            costs = []
            for fid in range(0,2):
                for cam in range(7):
                    print 'Epoch %d, FID %d, cam %d'%(epoch,fid,cam)
                    x,rois_np,labels = self.load_batch_train(fid,cam)
                    #visualize_batch(x,rois_np,labels)
                    p_out_train,loss = self.train_func(x,rois_np,labels)
                    print 'Loss Unaries',loss
                    costs.append(loss)
                    #x_out_test = test_func(rgb_theano,rois_np)
            

            #Save params
            if epoch%2 ==0:
                params_to_save  = self.getParams()
                pickle.dump(params_to_save,open(self.path_save_params  +'params_Unaries_%d.pickle'%epoch,'wb'))
                if fine_tune:
                    params_VGG  = self.mNet.getParams()
                    pickle.dump(params_VGG,open(self.path_save_params  +"params_VGG_%d.pickle"%epoch,'wb'))


            av_cost = np.mean(costs)
            f_logs = open(self.train_logs_path,  'a')
            f_logs.write('%f'%(av_cost) + '\n')
            f_logs.close()
            
                        #Test loss
            if test_fid > 0:
                test_costs = []
                fid = test_fid
                for cam in range(7):
                        print 'Test Epoch %d, FID %d, cam %d'%(epoch,fid,cam)
                        x,rois_np,labels = self.load_batch_train(fid,cam)
                        #visualize_batch(x,rois_np,labels)
                        p_out_test,test_loss = self.test_func(x,rois_np,labels)
                        test_costs.append(test_loss)


                av_test_cost = np.mean(test_costs)
                f_logs = open(self.test_logs_path,  'a')
                f_logs.write('%f'%(av_test_cost) + '\n')
                f_logs.close()
                
                
    #FUNCTIONS TO LOAD DATA
    
    def get_rois(self,fid,cam):
        n_parts = Config.n_parts
        thresh =0.40
        #####
        #Loading the image preprocessed with segmentor
        templates_array = self.room.templates_array
        image = self.room.load_images_stacked(fid)

        indices = templates_array.shape[1]
        indices_reduced,scores = self.room.get_indices_above(image,threshold= thresh)
        templates_array_reduced = templates_array[:,indices_reduced,:]
        #####
        #Now we have preselected bboxes
        print templates_array_reduced.shape
        templates = templates_array_reduced[n_parts -1 + n_parts*cam]

        crit_no_null = (templates[:,2]-templates[:,0])*(templates[:,3]-templates[:,1]) > 400 #We don't want empty boxes
        templates_no_null = templates[crit_no_null]
        indices_no_null = indices_reduced[crit_no_null]

        # rois fill
        rois_np = np.zeros((templates_no_null.shape[0],5)).astype(np.single)

        rois_np[:,1] = templates_no_null[:,1]
        rois_np[:,2] = templates_no_null[:,0]
        rois_np[:,3] = templates_no_null[:,3]
        rois_np[:,4] = templates_no_null[:,2]


        return rois_np,indices_no_null

    def get_rgb(self,fid,cam):
        #Load rgb image
        rgb = np.asarray(Image.open(Config.rgb_name_list[cam]%self.room.img_index_list[fid]))[:,:,0:3]
        H,W = np.shape(rgb)[0:2]
        rgb_theano = rgb.transpose((2,0,1))
        rgb_theano = rgb_theano.reshape((1,3,H,W))

        return rgb_theano

    def get_labels(self,fid,indices_no_null,rad = 1 ):
        #rad = radius to validate a detection
        #Load ground_truth
        GT_coordinates = np.floor(self.evaluator.get_GT_coordinates_fromjson(fid)).astype(np.int)
        det_coordinates = self.room.get_coordinates_from_Q_reduced(indices_no_null*0 + 1.0,indices_no_null).astype(np.int)

        #Find positive examples
        MAP_OK = np.zeros((self.room.H_grid,self.room.W_grid))

        for X in GT_coordinates.tolist() :
            MAP_OK[X[0],X[1]] = 1

    #     plt.imshow(MAP_OK)
    #     plt.show()
        #Maybe overkill but will use integral image in order to computer afterward iintegral inside area for detections
        MAP_OK_integral = MAP_OK.cumsum(axis =0).cumsum(axis =1)

        def integral_array(MAP_OK_integral,X):
            room = self.room
            return (MAP_OK_integral[min(X[0]+rad,room.H_grid-1),min(X[1]+rad,room.W_grid-1)]
        + MAP_OK_integral[max(X[0]-rad,0),max(X[1]-rad,0)] 
        - MAP_OK_integral[min(X[0]-rad,room.H_grid-1),min(X[1]+rad,room.W_grid-1)] 
        - MAP_OK_integral[min(X[0]+rad,room.H_grid-1),min(X[1]-rad,room.W_grid-1)])

        labels = [integral_array(MAP_OK_integral,X) > 0 for X in det_coordinates.tolist()]

        return np.asarray(labels).astype(np.int)

    def load_batch_train(self,fid,cam,sample_equal = True):
        rois_np,indices_no_null = self.get_rois(fid,cam)
        x = self.get_rgb(fid,cam)
        labels = self.get_labels(fid,indices_no_null)

        #We resample in order to have the same number of positive and negative examples
        if sample_equal:
            n_pos = labels.sum()
            ratio = n_pos*1.0/(labels.shape[0]-n_pos)

            select = []
            for i,lab in enumerate(labels.tolist()):
                if lab:
                    select.append(True)
                else:
                    if random.random() < ratio:
                        select.append(True)
                    else:
                        select.append(False)

            rois_np = rois_np[select]
            labels = labels[select]



        return x,rois_np,labels

    def load_batch_run(self,fid,cam):
        rois_np,indices_no_null = self.get_rois(fid,cam)
        x = self.get_rgb(fid,cam)

        return x,rois_np,indices_no_null


    def visualize_batch(self,x,rois_np,labels,i = 0,CNN_factor = 4):
        import copy
        rgb = copy.copy(x[i].transpose((1,2,0))) 

        for idbb, bbox in enumerate(rois_np.tolist()[:]):
            color = (100,0,0) if labels[idbb] else (0,0,100)
            bbox = np.asarray(bbox).astype(np.int)
            cv2.rectangle(rgb,(Config.CNN_factor*bbox[1],Config.CNN_factor*bbox[2]),
                          (Config.CNN_factor*bbox[3],Config.CNN_factor*bbox[4]),color,3)

        plt.imshow(rgb)
        plt.show()

    def visualize_positives(self,x,rois_np,labels,i = 0,CNN_factor = 4):
        import copy
        rgb = copy.copy(x[i].transpose((1,2,0))) 

        for idbb, bbox in enumerate(rois_np.tolist()[:]):
            color = (100,0,0)
            if labels[idbb]:
                bbox = np.asarray(bbox).astype(np.int)
                cv2.rectangle(rgb,(Config.CNN_factor*bbox[1],Config.CNN_factor*bbox[2]),
                              (Config.CNN_factor*bbox[3],Config.CNN_factor*bbox[4]),color,3)

        plt.imshow(rgb)
        plt.show()
        
        
    # FUNCTIONS TO RUN UNARIES
    #TOFINISH
    
    def run_bulk(self,fid_list = np.arange(len(Config.img_index_list))):
        n_bboxes = self.room.templates_array.shape[1]
        for fid in fid_list:
            print "FID", fid
            scores = np.zeros((self.room.n_cams,n_bboxes)) -10
            for cam in range(self.room.n_cams):
                x,rois_np,indices_no_null= self.load_batch_run(fid,cam)
                p_out_test = self.run_func(x,rois_np)
                scores[cam,indices_no_null] = np.log(p_out_test[:,0])

            np.save(self.unaries_out_path%Config.img_index_list[fid],scores)
            
    def run_test(self,fid = 0, cam =0):
        x,rois_np,l= self.load_batch_run(fid,cam)
        p_out_test = self.run_func(x,rois_np)
        self.visualize_positives(x,rois_np,p_out_test[:,0]>0.8)
        
        
    def run_bulk_features(self,fid_list = np.arange(len(Config.img_index_list)),save_features = True):
        n_bboxes = self.room.templates_array.shape[1]
        for fid in fid_list:
            print "FID", fid
            scores = np.zeros((self.room.n_cams,n_bboxes)) -10
            features = np.zeros((self.room.n_cams,n_bboxes,1024))
            for cam in range(self.room.n_cams):
                x,rois_np,indices_no_null= self.load_batch_run(fid,cam)
                p_out_test = self.run_func(x,rois_np)
                scores[cam,indices_no_null] = np.log(p_out_test[:,0])
                x_2_features = self.features_func(x,rois_np)
                features[cam,indices_no_null,:] = x_2_features


            np.save(self.unaries_out_path%Config.img_index_list[fid],scores)
            if save_features:
                np.save(Config.unaries_path_features%Config.img_index_list[fid],features)



    def run_features(self,fid = 0, cam =0):
        x,rois_np,l= self.load_batch_run(fid,cam)
        x_2_features = self.features_func(x,rois_np)
        return np.asarray(x_2_features)


            
            