
# coding: utf-8

# In[2]:

import matplotlib
matplotlib.use("nbagg")
import math
import numpy as np
import matplotlib.pyplot as plt
import cv2
import re, os, glob, pickle, shutil,sys
from random import randint
import time
from shutil import *

os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu0,floatX=float32"
import theano
import theano.tensor as T
from theano import *
theano.__version__
from theano.sandbox.cuda import dnn

import pandas as pd
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


from theano.compile.nanguardmode import NanGuardMode

from joblib import Parallel, delayed
import multiprocessing


#from pom_funcs import *
from pom_room import POM_room
from pom_evaluator import POM_evaluator
#import POMLayers1
from EM_funcs import *
import ZtoY
config.allow_gc =False


import Config


# In[11]:



#Start iterations
#create a new room
from GaussianNet import gaussianNet
em_it = 0
parts_predictor = gaussianNet()
verbose = True

if Config.use_bg_pretrained:
    #Initialise parts with background
    parts_predictor.run_inference(0,bg_pretrained = Config.use_bg_pretrained,params_scratch = True)
    
else:
    #Initialise parts with BKG-sub
    #init_parts(Config.bkg_path)
    pass


for em_it in range(1,7):
    
    if em_it > 1:
        os.system("python POM_parallel.py " + str(em_it))

        #Sample Z and prepare labels for NN
    ZtoY.SampleZ(em_it)
    ZtoY.prepare_Labels(em_it)
    
    if not Config.use_bg_pretrained:
        parts_predictor.train_bg(em_it)
    parts_predictor.train_parts(em_it,bg_pretrained = Config.use_bg_pretrained, params_scratch = False)
    parts_predictor.run_inference(em_it,bg_pretrained = Config.use_bg_pretrained)


# In[37]:

parts_predictor.run_test(0)


fid = 0
Q_out,Z_out,Shift = POMLauncher.run_POM(fid)
room.plot_output(Q_out,fid,1,8,thresh = 0.5,iteration=-1)


# In[45]:

room.save_dat(Q_out,fid,folder_out)




