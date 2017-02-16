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

os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=cpu,floatX=float32"
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
#from pom_evaluator import POM_evaluator
import POMLayers1

from EM_funcs import *
config.allow_gc =False

import Config


As = Config.EM_As
A_blacks = Config.EM_Ablacks

em_it = int(sys.argv[1])

room = POM_room(em_it)

POMLayers1.room = room #TODO : modify POMLayers1 so that we don't need room, just config
POMLauncher = POMLayers1.pomLayer()
POMLauncher.set_POM_params(a = As[em_it],alpha_black = A_blacks[em_it],prior_factor = Config.EM_POM_prior)

folder_out = Config.POM_out_folder%em_it

def runsave1(fid,folder_out):
    last = len(room.img_index_list)
    if fid < last:
        Q_out,Z_out,Shift = POMLauncher.run_POM(fid)
        room.save_dat(Q_out,fid,folder_out,verbose = True)

if not os.path.exists(folder_out):
    os.makedirs(folder_out)

for block in range(0,len(room.img_index_list)/Config.n_threads+1):
    if Config.verbose:
        print 'POM parallel starting %d-th frame'%(Config.n_threads*block)
    fids = range(Config.n_threads*block,Config.n_threads*(block+1))
    Parallel(n_jobs=Config.n_threads)(delayed(runsave1)(fid,folder_out) for fid in fids)
    


if Config.do_GS_validation:    
    #Change image list to run validation
    room.img_index_list = Config.val_index_list

    #Run grid search for params
    folder_out = Config.GS_out + './grid_search_EM%d'%em_it
    
    if not os.path.exists(Config.GS_out):
        os.makedirs(Config.GS_out)

    if not os.path.exists(folder_out):
        os.makedirs(folder_out)

    for fid_idx in Config.val_index_list:
        folder_fid_name = folder_out + '/%d'%fid_idx
        if not os.path.exists(folder_fid_name):
            os.makedirs(folder_fid_name)


    def runSearch(a,p,alpha_black):

        print 'Setting a = %f, p = %f'%(a,p)
        alphas_np = np.ones(room.n_parts,dtype='float32')
        alphas_np[0:room.n_parts-1] = a
        alphas_np[-1] = alpha_black
        POMLauncher.alphas.set_value(alphas_np)

        #Unaries
        POMLauncher.priors_factor.set_value(np.asarray(np.log(0.001)*p,dtype='float32'))
        #
        
        for fid in range(0,len(Config.val_index_list)):
            folder_fid_name = folder_out + '/%d'%Config.val_index_list[fid]
            print 'Running for frame','/%d'%Config.val_index_list[fid]
            Q_out,Z,Shift = POMLauncher.run_POM(fid,getZ = False,useshift= False)
            room.save_dat_withpath(Q_out,folder_fid_name+'/_%f_%f_%f_.dat'%(a,p,alpha_black))
            
    As = Config.GS_As
    Ps = Config.GS_Ps
    Ablacks = Config.GS_Ablacks
    grid_list = [(a,p,a_black) for a in As for p in Ps for a_black in Ablacks]
    print 'grid_list'
    print grid_list


    for block in range(0,len(grid_list)/Config.n_threads+1):
        params = grid_list[Config.n_threads*block:Config.n_threads*(block+1)]
        Parallel(n_jobs=Config.n_threads)(delayed(runSearch)(a,p,a_black) for a,p,a_black in params)


