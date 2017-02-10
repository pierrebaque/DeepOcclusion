import numpy as np
import os
import cv2
import sys
#sys.setrecursionlimit(1000000)

import Config

def init_parts(bkg_path,em_it = 1):
    emit_parts_root = Config.parts_root_folder%em_it
    if not os.path.exists(emit_parts_root):
        os.mkdir(emit_parts_root)

    for cam in Config.cameras_list:
        if not os.path.exists(emit_parts_root + 'c%d/'%cam):
            os.mkdir(emit_parts_root + 'c%d/'%cam)

        for fid in Config.img_index_list:
            print cam,fid
            im = cv2.imread(bkg_path%(cam,fid))[:,:,0]>0
            H,W = im.shape[0:2]
            parts_out = np.zeros((H,W,Config.n_parts))
            parts_out[:,:,-1] = im

            np.save(emit_parts_root+ 'c%d/%d.npy'%(cam,fid),parts_out)
    #Initialize gaussian parts
    gaussian_parts = np.zeros(8*(Config.n_parts-1))+5
    np.savetxt(emit_parts_root + 'gaussian_params.txt', gaussian_parts)
    
            
