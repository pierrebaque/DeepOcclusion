import os
import Config
import numpy as np
import random
from PIL import Image
import cv2
import math
import matplotlib.pyplot as plt
from IO_funcs import *

def SampleZ(em_it):
    #Extract bounding box coordinates
    W,H =  get_HW(Config.pom_file_path)

    bboxes_cam_list = []
    for cam in Config.cameras_list:
        bboxes = extract_BB_coordinates(Config.pom_file_path,cam)
        bboxes_cam_list.append(bboxes)

    #Sample according to .dat file and save
    out_dir = Config.Z_sample_folder%em_it

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    
    for fid in Config.img_index_list:
        
        write_file = out_dir + '%08d.txt'%fid
        f = open(write_file, 'w')
        f.write(Config.pom_file_path + "\n")
        f.write('rectID,personID,modified,a1,b1,c1,d1,a2,b2,c2,d2,a3,b3,c3,d3,a4,b4,c4,d4,a5,b5,c5,d5,a6,b6,c6,d6,a7,b7,c7,d7\n')

        Q_loc = get_table(Config.POM_out_folder%em_it + '%08d.dat'%(fid))
        flat_q = np.clip(Q_loc,1e-3,0.999999)
        detections = np.where(flat_q>0.95)
        
        pid = 0
        for detection in detections[0]:
            if flat_q[detection] >random.random():
                string = "%d,%d,%d"%(detection,pid,1)
                for cam in Config.cameras_list:
                    bboxes =bboxes_cam_list[cam]
                    string += ",%d,%d,%d,%d"%(bboxes[detection][0],bboxes[detection][1],
                                              bboxes[detection][2],bboxes[detection][3])
                f.write(string + '\n')
                pid += 1
        f.close()

def isinside(i,j,coordinates):
    return coordinates[1] <= i and coordinates[3]>=i and coordinates[0] <= j and coordinates[2] >= j

def getheight(coordinates):
    return coordinates[3] - coordinates[1]

def getshift(i,j,coordinates):
    #We split in 3*3 parts
    x_0 =  ((coordinates[1] + coordinates[3])/2.0 - i) / (coordinates[3] - coordinates[1])
    y_0 =  ((coordinates[0] + coordinates[2])/2.0 - j) / (coordinates[2] - coordinates[0])
    
    x_1 =  ((coordinates[3])*1.0 - i) / (coordinates[3] - coordinates[1])
    y_1 =  ((coordinates[2])*1.0 - j) / (coordinates[2] - coordinates[0])
    
    return np.asarray((x_0,y_0,x_1,y_1))                

def prepare_Labels(em_it):
    
    #Prepare folders
    if not os.path.exists(Config.labels_folder%em_it):
        os.mkdir(Config.labels_folder%em_it)

    train_img_dir = Config.labels_folder%em_it + 'trainImg/'
    train_labels_dir = Config.labels_folder%em_it + 'trainLabels/'

    if not os.path.exists(train_img_dir):
        os.mkdir(train_img_dir)

    if not os.path.exists(train_labels_dir):
        os.mkdir(train_labels_dir)
        
    labels_dir = Config.Z_sample_folder%em_it
    labels_path = labels_dir + '%08d.txt'
    
    #Get dimensions of original images
    img = np.asarray(Image.open(Config.rgb_name_list[0]%(Config.img_index_list[0])))
    H,W = img.shape[0:2]
    
    CNN_factor = Config.CNN_factor

    out_id =0
    for i_frame,fid in enumerate(Config.img_index_list):

        #Reading the labels file
        f = open(labels_path%fid, 'r')

        detections =[[] for cam in Config.cameras_list]
        for lid,line in enumerate(f):
            if lid >1:
                rect_coordinates = np.asarray(np.fromstring(line, dtype=int, sep=','))
                for cam in Config.cameras_list:
                    detections[cam].append(rect_coordinates[3+cam*4:3+cam*4 +4 ])

        #Loop through cameras

        for cam_id in Config.cameras_list:
            #print 'fid,cam_id',fid,cam_id
            #Get the map of labels that we want to predict
            out_map = np.zeros((H/CNN_factor,W/CNN_factor,5)) -1
            if Config.use_bg_pretrained:
                bkg = 1
            else:
                bkg = np.asarray(Image.open(Config.bkg_path%(cam_id,fid))) 
            
            for I in range(H/CNN_factor):
                for J in range(W/CNN_factor):
                    selected_rectangle = [-1,-1,-1,-1]
                    for rect in detections[cam_id ]:
                        #print rect
                        if isinside(CNN_factor*I,CNN_factor*J,rect):
                            if selected_rectangle[0] == -1 or selected_rectangle[3] < rect[3]: #We checked which rectangle is in front of the other
                                selected_rectangle = rect
                    #Now we store what we need for i,j
                    if selected_rectangle[0]>-1:
                        inside = 1
                        out_map[I,J,0] = inside #It is under the dimension of the full image
                        out_map[I,J,1:] = getshift(CNN_factor*I,CNN_factor*J,selected_rectangle)
            

            if Config.use_bg_pretrained == False:
                H_bkg,W_bkg = bkg.shape #Not exactly the same as out map because of border treatment of CNN
                out_map[0:H_bkg,0:W_bkg,0] *= bkg
                out_map[H_bkg:,:,0] = 0
                out_map[:,W_bkg:,0] = 0

            #Do data augmentation with cropping and resizing and save
            img = np.asarray(Image.open(Config.rgb_name_list[cam_id]%fid)) 

            for proportion_crop in Config.data_augmentation_proportions:
                #print 'transfo',proportion_crop
                proportion_crop = int(proportion_crop)
                print proportion_crop
                OK_to_select = False
                count = 0
                while OK_to_select==False:
                    top_left_corner_x, top_left_corner_y= (random.randint(0,(proportion_crop-1)*H/proportion_crop),
                    random.randint(0,(proportion_crop-1)*W/proportion_crop))
                    count += 1
                    if np.sum(out_map[top_left_corner_x/4:top_left_corner_x/4 + H/(4*proportion_crop) ,
                                      top_left_corner_y/4:top_left_corner_y/4 + W/(4*proportion_crop),0]>0) >10 or count >10:
                        OK_to_select = True


                img_out = ( img[top_left_corner_x:top_left_corner_x + H/proportion_crop,
                                top_left_corner_y:top_left_corner_y + W//proportion_crop] )

                label_out = (out_map[top_left_corner_x/4:top_left_corner_x/4 + int(math.ceil(H/(4.0*proportion_crop))),
                                     top_left_corner_y/4:top_left_corner_y/4 + int(math.ceil(W/(4.0*proportion_crop)))] )

                img_out = cv2.resize(img_out,dsize = (0,0),fx = proportion_crop/2.0 , fy = proportion_crop/2.0)

                if proportion_crop > 1:
                    label_out = label_out.repeat(proportion_crop/2,axis= 0).repeat(proportion_crop/2,axis= 1)
                else:
                    label_out = label_out[::2,::2]

                #Little hack to adjust the border (half a pixel )
                if proportion_crop == 4:
                    label_out = label_out[0:H/8,0:W/8]
                #print label_out.shape
    #             plt.imshow(label_out[:,:,1])
    #             plt.show()
    #             break

                #Save labels in txt
                label_flat = label_out.reshape((label_out.shape[0]*label_out.shape[1],5))

                with open(train_labels_dir+'labels%08d.txt'%out_id,'wb') as f:
                    np.savetxt(f,np.asarray(1000*label_flat,dtype='int32'),fmt = '%d')

                #Save img in png
                plt.imsave(train_img_dir+'img%08d.png'%out_id,img_out)
                out_id +=1

    #     break


