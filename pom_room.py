import random
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import cv2
import re, os, glob, pickle, shutil
import random

import Config
class POM_room(object):
    
    #def __init__(self,pom_file_path,parts_root_folder,img_index_list,n_parts,resize_pom = 4,cameras_list = range(7),HW_grid = (-1,-1),Sigma_factor = 2,with_templates = True):
    def __init__(self,em_it,with_templates = True):
        
        # Config about POM templates
        self.parts_root_folder = Config.parts_root_folder%em_it
        self.n_parts = Config.n_parts #Number of classes of classifier. Including foreground class which is the last one
        self.pom_file_path =  Config.pom_file_path
        self.resize_pom = Config.resize_pom #Difference in ratio between dimensions for POM file and images saved (4 when images come from VGG)
        self.cameras_list = Config.cameras_list
        self.n_cams = len(self.cameras_list)
        self.img_index_list = Config.img_index_list #images to use
        self.image_path_format = self.parts_root_folder + 'c%d/%d.npy'
        self.gaussian_params_path = self.parts_root_folder +'gaussian_params.txt'
        self.H_grid, self.W_grid = Config.H_grid, Config.W_grid #Usefullf if grid defined
                    

        
        if with_templates:

            #Size of parts compared to Sigma
            self.Sigma_factor = Config.Sigma_factor

            # Config about POM images
            self.H,self.W = self.get_HW_from_img()
            self.extract_templates()
        

    def get_HW_from_img(self):
        '''
        Output : Shape of the images we are going to use as input.
        '''
        
        im = np.load(self.image_path_format%(0,self.img_index_list[0]))
        H,W = im.shape[0:2]
        return H,W


    def extract_BB_coordinates(self,camera):
        '''
        In : camera id
        Out : List of all bounding boxes coordinates on this view, as defined by the pom file.
        '''

        f = open(self.pom_file_path, 'r')
        lines = f.readlines()
        bounding_boxes =[]
        current_object =1
        for i,line in enumerate(lines):

            if line.find('RECTANGLE %d'%camera) > -1:
                bounding_boxes.append(self.parse_BB_from_line(line))
        return bounding_boxes

    def parse_BB_from_line(self,line):
        '''
        In : line string
        Out : coordinates of the box in the parsed line, where we set random 0-size coordinates for non-visible and resize to match the resizing used in the background sub.
        '''
        resize = self.resize_pom
        line_split = line.split(' ')
        if line_split[3] == 'notvisible\n':
            rand_H,rand_W = random.randint(0,self.H-1),random.randint(0,self.W-1)
            return [rand_W,rand_H,rand_W,rand_H]
        else:
            return [np.int(line_split[3])/resize,np.int(line_split[4])/resize,np.int(line_split[5])/resize,np.int(line_split[6])/resize]


    #Extract coordinates of BBs as in normal POM file

    def extract_templates(self):
        '''
        Output: Array of shape (n_cameras*n_parts,n_boxes,4) which contains templates 2D coordinates in projection on each camera.
        '''
        N_cameras = len(self.cameras_list)

        H,W =  self.H,self.W#resize_pom)
        n_parts = self.n_parts

        bboxes_cam_list =[]

        for cam in self.cameras_list:
            bboxes_cam_list.append(self.extract_BB_coordinates(cam))

        #Load the gaussian parameters
        gauss_params = np.loadtxt(self.gaussian_params_path,dtype = 'int32')
        gauss_params = gauss_params.reshape((n_parts-1,2,4))

        templates_array =np.zeros((N_cameras*n_parts,len(bboxes_cam_list[0]),4),dtype = 'int32')


        for i in range(0,len(bboxes_cam_list[0])):
            for cam in self.cameras_list:
                bboxes = bboxes_cam_list[cam]
                bb_midx = (bboxes[i][3] + bboxes[i][1])/2
                bb_midy = (bboxes[i][2] + bboxes[i][0])/2
                bb_sizex = (bboxes[i][3] - bboxes[i][1])
                bb_sizey = (bboxes[i][2] - bboxes[i][0])

                for part in range(n_parts-1):
                    alphax = gauss_params[part,0,0]
                    alphay = gauss_params[part,0,1]
                    sigmax = gauss_params[part,1,0]
                    sigmay = gauss_params[part,1,1]
                    # Compute coordinates of new bb
                    x0 = bb_midx - (alphax*bb_sizex)/1000 - (sigmax*self.Sigma_factor*bb_sizex)/1000
                    y0 = bb_midy - (alphay*bb_sizey)/1000 - (sigmay*self.Sigma_factor*bb_sizey)/1000
                    x1 = bb_midx - (alphax*bb_sizex)/1000 + (sigmax*self.Sigma_factor*bb_sizex)/1000
                    y1 = bb_midy - (alphay*bb_sizey)/1000 + (sigmay*self.Sigma_factor*bb_sizey)/1000
                    # Crop coordinates to stay inside image
                    x0 = max(x0,0)
                    y0 = max(y0,0)
                    x1 = min(x1,H-1)
                    y1 = min(y1,W-1)
                    if (x1 - x0) > 3 and (y1 - y0) > 3:
                        templates_array[n_parts*cam + part,i,:] = np.asarray([x0,y0,x1,y1])
                    else:
                        rand_H,rand_W = random.randint(0,H-1),random.randint(0,W-1)
                        templates_array[n_parts*cam + part,i,:] = np.asarray([rand_H,rand_W,rand_H,rand_W])
                #now add full box in last position
                x0 = max(bboxes[i][1],0)
                y0 = max(bboxes[i][0],0)
                x1 = min(bboxes[i][3],H-1)
                y1 = min(bboxes[i][2],W-1)

                templates_array[n_parts*cam + n_parts - 1,i,:] = np.asarray([x0,y0,x1,y1])
        
        self.templates_array = templates_array


    def load_images_stacked_old(self,fid,verbose = False):
        im_out = []
        for cam in self.cameras_list:
            for part in range(self.n_parts):
                if verbose:
                    print "Loading " + self.image_path_format%(cam,part,self.img_index_list[fid])
                im = cv2.imread(self.image_path_format%(cam,part,self.img_index_list[fid]))
                im_out.append(im[:,:,0]>0)

        image = np.asarray(np.stack(im_out))

        return image
    
    def load_images_stacked(self,fid,verbose = False):
        im_out = []
        for cam in self.cameras_list:
            if verbose:
                print "Loading " + self.image_path_format%(cam,self.img_index_list[fid])
            im = np.load(self.image_path_format%(cam,self.img_index_list[fid]))
            im_out.append(im)

        image = np.asarray(np.concatenate(im_out,axis = 2)).transpose((2,0,1))

        return image


    def get_indices_above(self,image,threshold = 0.6):

        n_vars = self.templates_array.shape[1]
        img_fg = image[self.n_parts-1::self.n_parts]
        templates_fg = self.templates_array[self.n_parts-1::self.n_parts]

        aux = np.cumsum(img_fg,axis = 1)
        integral_img = np.cumsum(aux,axis = 2)

        scores = np.zeros(n_vars)
        sizes = np.zeros(n_vars)

        for cam in range(templates_fg.shape[0]):
            scores += (integral_img[cam,templates_fg[cam,:,0],templates_fg[cam,:,1]] + integral_img[cam,templates_fg[cam,:,2],templates_fg[cam,:,3]] - integral_img[cam,templates_fg[cam,:,0],templates_fg[cam,:,3]] - integral_img[cam,templates_fg[cam,:,2],templates_fg[cam,:,1]]) 
            sizes += np.maximum((templates_fg[cam,:,2]-templates_fg[cam,:,0])*(templates_fg[cam,:,3]-templates_fg[cam,:,1]),4.0)
        scores = scores / sizes

        return np.where(scores > threshold)[0],scores

    def plot_output(self,Q_out,fid,cam,part,thresh = 0.9,iteration = -1,Shift = []):

        image =self.load_images_stacked(fid)

        img_cam = image[self.n_parts*cam+part]
        Q_plot = Q_out[iteration]
        if len(Shift) == 0:
            templates_cam = self.templates_array[self.n_parts*cam+part] 
        else:
            templates_cam = self.templates_array[self.n_parts*cam+part] + Shift[iteration][self.n_parts*cam+part]
        H,W = image.shape[1:]
        img_out = np.zeros((H,W,3))
        img_out[:,:,0] = img_cam
        Q_abs = np.ones((H,W))
        for i in range(templates_cam.shape[0]):
            if Q_plot[i] > 0.001:
            #if scores[i]>0.6:
                Q_abs[templates_cam[i,0]:templates_cam[i,2],templates_cam[i,1]:templates_cam[i,3]] *= 1-Q_plot[i]
                img_out[:,:,2] = 1-Q_abs
                if Q_plot[i] > thresh:
                    cv2.rectangle(img_out,(templates_cam[i,1],templates_cam[i,0]),(templates_cam[i,3],templates_cam[i,2]),(0,1,0))

        img_out[:,:,2] = 1-Q_abs

        plt.imshow(img_out)
        plt.show()
        
    def save_dat(self,Q_out,fid,folder_out,iteration = -1):
        out_path= folder_out + '%08d.dat'%self.img_index_list[fid]

        if not os.path.exists(folder_out):
            os.makedirs(folder_out)
        Q_save = Q_out[iteration]
        f = open(out_path,'w')
        for i in range(Q_save.shape[0]):
            string = '%d %f\n'%(i,Q_save[i])
            f.write(string)
        f.close()
        
    def save_dat_withpath(self,Q_out,out_path,iteration = -1):

        Q_save = Q_out[iteration]
        f = open(out_path,'w')
        for i in range(Q_save.shape[0]):
            string = '%d %f\n'%(i,Q_save[i])
            f.write(string)
        f.close()
        
    def get_coordinates_from_Q(self,Q_det,q_thresh = 0.5):
        det_ID = np.asarray(np.where(Q_det>q_thresh))[0]
        det_coordinates = np.float32(np.stack([det_ID/self.W_grid,det_ID%self.W_grid]).T)

        return det_coordinates

    def get_coordinates_from_Q_reduced(self,Q_det,indices_reduced,q_thresh = 0.5):
        det_ID_reduced = np.asarray(np.where(Q_det>q_thresh))[0]
        det_ID = indices_reduced[det_ID_reduced]
        det_coordinates = np.float32(np.stack([det_ID/self.W_grid,det_ID%self.W_grid]).T)

        return det_coordinates
    
    def show_detection_MAP(self,X_coordinates,X_map = np.zeros((1,1))):
        '''
        Allows to compare two maps:
        X_coordinates will be in yellow and X_map in blue
        '''
        n_x = X_coordinates.shape[0]

        if np.sum(X_map) ==0:
            add =1
            X_map = np.zeros((self.H_grid,self.W_grid))
        else:
            add =2

        for i_x in range(n_x):
            X_map[int(X_coordinates[i_x,0]) -2:int(X_coordinates[i_x,0]) +2,int(X_coordinates[i_x,1]) - 2 : int(X_coordinates[i_x,1]) + 2] += add
        print 'invert before print'
        plt.imshow(X_map[:,::-1],interpolation='nearest')
        plt.show()

        return X_map

    def show_heatmap(self,Q):
        plt.imshow(np.log(Q.reshape(self.H_grid,self.W_grid)))
        plt.colorbar()
        plt.show()


        
