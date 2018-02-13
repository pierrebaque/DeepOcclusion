import pickle
import numpy as np
import os
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,device=gpu2,floatX=float32"
from PIL import Image
import json
from scipy import ndimage
import math
import random

from UnariesNet import unariesNet
import MyConfig


class unaryNet2(unariesNet):
    def __init__(self):
        unariesNet.__init__(self)
        self.trainImgsPath = MyConfig.trainImgPath
        self.trainLabelsPath = MyConfig.trainLabelPath
        #Path save params
        self.path_save_params = MyConfig.unaries_params_path
        self.train_logs_path = MyConfig.unaries_train_log
        
        self.jsonFile = MyConfig.jsonFile

    def loadRGBimg(self, dataPath, imgName):
        rgb = np.asarray(Image.open(dataPath + imgName))[:, :, 0:3]
        H, W = np.shape(rgb)[0:2]
        self.imgH = H
        self.imgW = W
        rgb_theano = rgb.transpose((2, 0, 1))
        rgb_theano = rgb_theano.reshape((1, 3, H, W))
        return rgb_theano

    
    def getIoU(self, boxA, boxB):
        #find intersection box
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # compute the area of intersection rectangle
        interArea = max(0,xB - xA + 1) * max(0,yB - yA + 1)

        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        iou = interArea / float(boxAArea + boxBArea - interArea)

        return iou
    
    def getBoxes(self, boxList):
        boxs = np.array(boxList)
        # get positive boxes
        ROI_t_num = boxs.shape[0]
        label_t = [True] * ROI_t_num

        #generate a mask for negative boxs
        gt_mask = np.zeros((self.imgH, self.imgW))
        for box in boxList:
            bbox = np.ones(( box[3], box[2] ))
            gt_mask[ box[1]:box[1]+box[3], box[0]: box[0]+box[2] ] = bbox

        #get negative boxs
        fboxs = []
        minW = 8
        minH = 8
        while( len(fboxs)<ROI_t_num ):
            x0 = random.randint(0, self.imgW - minW )
            y0 = random.randint(0, self.imgH - minH)
            x1 = random.randint(x0+minW, self.imgW )
            y1 = random.randint(y0+minH, self.imgH)
            
            #check whether the randomly generated box can be a false sample
            falseBox = True
            for box in boxList:
                Iou = self.getIoU([box[0],box[1],box[0]+box[2],box[1]+box[3]],[x0,y0,x1,y1])
                if Iou > 0.6:
                    falseBox = False
                    break
            if falseBox:
                fboxs.append([x0,y0,x1,y1])
           
        ROI_f_num = len(fboxs)
        label_f = [False] * ROI_f_num
        
        #gather all boxes and labels
        rois_np = np.zeros((ROI_t_num+ROI_f_num, 5)).astype(np.single)
        if ROI_t_num > 0:
            rois_np[:ROI_t_num, 1:3] = boxs[:, 0:2]
            rois_np[:ROI_t_num, 3] = boxs[:, 0] + boxs[:, 2]
            rois_np[:ROI_t_num, 4] = boxs[:, 1] + boxs[:, 3]
        if ROI_f_num > 0:
            rois_np[ROI_t_num:, 1:5] = fboxs

        bbox_label = label_t + label_f
        
        return rois_np/4, bbox_label

    def load_batch_train(self, img, boxes):
        x = self.loadRGBimg(self.trainImgsPath, img)
        roi, label = self.getBoxes(boxes)
        #self.visualize_batch(x,roi,label)
        return x, roi, label

    def train(self, resume_epoch=0, fine_tune=True):
        if resume_epoch == 0:
            f_logs = open(self.train_logs_path, 'w')
            f_logs.close()

        else:
            prev_epoch = resume_epoch-1
            params_to_load = pickle.load(open(self.path_save_params + 'params_Unaries_%d.pickle' % prev_epoch))
            self.setParams(params_to_load)
            if fine_tune:
                params_VGG = pickle.load(open(self.path_save_params + 'params_VGG_%d.pickle' % prev_epoch))
                self.mNet.setParams(params_VGG)
                
        with open(MyConfig.unaries_boxlist) as read_file:
            u_boxList = json.load(read_file)
        with open(MyConfig.unaries_imgList) as read_file:
            u_imgList = json.load(read_file)
            
        for epoch in range(resume_epoch, MyConfig.u_epochs):
            costs = []
            for img, boxes in zip(u_imgList, u_boxList):
                print 'Epoch', epoch, ' img=', img, ', num of bbox = ', len(boxes)
                x, rois_np, labels = self.load_batch_train(img, boxes)
                # self.visualize_batch(x,rois_np,labels)

                p_out_train, loss = self.train_func(x, rois_np, labels)
                print 'Loss Unaries', loss
                costs.append(loss)
                # x_out_test = test_func(rgb_theano,rois_np)

            # Save params
            if epoch % 2 == 0:
                params_to_save = self.getParams()
                pickle.dump(params_to_save, open(self.path_save_params + 'params_Unaries_%d.pickle' % epoch, 'wb'))
                if fine_tune:
                    params_VGG = self.mNet.getParams()
                    pickle.dump(params_VGG, open(self.path_save_params + "params_VGG_%d.pickle" % epoch, 'wb'))

            av_cost = np.mean(costs)
            f_logs = open(self.train_logs_path, 'a')
            f_logs.write('%f' % (av_cost) + '\n')
            f_logs.close()

def checkPath(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main():
    unaryNet = unaryNet2()
    checkPath(unaryNet.path_save_params)
    unaryNet.train()

if __name__ =="__main__":
    main()