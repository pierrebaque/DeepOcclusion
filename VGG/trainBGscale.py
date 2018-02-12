import numpy as np
# import matplotlib.pyplot as plt
from PIL import Image
import sys
import pickle

# import CNNet
import os
os.environ["THEANO_FLAGS"] = "device=gpu2,floatX=float32"
import VGGNet
import Optimisation
import BGsubstract

import theano
import theano.tensor as T
from theano.tensor.nnet.conv import conv2d
# from theano.tensor.signal.downsample import max_pool_2d
from theano.tensor.shared_randomstreams import RandomStreams
srng = RandomStreams()

import MyConfig

#define Theano graph
x = T.tensor4('x')#RGB images
y = T.tensor4('y')#labels (2 channels in dim 1 otherwise same shape as x)

#pass Alex Net on it
# mNet = CNNet.CNNet(x)
mNet = VGGNet.VGG(x)
x_activ = mNet.activation_volume
y_activ = mNet.reshapeInputImageToActivationVol(y)

#pass the activation volume and desired output to Background foreground NN
mBGsub = BGsubstract.BGsubstract(x_activ,False)
#mBGsub = BGsubstract.BGsubstract(x_activ)#load pretrained and continue training
#get what we neet to define loss
p_fb_flat_train = mBGsub.p_fb_flat_train
p_fb_flat_test = mBGsub.p_fb_flat_test
params = mBGsub.params
#get what we neet to check test
p_fb = mBGsub.p_fb

#define cost function to optimize
y_activ_flat = y_activ.dimshuffle(0,2,3,1).reshape((y_activ.shape[0]*y_activ.shape[2]*y_activ.shape[3],y_activ.shape[1]))

#take on all image
#cost = T.mean(T.nnet.categorical_crossentropy(p_fb, y_activ_flat))
#or take only a few pixels in image
nbRandomSamples = p_fb_flat_train.shape[0]
permutations_samples = srng.permutation(n=p_fb_flat_train.shape[0], size=(1,))[0]#create a vector of size (1,shape)
cost_train = T.mean(T.nnet.categorical_crossentropy(p_fb_flat_train[permutations_samples[0:nbRandomSamples]], y_activ_flat[permutations_samples[0:nbRandomSamples]]))
cost_pred = T.mean(T.nnet.categorical_crossentropy(p_fb_flat_test[permutations_samples[0:nbRandomSamples]], y_activ_flat[permutations_samples[0:nbRandomSamples]]))

#updates = Optimisation.momentum(cost, params, learning_rate=0.0001, momentum=0.9)
updates = Optimisation.adam(cost_train, params, learn_rate = 0.0005)

#reshape p_fb for printing
downsampled_x_rgb = x_activ[:,0:3]

# compile theano functions
train = theano.function([x, y], cost_train, updates=updates)
getCost = theano.function([x, y], cost_pred)
getRGBdownsampled = theano.function([x], downsampled_x_rgb)
predict = theano.function([x], p_fb)

#create function to get training and testing images
def getTrainingDatafromSet(filename,scaleRatio):
    # if id_image<74 :
    #     filename = "../../CaffeSeg/data/PNGImagesCrop/FudanPed%05d.png"%(id_image+1)
    #     #filename = "../../CaffeSeg/data/PNGImages/FudanPed%05d.png"%(id_image+1)
    # else:
    #     filename = "../../CaffeSeg/data/PNGImagesCrop/PennPed%05d.png"%(id_image-73)
    #     #filename = "../../CaffeSeg/data/PNGImages/PennPed%05d.png"%(id_image-73)
    #read file
    img_pil = Image.open(filename)
    img_pil = img_pil.resize((int(scaleRatio*img_pil.size[0]),int(scaleRatio*img_pil.size[1])), Image.ANTIALIAS)

    img = np.asarray(img_pil, dtype=theano.config.floatX)
    img = img[:,:,0:3]

    #create tensor
    return img.transpose(2,0,1).reshape((1,3,img.shape[0],img.shape[1]))
    

def getTrainingLabelfromSet(filenameMsk,scaleRatio):
    # if id_image<74 :
    #     filenameMsk = "../../CaffeSeg/data/PedMasksCrop/FudanPed%05d_mask.png"%(id_image+1)
    #     #filenameMsk = "../../CaffeSeg/data/PedMasks/FudanPed%05d_mask.png"%(id_image+1)
    # else:
    #     filenameMsk = "../../CaffeSeg/data/PedMasksCrop/PennPed%05d_mask.png"%(id_image-73)
    #     #filenameMsk = "../../CaffeSeg/data/PedMasks/PennPed%05d_mask.png"%(id_image-73)

    #read file
    imgMask_pil = Image.open(filenameMsk)
    imgMask_pil = imgMask_pil.resize((int(scaleRatio*imgMask_pil.size[0]),int(scaleRatio*imgMask_pil.size[1])), Image.ANTIALIAS)

    imgMask = np.asarray(imgMask_pil, dtype=theano.config.floatX)
    
    #have 0 for bg and 1 or + for people => need to set everything to 1
    imgMask = imgMask[:, :, 0]
    imgMask[imgMask>0]=1
    

    #create tensor
    tensorMask = np.empty((1,2,imgMask.shape[0],imgMask.shape[1]),dtype=theano.config.floatX)
    tensorMask[0,0]=imgMask #let s put the foreground in first channel
    tensorMask[0,1]=1.-imgMask #let s put the background in second channel
    return tensorMask

    
#function to visualize what we are learning
def savePredToFile(pred,i):
    #here prediction is given as a tensor, we just want to plot channel 0 which is proba of foreground
    im = Image.fromarray(np.uint8(pred[0,0]*255.))
    checkPath('./logs/')
    im.save("./logs/pred_it%d.png"%i)

def saveRGBtensorToFile(x):
    #here prediction is given as a tensor, we just want to plot channel 0 which is proba of foreground
    im = Image.fromarray(np.uint8(x[0].transpose(1,2,0)*255.))
    checkPath('./logs/')
    im.save("./logs/im_c03.png")
    
def checkPath(outpath):
    if not os.path.exists(outpath):
        os.makedirs(outpath)
        
def loadImgList(dataPath, data_ext):
    files = [f for f in os.listdir(dataPath) if os.path.isfile(dataPath + f)]
    files = [i for i in files if i.endswith('.'+data_ext)]
    return files



#load pretrained parameters
#mBGsub.setParams(pickle.load(open("./models/paramsBG.pickle","rb")))

print "Training..."
iteration = 0
cross_pred = []
callBackEveryN = 1000

imgPath = MyConfig.trainImgPath
labelPath = MyConfig.trainMaskPath
training_images = loadImgList(imgPath, MyConfig.fileExt)
nb_training_images = len(training_images)
print 'number of training images = ', nb_training_images

# load testing data
nb_epoque = 5
id_test_image = 3 #randomly select one
x_test = getTrainingDatafromSet(imgPath + training_images[id_test_image],1)
y_test = getTrainingLabelfromSet(labelPath + training_images[id_test_image],1)

#get downsampled rgb x test
x_test_act_rgb = getRGBdownsampled(x_test)
saveRGBtensorToFile(x_test_act_rgb)

checkPath(MyConfig.outputPath)
for e in np.arange(nb_epoque):
    print "Epoch %d..."%e
    #for id_img in np.arange(1,nb_training_images):
    permutation_img = np.random.permutation(np.arange(0,nb_training_images))
    for id_img in permutation_img:
        #print "Load training data image %d..."%id_img
        imgName = training_images[id_img]

        # scaleRatio = np.random.uniform(low=0.8, high=1.4)
        scaleRatio = 1.0
        print 'epo%d'%e, ', imgName=', imgName
        x_train = getTrainingDatafromSet(imgPath+imgName,scaleRatio)
        y_train = getTrainingLabelfromSet(labelPath+imgName,scaleRatio)

        for i in range(10):
            train(x_train, y_train)

        if iteration % callBackEveryN == 0:
            costTest = getCost(x_test, y_test)
            cross_pred.append(costTest)
            print costTest

            prediction = predict(x_test)
            savePredToFile(prediction,iteration/callBackEveryN)

        iteration+=1
    pickle.dump(mBGsub.getParams(),open(MyConfig.outputPath+"paramsBG_e%d.pickle"%e,'wb'))

#save model parameters
pickle.dump(mBGsub.getParams(),open(MyConfig.outputPath+MyConfig.paramFileName+'.pickle','wb'))









