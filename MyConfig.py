#Images for training
datasetPath = './sample_data/'
trainImgPath = datasetPath + 'trainImgs/' #where the training images are

#================ for training GMM and regression mode ==================
trainLabelPath = datasetPath + 'trainLabels/' # path to corresponding label for training GMM
imgExt = 'png'
labelName = 'labels%s.json' #format of label name, %s is the training image's file name before imgExt
iterations = 1
epochs = 80

log_path = './Potentials/Parts/log_GMM/'
bgParams_path = './VGG/models/paramsBG.pickle' #bg parameters
net_params_path = './Potentials/Parts/log_GMM/net/' # path to save parameters

#for testing GMM
testImgPath = datasetPath + 'trainImgs/'



#================ for training UnariesNet ================
jsonFile ='./sample_data/json_file/' #path to the list of images and corresponding bounding box
unaries_boxlist = jsonFile+'boxList.json' # the list of ground truth bounding box with format [x0,y0,W,H]
unaries_imgList = jsonFile+'imgList.json' # the corresponding list of image name
unaries_params_path = './Unaries/trainedModels/'#'../DeepOcclusion/Unaries/trainedModels_noCrowd/'
unaries_train_log = './Unaries/train_unaries.txt'
u_epochs = 80

#for testing unariesNet
unaryTestPath = datasetPath + 'trainImgs/'
unary_storedParam = unaries_params_path + 'params_Unaries_59.pickle'
refinedVGG_storedParam = unaries_params_path + 'params_VGG_59.pickle'
test_u_boxList = jsonFile+'boxList.json'
test_u_imgList = jsonFile+'imgList.json'
