#Images for training
datasetPath = './sample_data/'
trainImgPath = datasetPath + 'trainImgs/' #where the training images are

#================ For bg substract ================
trainMaskPath = datasetPath + 'trainMasks/' #path to corresponding ground truth of foreground with the same file name as training images
fileExt = 'png' #file extension of training data and ground truth


#save parameters of bg substract 
outputPath = './models_test/'
paramFileName = 'paramsBG'

#testing
test_imgPath = datasetPath + 'trainImgs/'
test_labelPath = datasetPath + 'trainMasks/'
test_fileExt = 'png'