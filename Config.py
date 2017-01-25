#Technical configs
n_threads = 30
verbose = True

#Global configurations
img_index_list = range(0,1000,50) #images to use
H_grid, W_grid = 300,120

# Config about CRF Inference
parts_root_folder = './Parts/EM%d/'
n_parts = 9 #Number of classes of classifier. Including foreground class which is the last one
pom_file_path ='./rectangles120x300.pom'
resize_pom = 4 #Difference in ratio between dimensions for POM file and images saved (4 when images come from VGG)
cameras_list = range(7)
n_threads = 37
#gaussian_params_path = parts_root_folder +'gaussian_params.txt'


# Config about POM
Sigma_factor = 3
resize_pom = 4#Difference in ratio between dimensions for POM file and images saved (4 when images come from VGG)
exclusion_rad = 5

# Config about 
bkg_path = './bkg/c%d/%d.png'

#Folder to save POM estimates
POM_out_folder = './POMOutput/EM%d/'
#Folder to save Z samples
Z_sample_folder = './ZSamples/EM%d/'
#Folder to save labels for CNN
labels_folder = './Labels/EM%d/'

#Folder to save test_image
img_logs = './VGG/img_logs/'

#About RGB images

img_path = '/cvlabdata1/cvlab/datasets_people_tracking/ETH/day_2/'

#List of camera names
rgb_name_list = []
rgb_name_list.append(img_path+'cvlab_camera1/images_1/begin/%08d.png')
rgb_name_list.append(img_path+'cvlab_camera2/images_1/begin/%08d.png')
rgb_name_list.append(img_path+'cvlab_camera3/images_1/begin/%08d.png')
rgb_name_list.append(img_path+'cvlab_camera4/images_1/begin/%08d.png')
rgb_name_list.append(img_path+'idiap_camera1/images_1/begin/%08d.png')
rgb_name_list.append(img_path+'idiap_camera2/images_1/begin/%08d.png')
rgb_name_list.append(img_path+'idiap_camera3/images_1/begin/%08d.png')

#About CNN
CNN_factor = 4
n_epochs = 40
logs_path = 'VGG/logs/'
net_params_path = 'VGG/NetParams/'
data_augmentation_proportions = [1,2,2,2,4,4,4,4]
BGrate = 5e-5
Regrate = 5e-4
use_bg_pretrained = True

#Val configurations
val_index_list = range(0,250,50) #images to use for validation
GS_out = './Results/validation/GridSearch/First/' #where to store validation dat files
GS_num_iter = 150 #Number of iterations in validation
do_GS_validation = True #Do grid search for validation

#Parameters for POM that we test
GS_As = [0,1,1.5,2.0] 
GS_Ps = [150,200,250] 
GS_Ablacks = [0,0.8,2.0]

