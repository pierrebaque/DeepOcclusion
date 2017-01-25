#Technical configs
n_threads = 7
verbose = True

#Global configurations
img_index_list = range(25,1000,50) #images to use
H_grid, W_grid = 30,44

# Config about CRF Inference
parts_root_folder = './Parts/EM%d/'
n_parts = 9 #Number of classes of classifier. Including foreground class which is the last one
pom_file_path ='/cvlabdata1/cvlab/datasets_people_tracking/terrace/pom/rectangles_adjusted.pom'
resize_pom = 4 #Difference in ratio between dimensions for POM file and images saved (4 when images come from VGG)

# Config about POM
Sigma_factor = 3
resize_pom = 4#Difference in ratio between dimensions for POM file and images saved (4 when images come from VGG)
exclusion_rad = 2

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

img_path = '/cvlabdata1/cvlab/datasets_people_tracking/terrace/images/'

#List of camera names
rgb_name_list = []
cameras_list = range(4)
rgb_name_list.append(img_path+'cam0/%04d.png')
rgb_name_list.append(img_path+'cam1/%04d.png')
rgb_name_list.append(img_path+'cam2/%04d.png')
rgb_name_list.append(img_path+'cam3/%04d.png')

rgb_name_list = rgb_name_list[0:len(cameras_list)]

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
do_GS_validation = False #Do grid search for validation

#Parameters for POM that we test
GS_As = [0,1,1.5,2.0] 
GS_Ps = [150,200,250] 
GS_Ablacks = [0,0.8,2.0]

