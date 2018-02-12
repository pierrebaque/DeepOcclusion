from GaussianNet import gaussianNet
from net_functions import *
import os
os.environ["THEANO_FLAGS"] = "device=gpu2, floatX=float32"
from PIL import Image
import pickle
import time
import json

import MyConfig


class gaussian2(gaussianNet):
    def __init__(self):
        gaussianNet.__init__(self)
        self.trainImgsPath = MyConfig.trainImgPath
        self.trainLabelsPath = MyConfig.trainLabelPath
        self.imgList = []
        self.labelList = []

    def checkPath(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def loadImgList(self, dataPath, data_ext):
        files = [f for f in os.listdir(dataPath) if os.path.isfile(dataPath + f)]
        files = [i for i in files if i.endswith('.'+data_ext)]
        self.imgList = files

    def generateLabelList(self, imgList, img_ext):
        labelList = [MyConfig.labelName%f[:-(len(img_ext)+1)] for f in imgList]
        self.labelList = labelList

    def loadjsonData(self, dataPath, jsonfile):
        with open(dataPath + jsonfile) as read_file:
            data = json.load(read_file)
        return np.array(data)

    def load_batch(self, local_training_set_indices, train=True, from_generated=False):
        batch_size = len(local_training_set_indices)

        rgb_list = []
        labels_list = []
        for idx in local_training_set_indices:

            # rgb = np.asarray(Image.open(self.imgs[fid]))[:, :, 0:3]
            rgb = np.asarray( Image.open(self.trainImgsPath+self.imgList[idx]) )[:, :, 0:3]
            H, W = np.shape(rgb)[0:2]
            rgb_theano = rgb.transpose((2, 0, 1))
            rgb_theano = rgb_theano.reshape((1, 3, H, W))
            rgb_list.append(rgb_theano)

            # if train:
            #     labels = np.clip(np.loadtxt(self.labels_path % fid), -1000, 1000)
            CNN_factor = 4
            H_lab, W_lab = H / CNN_factor, W / CNN_factor

            # print labels.shape
            labels = self.loadjsonData(self.trainLabelsPath, self.labelList[idx])
            labels = labels.reshape(H_lab, W_lab, 5)
            labels = labels.transpose((2, 0, 1))
            labels = labels.reshape(1, 5, H_lab, W_lab)
            labels_list.append(labels)

        x_in = np.concatenate(rgb_list, axis=0)
        y_in = np.concatenate(labels_list, axis=0)

        return x_in, y_in
    
    def optimize_gaussians_online(self,all_indices,gaussian_minibatch_size = 4,from_generated = False):
        number_of_minibatches = len(all_indices)/gaussian_minibatch_size

        self.go_zero_sum_func()

        #we are computing the sums that will be used to update the gaussians
        for b in range(0,number_of_minibatches):

            local_indices =  all_indices[b*gaussian_minibatch_size:(b+1)*gaussian_minibatch_size]
            #print 'gaussian minibatch',b
            x_in,y_in = self.load_batch(local_indices,train = True,from_generated = from_generated)

            self.train_sums_func(x_in,y_in)

        self.train_gaussians(x_in[0:2], y_in[0:2])
        return

    def train_parts(self, em_it, training_round = 0):

        batch_size = 4
        generated_training_set_size = len(self.imgList)
        epoch_set_size = min(generated_training_set_size,400)
        update_gaussian_every_batch_iters = min(generated_training_set_size,100)
        gaussian_fitting_size = min(generated_training_set_size,500)
        train_logs_path = MyConfig.log_path
        self.checkPath(train_logs_path)
        params_bg = pickle.load(open(MyConfig.bgParams_path))
        self.mBGsub.setParams(params_bg)

        if training_round ==0:
            log_filename = 'train_%d.txt' % em_it
            self.checkPath(train_logs_path)
            f_logs = open(train_logs_path+log_filename, 'w')
            f_logs.close()
            if em_it == 0:
                #initilize params
                init_gaussian_params = init_all_gaussian_params(self.n_leaves)
                load_gaussian_params_fromshared(self.params_gaussian, init_gaussian_params)
                random_reg_params = self.regression_net.get_random_regression_params()
                self.regression_net.load_regression_params(random_reg_params)
                print 'finished initialization'
            else:
                #load parameters from previous em iteration
                params_regression = pickle.load(
                    open(MyConfig.net_params_path + 'EM%d_params_regression.pickle' % (em_it - 1)))
                self.regression_net.load_regression_params(params_regression)
                gaussian_params = pickle.load(
                    open(MyConfig.net_params_path + 'EM%d_params_gaussian.pickle' % (em_it - 1)))
                load_gaussian_params(self.params_gaussian, gaussian_params)
        else:
            #load parameters from previous training iteration
            params_regression = pickle.load(open(MyConfig.net_params_path
                                                 + 'params_regression_%d.pickle' % (training_round-1)))
            self.regression_net.load_regression_params(params_regression)
            gaussian_params = pickle.load(open(MyConfig.net_params_path
                                               + 'params_gaussian_%d.pickle' % (training_round-1)))
            load_gaussian_params(self.params_gaussian, gaussian_params)


        #learning regression parameters
        for iterIdx in range(training_round, MyConfig.epochs):
            print'epoch %d' %(iterIdx)
            generated_training_set_order = np.random.permutation(np.arange(0, generated_training_set_size))

            # Train
            av_cost = 0
            for batch in range(0,epoch_set_size/batch_size):

                local_training_set_indices = generated_training_set_order[batch*batch_size:(batch+1)*batch_size]
                x_in,y_in = self.load_batch(local_training_set_indices,train = True,from_generated = True)
                t_start = time.time()
                cost = self.train_decision_func(x_in,y_in)[0]
                t_end = time.time()
                print 'regression training time %f' % (t_end - t_start)
                av_cost+=cost

                #Optimise gaussian
                local_training_set_indices = generated_training_set_order[batch*batch_size:
                                                                          batch*batch_size+gaussian_fitting_size]

                if batch % update_gaussian_every_batch_iters == update_gaussian_every_batch_iters - 1:
                    self.optimize_gaussians_online(local_training_set_indices,from_generated=True)

            av_cost = av_cost / (epoch_set_size/batch_size)
            print 'av_cost = %f' %av_cost
            f_logs = open(train_logs_path+log_filename, 'a')
            f_logs.write('%f' % (av_cost) + '\n')
            f_logs.close()

            # Save Params after each two iterations
            if iterIdx % 2 == 0:
                params_regression = self.regression_net.save_regression_params() # load regression params to the var on left side of =
                gaussian_params = save_gaussian_params(self.params_gaussian)
                self.checkPath(MyConfig.net_params_path)
                with open(MyConfig.net_params_path + 'params_regression%d_%d.pickle'%(em_it, iterIdx), 'wb') as a:
                    pickle.dump(params_regression, a)
                with open(MyConfig.net_params_path + 'params_gaussian%d_%d.pickle'%(em_it, iterIdx), 'wb') as a:
                    pickle.dump(gaussian_params, a)

                # Run small test
                #self.run_test(em_it, reload_params=False, name='test_em%d_it%d_' % (em_it, iterIdx))

        params_regression = self.regression_net.save_regression_params()
        gaussian_params = save_gaussian_params(self.params_gaussian)
        self.checkPath(MyConfig.net_params_path)
        with open(MyConfig.net_params_path  + 'EM%d_params_regression.pickle'%em_it,'wb') as a:
            pickle.dump(params_regression,a)
        with open(MyConfig.net_params_path  + 'EM%d_params_gaussian.pickle'%em_it,'wb') as a:
            pickle.dump(gaussian_params,a)


def main(reloadData = False):
    gaussianModel = gaussian2()
    gaussianModel.loadImgList( gaussianModel.trainImgsPath, MyConfig.imgExt)
    gaussianModel.generateLabelList( gaussianModel.imgList, MyConfig.imgExt)
    print 'start training'
    for em_iter in range(MyConfig.iterations):
        gaussianModel.train_parts(em_iter)

if __name__ =="__main__":
    main()
