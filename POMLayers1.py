import matplotlib.pyplot as plt
import theano
import theano.tensor as T
import numpy as np
import os

from joblib import Parallel, delayed
import multiprocessing
import Config


'''
Need to be loaded with a room attached using :
import POMLayers
POMLayers.room = room
'''
prior_factor_shift = 1000

class pomLayer:
    
    def __init__(self):
        self.unaries_path = Config.unaries_path
        #Shared variables
        #Higher order
        alphas_np = np.ones(room.n_parts,dtype='float32')
        alphas_np[0:room.n_parts-1] = 1.8
        alphas_np[-1] = 0.6
        alphas = theano.shared(alphas_np,target = 'cpu')

        #Unaries
        priors_factor = theano.shared(np.asarray(np.log(0.001)*160.0,dtype='float32'),target = 'cpu')


        #Constant 
        theano_templates = T.itensor3('templates')
        theano_indices = T.ivector('templates')

        N_vars_theano = theano_templates.shape[1]
        Q_in = T.fvector('Q_in')
        priors_in = T.vector('prior',dtype= 'float32')
        my_d4 = T.TensorType('float32', (False,False,False,False))
        Img = my_d4('Image')
        Temp= T.fscalar('Temperature')
        step= T.fscalar('Step')

        nb_mf_iters =T.iscalar('nb_mf_iter')
        clampings_theano = T.matrix('clampings')
        #Learning variables
        HungarianLabels = T.fvector('HungarianLabels')
        HungarianMasks = T.fvector('HungarianMasks')

        #Multiply prior by factor
        priors = priors_factor*priors_in

        #Bounding-Box shift
        BB_shift = T.alloc(0,room.n_parts*room.n_cams,N_vars_theano,4)
        BB_shift = T.cast(BB_shift, 'int32')

        seq = T.arange(nb_mf_iters)
        #SAME THING
        #With shift
        Q_s_shift, scan_updates = theano.scan(fn=pomLayer.inference_step_shift,
                                                outputs_info=[Q_in,BB_shift],
                                                sequences=[seq],
                                                non_sequences=[priors,theano_templates,
                                                               theano_indices,Img,Temp,clampings_theano,alphas,step])

        #Without  shift
        Q_s, scan_updates = theano.scan(fn=pomLayer.inference_step,
                                                outputs_info=[Q_in],
                                                sequences=[seq],
                                                non_sequences=[priors,theano_templates,
                                                               theano_indices,Img,Temp,clampings_theano,alphas,step])


        #function
        self.infer_function_shift = theano.function(inputs=[Img,Temp,step,Q_in,priors_in,
                                                       nb_mf_iters,clampings_theano,theano_templates,theano_indices],
                                               outputs=[Q_s_shift[0][-2:-1],Q_s_shift[1][-2:-1]],
                                               updates=[], allow_input_downcast=True,on_unused_input='warn')
        
        self.infer_function = theano.function(inputs=[Img,Temp,step,Q_in,priors_in,
                                                 nb_mf_iters,clampings_theano,theano_templates,theano_indices], outputs=Q_s,
                                         updates=[], allow_input_downcast=True,on_unused_input='warn')
        #computing partition func etc...

        logZ = pomLayer.compute_logZ(Q_in,priors,theano_templates,theano_indices,Img,Temp,alphas,step)

        self.logZ_function = theano.function(inputs=[Img,Temp,step,Q_in,priors_in,theano_templates,theano_indices],
                                        outputs=[logZ], updates=[], allow_input_downcast=True,on_unused_input='warn')
        
        # test_out = inference_step_shift(0,Q_in,BB_shift,priors,theano_templates,theano_indices,Img,Temp,clampings_theano,alphas,step)
        # test_function = theano.function(inputs=[Img,Temp,step,Q_in,priors_in,nb_mf_iters,clampings_theano,theano_templates,theano_indices], outputs=test_out, updates=[], allow_input_downcast=True,on_unused_input='warn')#,profile= True)
        
        
        #Define access to shared variables
        self.alphas = alphas
        self.priors_factor = priors_factor

    #THEANO FUNCTIONS USED IN POM
    @staticmethod
    def compute_aux_image(chan,logQ_abs,theano_templates,H,W):
        '''
        Sets value contained in logQ_abs at each corner of the rectangles contained in theano_templates for channel chan.
        This is used before applying computing the integral images.
        '''
        aux = T.alloc(0.0,H,W)
        aux = T.cast(aux,'float32')
        aux = T.inc_subtensor(aux[theano_templates[chan,:,2],theano_templates[chan,:,3] ],logQ_abs)
        aux = T.inc_subtensor(aux[theano_templates[chan,:,2],theano_templates[chan,:,1] ],-1*logQ_abs)
        aux = T.inc_subtensor(aux[theano_templates[chan,:,0],theano_templates[chan,:,3] ],-1*logQ_abs)
        aux = T.inc_subtensor(aux[theano_templates[chan,:,0],theano_templates[chan,:,1] ],logQ_abs)

        return aux

    @staticmethod
    def compute_av_image(Q_current,Img,theano_templates,epsilon_prob = 1e-7):

        '''
        Input : Current probabilities of presence on ground, Images and templates with rectangle coordinates.
        Ouptut : Average probability of absence at each pixel on each channel.
        '''
        logQ_abs = T.log(T.clip(1 - Q_current,epsilon_prob,1-epsilon_prob*5000))
        H,W = room.H, room.W
        n_channels = room.n_cams*room.n_parts
        result, updates = theano.scan(fn = pomLayer.compute_aux_image,
               sequences = theano.tensor.arange(n_channels),
               non_sequences = [logQ_abs,theano_templates,H,W])
        aux_final = T.stack(result)
        aux_final = T.reshape(aux_final,(1,n_channels,H,W))
        aux_vert = T.extra_ops.cumsum(aux_final,axis = 2)
        aux_hor = aux_vert.cumsum(axis = 3)
        #T.clip(aux_hor,-1e20,1e20)
        Av_abs =T.exp(aux_hor)

        return Av_abs

    @staticmethod
    def compute_logZ(Q_current,priors,theano_templates,theano_indices,Img,Temp,alphas,step,epsilon_prob = 1e-7):
        '''
        Input : 
        - Probability of presence on the ground plane.
        - Prior probabilities of presence.
        - Rectangle templates.
        - Indices of templates inside the complete pool (used for pairwise).
        - Images with parts
        - Temperature used
        - Coefficient given to each part with Full background last
        - Inference Step (not used)
        Output :
        Compute the log-partition function of the POM CRF (i.e. -KL + Const).
        '''

        #higher order energy
        n_channels = room.n_cams*room.n_parts
        Av_abs = pomLayer.compute_av_image(Q_current, Img,theano_templates)
        #Reshape the alphas
        alphas_long = T.concatenate([alphas for i in range(len(room.cameras_list))])
        alphas_long = alphas_long.reshape((1,n_channels,1,1))

        pixels_wise_dist = alphas_long*Img*Av_abs 
        pixels_wise_dist_black = alphas[-1]*Img*Av_abs + (1-Img)*(1-Av_abs) 
        pixels_wise_dist = T.set_subtensor(pixels_wise_dist[:,room.n_parts-1::room.n_parts,:,:],pixels_wise_dist_black[:,room.n_parts-1::room.n_parts,:,:])
        pixels_wise_dist_sum = T.sum(pixels_wise_dist,axis =(0,1,2,3))
        #pairwise energy
        E0_E1 = pomLayer.compute_pairwise(Q_current,theano_indices)
        pairwise_E = -T.dot(E0_E1,Q_current)
        #unaries
        unaries = -1*T.sum(Q_current*priors,axis =0)

        entropy = -1*T.sum(Q_current*T.log(Q_current)+ (1-Q_current)*T.log(1-Q_current),axis =0)

        return -(pixels_wise_dist_sum + unaries + pairwise_E)/Temp + entropy

    #Batch size is assumed to be 1 always
    @staticmethod
    def inference_step(i_iter,Q_current,priors,theano_templates,theano_indices
                       ,Img,Temp,clampings,alphas,step = 0.2,epsilon_prob = 1e-7):
        '''
        Input : 
        - Current Probability of presence on the ground plane.
        - Prior probabilities of presence.
        - Rectangle templates.
        - Indices of templates inside the complete pool (used for pairwise).
        - Images with parts
        - Temperature used
        - Coefficient given to each part with Full background last
        - Inference Step 
        Output :
        New probability after inference step.


        '''

        #Compute average image using integral image
        H,W = room.H, room.W
        n_channels = room.n_cams*room.n_parts
        n_vars = T.shape(theano_templates)[1]
        Av_abs = pomLayer.compute_av_image(Q_current, Img,theano_templates)

        #Average difference image
        alphas_long = T.concatenate([alphas for i in range(len(room.cameras_list))])
        alphas_long = alphas_long.reshape((1,n_channels,1,1))
        Av_diff = (alphas_long*Img)*Av_abs
        Av_diff_black = ((alphas[-1]+1)*Img-1)*Av_abs
        Av_diff = T.set_subtensor(Av_diff[:,room.n_parts-1::room.n_parts,:,:],Av_diff_black[:,room.n_parts-1::room.n_parts,:,:])
        #Compute integral image wich is the sum of Av_diff for the rectangle on top left of each pixel
        aux_vert = Av_diff.cumsum(axis = 2) #- Av_diff
        Integral_diff = aux_vert.cumsum(axis = 3) #- aux_vert
        #Now, for each variable, extract the integral in the rectangle
        variable_integrals = T.zeros_like(Q_current,dtype= 'float32')

        result, updates = theano.scan(fn=lambda chan, variable_integrals,Integral_diff: variable_integrals + Integral_diff[0,chan,theano_templates[chan,:,0],theano_templates[chan,:,1]] + Integral_diff[0,chan,theano_templates[chan,:,2],theano_templates[chan,:,3]] - Integral_diff[0,chan,theano_templates[chan,:,0],theano_templates[chan,:,3]] - Integral_diff[0,chan,theano_templates[chan,:,2],theano_templates[chan,:,1]],
                                      outputs_info = variable_integrals,
                                      sequences = theano.tensor.arange(n_channels),
                                      non_sequences = Integral_diff)

        final_integrals = result[-1]
        d0_d1 = final_integrals*(1/(1-Q_current))

        #Add pairwise terms
        E0_E1 = d0_d1 + pomLayer.compute_pairwise(Q_current,theano_indices)

        #perform update
        new_Q = T.nnet.sigmoid(step*(E0_E1 + priors)/Temp+(1-step)*T.log(Q_current/(1-Q_current)))

        return T.clip(new_Q,epsilon_prob,1.0 -epsilon_prob)

    @staticmethod
    def compute_pairwise(Q_current,theano_indices,epsilon_prob = 1e-7):
        '''
        Input : 
        - Current Probability of presence on the ground plane.
        - Indices of templates inside the complete pool (used for pairwise).
        Output :
        Pairwise terms used in Mean-Field update.
        '''
        #Pairwise
        pair_radius = Config.exclusion_rad
        np_kernel = np.ones((1,1,2*pair_radius+1,2*pair_radius+1), dtype='float32')*5000
        np_kernel[0,0,pair_radius,pair_radius] = 0
        pairwise_kernel = theano.shared(np_kernel)
        ####

        H_grid,W_grid = room.H_grid,room.W_grid
        Q_map = T.alloc(epsilon_prob,1,1,H_grid,W_grid)
        T.cast(Q_map,'float32')
        #To be implemented
        Q_map = T.set_subtensor(Q_map[0,0,theano_indices//W_grid,theano_indices%W_grid],Q_current)
        E_conv = T.nnet.conv2d(Q_map, pairwise_kernel, border_mode='half')

        E0_E1 = E_conv[0,0,theano_indices[:]//W_grid,theano_indices[:]%W_grid]
        return -1*E0_E1


    # Functions used for POM with Shift
    @staticmethod
    def inference_step_shift(i_iter,Q_current,BB_shift_current,priors,theano_templates_prior
                             ,theano_indices,Img,Temp,clampings,alphas,step = 0.2,epsilon_prob = 1e-7):
        '''
        Input : 
        - Current Probability of presence on the ground plane.
        - Prior probabilities of presence.
        - Rectangle templates.
        - Indices of templates inside the complete pool (used for pairwise).
        - Images with parts
        - Temperature used
        - Coefficient given to each part with Full background last
        - Inference Step 
        Output :
        New probability after inference step.


        '''
        #Get usefull variables
        H,W = room.H, room.W
        n_channels = room.n_cams*room.n_parts
        n_vars = T.shape(theano_templates_prior)[1]

        #Replace theano templates by prior + shift
        theano_templates = theano_templates_prior + BB_shift_current
        #Clip to H and W
        theano_templates = T.set_subtensor(theano_templates[:,:,0],T.clip(theano_templates[:,:,0],0,H-1))
        theano_templates = T.set_subtensor(theano_templates[:,:,1],T.clip(theano_templates[:,:,1],0,W-1))
        theano_templates = T.set_subtensor(theano_templates[:,:,2],T.clip(theano_templates[:,:,2],0,H-1))
        theano_templates = T.set_subtensor(theano_templates[:,:,3],T.clip(theano_templates[:,:,3],0,W-1))

        #Compute average image using integral image
        Av_abs = pomLayer.compute_av_image(Q_current, Img,theano_templates)

        #Average difference image
        alphas_long = T.concatenate([alphas for i in range(len(room.cameras_list))])
        alphas_long = alphas_long.reshape((1,n_channels,1,1))
        Av_diff = (alphas_long*Img)*Av_abs
        Av_diff_black = ((alphas[-1]+1)*Img-1)*Av_abs
        Av_diff = T.set_subtensor(Av_diff[:,room.n_parts-1::room.n_parts,:,:],Av_diff_black[:,room.n_parts-1::room.n_parts,:,:])
        #Compute integral image wich is the sum of Av_diff for the rectangle on top left of each pixel
        aux_vert = Av_diff.cumsum(axis = 2) #- Av_diff
        Integral_diff = aux_vert.cumsum(axis = 3) #- aux_vert
        #Now, for each variable, extract the integral in the rectangle
        variable_integrals = T.zeros_like(Q_current,dtype= 'float32')

        result, updates = theano.scan(fn=lambda chan, variable_integrals,Integral_diff: variable_integrals + Integral_diff[0,chan,theano_templates[chan,:,0],theano_templates[chan,:,1]] + Integral_diff[0,chan,theano_templates[chan,:,2],theano_templates[chan,:,3]] - Integral_diff[0,chan,theano_templates[chan,:,0],theano_templates[chan,:,3]] - Integral_diff[0,chan,theano_templates[chan,:,2],theano_templates[chan,:,1]],
                                      outputs_info = variable_integrals,
                                      sequences = theano.tensor.arange(n_channels),
                                      non_sequences = Integral_diff)

        final_integrals = result[-1]
        d0_d1 = final_integrals*(1/(1-Q_current))

        #Add pairwise terms
        E0_E1 = d0_d1 + pomLayer.compute_pairwise(Q_current,theano_indices)

        #perform update
        new_Q = T.nnet.sigmoid(step*(E0_E1 + priors)/Temp+(1-step)*T.log(Q_current/(1-Q_current)))

        #####Compute BB shift    
        # Compute BB Integral
        #Hack     
        Av_diff = Img*1.0*Av_abs[0]
        aux_vert = Av_diff.cumsum(axis = 2) - Av_diff
        Integral_diff = aux_vert.cumsum(axis = 3) - aux_vert
        ##### End Hack

        BB_integral = pomLayer.BB_integral_from_integral_image(Integral_diff,theano_templates,n_channels)

        # Same way, Compute BB Integral X. It is the same thing as integral diff, weighted by the X coordinate
        X_map = T.arange(H)
        X_map = X_map.reshape((H,1)).repeat(W,axis = 1).dimshuffle('x',0,1)
        Av_diff_X = Img*1.0*Av_abs[0]*X_map
        #Compute integral image
        aux_vert_X = Av_diff_X.cumsum(axis = 2) - Av_diff_X
        Integral_diff_X = aux_vert_X.cumsum(axis = 3) - aux_vert_X

        BB_integral_X = pomLayer.BB_integral_from_integral_image(Integral_diff_X,theano_templates,n_channels)

        # Same way, Compute BB Integral Y. It is the same thing as integral diff, weighted by the Y coordinate
        Y_map = T.arange(W)
        Y_map = Y_map.reshape((1,W)).repeat(H,axis = 0).dimshuffle('x',0,1)
        Av_diff_Y = Img*1.0*Av_abs[0]*Y_map
        #Compute integral image
        aux_vert_Y = Av_diff_Y.cumsum(axis = 2) - Av_diff_Y
        Integral_diff_Y = aux_vert_Y.cumsum(axis = 3) - aux_vert_Y

        BB_integral_Y = pomLayer.BB_integral_from_integral_image(Integral_diff_Y,theano_templates,n_channels)


        #Compute shift X , Y
        #Get coordinates of center in prior
        X_mid = (theano_templates_prior[:,:,2] + theano_templates_prior[:,:,0] - 1 )/2.0
        Y_mid = (theano_templates_prior[:,:,3] + theano_templates_prior[:,:,1] - 1 )/2.0
        #XY_area = (theano_templates_prior[:,:,2] - theano_templates_prior[:,:,0])* (theano_templates_prior[:,:,3] - theano_templates_prior[:,:,1]) + 1.0 Not used anymore

        #Now we are ready to compute shifts
        #prior_factor = 90.0
        Q_normalizer =1.0/(1-Q_current).dimshuffle('x',0)

        shift_X = (Q_normalizer*BB_integral_X - Q_normalizer*X_mid*BB_integral)/(Q_normalizer*BB_integral + prior_factor_shift)
        shift_Y = (Q_normalizer*BB_integral_Y - Q_normalizer*Y_mid*BB_integral)/(Q_normalizer*BB_integral + prior_factor_shift)


        #Finalize by puting to the right formalt
        BB_shift_new = T.stack([shift_X,shift_Y,shift_X,shift_Y],axis = 2)
        BB_shift_new = T.cast(T.round(BB_shift_new),'int32')
        #Don't shift the BB corresponding to the full BB
        BB_shift_new = T.set_subtensor(BB_shift_new[room.n_parts-1::room.n_parts,:,:],0)

        return T.clip(new_Q,epsilon_prob,1.0 -epsilon_prob*5000),BB_shift_new
        #return T.clip(new_Q,epsilon_prob,1.0 -epsilon_prob*5000),BB_shift_new,BB_integral,BB_integral_X,theano_templates_prior

    @staticmethod
    def BB_integral_from_integral_image(Integral_diff,theano_templates,n_channels):

        result, updates = theano.scan(fn=lambda chan,Integral_diff:  Integral_diff[0,chan,theano_templates[chan,:,0],theano_templates[chan,:,1]] + Integral_diff[0,chan,theano_templates[chan,:,2],theano_templates[chan,:,3]] - Integral_diff[0,chan,theano_templates[chan,:,0],theano_templates[chan,:,3]] - Integral_diff[0,chan,theano_templates[chan,:,2],theano_templates[chan,:,1]],
                                      outputs_info = [],
                                      sequences = theano.tensor.arange(n_channels),
                                      non_sequences = Integral_diff)

        return result
    

    #METHODS TO RUN POM
    
    def set_POM_params(self,a,alpha_black,prior_factor):

        #print 'Setting a = %f, p = %f'%(a,p)
        alphas_np = np.ones(room.n_parts,dtype='float32')
        alphas_np[0:room.n_parts-1] = a
        alphas_np[-1] = alpha_black
        self.alphas.set_value(alphas_np)

        #Unaries
        self.priors_factor.set_value(np.asarray(np.log(0.001)*prior_factor,dtype='float32'))

    
    def run_POM(self,fid,getZ = False,n_iter_pom = 150,step_0 = 0.005,T_0 = 10.0,useshift = False,use_unaries = True):
        #Params
        thresh =0.2
        epsilon_prob = 1e-7
        initial_q = 0.01
        #####

        templates_array = room.templates_array
        image = room.load_images_stacked(fid)

        indices = templates_array.shape[1]
        indices_reduced,scores = room.get_indices_above(image,threshold= 0.4)

        #set priors
        #load unaries
        if use_unaries:
            unaries_logp = np.load(self.unaries_path%room.img_index_list[fid])
            #process them
            unaries_E = -1*unaries_logp  
            unaries = unaries_E.clip(0.1,2).min(axis = 0)*2.0

            #Doesn't really bring speedup. WHY??
            unaries_reduced = unaries[indices_reduced]
            indices_reduced = indices_reduced[unaries_reduced < 4]

            #Finalize with reduced
            priors_np = unaries[indices_reduced]
        else:
            priors_np = 1.0 + 0.0*templates_array[0,indices_reduced,0]

        templates_array_reduced = templates_array[:,indices_reduced,:]
        N_vars = templates_array_reduced.shape[1]

        #reshape image for theano
        image_reshaped = np.reshape(image,(1,image.shape[0],image.shape[1],image.shape[2]))

        #initialize clampings
        clampings = np.zeros((2,N_vars))
        clampings[0] += epsilon_prob 
        clampings[1] += 1-epsilon_prob 
        clamplist = [clampings]

        #Choose Q initial
        if use_unaries:
            Q_init = np.exp(-1*unaries[indices_reduced])
        else:
            Q_init = np.ones(templates_array_reduced.shape[1])*initial_q
        #Launch inference
        if useshift:
            Q_out, Shift = self.infer_function(image_reshaped,T_0,step_0,Q_init,priors_np,
                                                      n_iter_pom,clamplist[0],templates_array_reduced,indices_reduced)
        else:

            Q_out = self.infer_function(image_reshaped,T_0,step_0,Q_init,priors_np,
                                               n_iter_pom,clamplist[0],templates_array_reduced,indices_reduced)

        Z_out  = []
        if getZ:
            for Q_t in Q_out:
                Z_out.append(self.logZ_function(image_reshaped,T_0,step_0,Q_t[:],priors_np,
                                                       templates_array_reduced,indices_reduced))

        #Plunge Q_out which is defined over reduced templates into complete templates

        Q_out_full =[]
        for Q_t in Q_out:
            Q_t_full = np.zeros(templates_array.shape[1]) + epsilon_prob
            Q_t_full[indices_reduced] = Q_t
            Q_out_full.append(Q_t_full)

        Shift_full =[]
        if useshift:
            for Shift_t in Shift:
                Shift_t_full = np.zeros((room.n_cams*room.n_parts,templates_array.shape[1],4),dtype = 'int32')
                Shift_t_full[:,indices_reduced,:] = Shift_t
                Shift_full.append(Shift_t_full)

        return Q_out_full,Z_out,Shift_full

    
    #OTHER BASELINE METHODS
    def run_NMS(self,fid,room,rad = 7,thresh_p  = 0.8):
    #Load image into tensor

        #set priors
        #load unaries
        unaries_logp = np.load(self.unaries_path%room.img_index_list[fid])
        unaries_logp = unaries_logp.clip(np.log(0.2),1000)
        unaries_logp_max = np.max(unaries_logp,axis = 0)
        #process them
        unaries = unaries_logp_max.reshape((room.H_grid,room.W_grid))

        Q_out = np.zeros(room.H_grid*room.W_grid)
        while unaries.max() > np.log(thresh_p): 
            #print unaries.max()
            flat_max = np.argmax(unaries)
            x,y = flat_max/room.W_grid,flat_max%room.W_grid
            Q_out[flat_max] = np.exp(unaries[x,y])
            unaries[max(0,x - rad) : min(room.H_grid,x + rad),max(0,y - rad) : min(room.W_grid,y + rad)] = -100
            

        return [Q_out]

    def run_NMS_sum(self,fid,room,rad = 7,thresh_p  = 0.3):
    #Load image into tensor

        #set priors
        #load unaries
        unaries_logp = np.load(self.unaries_path%room.img_index_list[fid])
        unaries_logp = unaries_logp.clip(np.log(0.1),1000)
        unaries_logp_sum = np.sum(unaries_logp,axis = 0)
 
        #process them
        unaries = unaries_logp_sum.reshape((room.H_grid,room.W_grid))

        Q_out = np.zeros(room.H_grid*room.W_grid)
        while unaries.max() > np.log(thresh_p)*7: 

            flat_max = np.argmax(unaries)
            x,y = flat_max/room.W_grid,flat_max%room.W_grid
            unaries[max(0,x - rad) : min(room.H_grid,x + rad),max(0,y - rad) : min(room.W_grid,y + rad)] = -100
            Q_out[flat_max] = 1

        return [Q_out]


    def run_RCNNdetector(self,fid,room,rad = 7,thresh = 0.5):
    #Load image into tensor
        #Unaries
        RCNN_path = '../../../RCNN/Faster-RCNN_TF/tools/ETH_out/c%df%08d.npy'

        Q = np.zeros(room.templates_array.shape[1])
        for cam in range(room.n_cams):
            #load dets
            detections = np.load(RCNN_path%(cam,room.img_index_list[fid]))/resize_pom
            dets_bottom_x = detections[:,3]
            dets_bottom_y = (detections[:,0] + detections[:,2])/2

            #load templates
            templates = room.templates_array[room.n_parts*cam + room.n_parts -1]
            print templates.shape
            templates_bottom_x = templates[:,2]
            templates_bottom_y = (templates[:,1] + templates[:,3])/2

            for i in range(detections.shape[0]):
                #print detections[i,-1]
                if detections[i,-1]*resize_pom > thresh:
                    select = np.argmin((templates_bottom_x - dets_bottom_x[i])**2 + (templates_bottom_y - dets_bottom_y[i])**2)
                    #print select
                    Q[select] = 1

        #Run NMS on top
        Q_reshape = Q.reshape((room.H_grid,room.W_grid))
        Q_out = 0*Q
        while Q_reshape.max() ==1 : 
            flat_max = np.argmax(Q_reshape)
            x,y = flat_max/room.W_grid,flat_max%room.W_grid
            Q_reshape[max(0,x - rad) : min(room.H_grid,x + rad),max(0,y - rad) : min(room.W_grid,y + rad)] = 0
            Q_out[flat_max] = 1

        return [Q_out]

    


