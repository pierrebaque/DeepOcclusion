import numpy as np

import theano
import theano.tensor as T
from theano.tensor.nnet.conv import conv2d

#define generic functions
def toShared(x):
	return theano.shared(np.asarray(x,dtype=theano.config.floatX))

def listToShared(x):
	result = []
	for xc in x:
		result.append(toShared(xc))
	return result

def cropTensorx(x_, fitler_size3):
	sf3 = (fitler_size3-1)/2
	return x_[:,:, :,sf3: -sf3]

def convSameSizex(x,w):
	return cropTensorx(conv2d(x, w, border_mode='full'),w.shape[3])

def LRN(x, local_size = 5, alpha= 0.0001,beta= 0.75):
	#want to do convolution but along the channel with zero padding
	#=> need to change dimension to put channels in
	x2 = x**2 # dim = nbSample,channels,h,w
	x2_t = x2.dimshuffle((0,2,3,1))# dim = nbSample,h,w,channels

	#want to convol filter of ones along channels
	#filter must have dim : h,h,1,local_size and filter = ones for h1=h2
	#=> create an identity matrix hxh, repeat it on third dimension and we got it.
	h = x2.shape[2]
	idh = theano.tensor.eye(h,h,0)
	filterw=theano.tensor.repeat(idh,local_size).reshape((h,h,1,local_size))

	#apply filter
	x2tf = convSameSizex(x2_t,filterw)
	x2f = x2tf.dimshuffle((0,3,1,2))

	return x/((1+alpha*x2f/local_size)**beta)

def cropTensor(x_, fitler_size):
            return x_[:,:, (fitler_size-1)/2: -(fitler_size-1)/2,(fitler_size-1)/2: -(fitler_size-1)/2]

def convSameSizeS(x,w,ss):
	return cropTensor(conv2d(x, w, border_mode='full',subsample = ss),w.shape[3])
def convSameSize(x,w):
	return cropTensor(conv2d(x, w, border_mode='full'),w.shape[3])

#divide the number of input channels in groups
#each filter can only take into account one group
def groupConv2d(x,w,ss = (1,1) ,g = 2):
	outToConcatenate = []

	nb_channel_in = x.shape[1]
	nb_channel_in_per_group = nb_channel_in/g
	nb_channel_out = w.shape[0]
	nb_channel_out_per_group = nb_channel_out/g

	#out_group0 = convSameSize(x[:,0:48], w[0:128])[:,:,::ss[0],::ss[1]]
	#out_group1 = convSameSize(x[:,48:96], w[128:256])[:,:,::ss[0],::ss[1]]
	#outToConcatenate.append(out_group0)
	#outToConcatenate.append(out_group1)
		
	for gc in range(g):
		index_channel_in = gc * nb_channel_in_per_group
		index_channel_out = gc * nb_channel_out_per_group
		#out_group = convSameSizeS(x[:,index_channel_in:index_channel_in+nb_channel_in_per_group], w[index_channel_out:index_channel_out+nb_channel_out_per_group], ss)
		out_group = convSameSize(x[:,index_channel_in:index_channel_in+nb_channel_in_per_group], w[index_channel_out:index_channel_out+nb_channel_out_per_group])[:,:,::ss[0],::ss[1]]
		outToConcatenate.append(out_group)

	return T.concatenate(outToConcatenate,axis = 1)

def groupDeconv2D(x,w,g=2):
	if g>2:
		print "groupDeconv2D not implemented for g>2"
	outToConcatenate = []

	nb_channel_in = x.shape[1]
	nb_channel_in_per_group = nb_channel_in/g

	out1 = convSameSize(x[:,0:nb_channel_in_per_group],w[:,0:nb_channel_in_per_group])
	out2 = convSameSize(x[:,nb_channel_in_per_group:],w[:,nb_channel_in_per_group:])

	outToConcatenate.append(out1)
	outToConcatenate.append(out2)

	return T.concatenate(outToConcatenate,axis = 1)



