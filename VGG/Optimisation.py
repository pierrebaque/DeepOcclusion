import theano
import theano.tensor as T
import numpy as np

from theano.tensor.nnet.conv import conv2d

def momentum(cost, params, learning_rate = 0.0005, momentum=0.9):
    grads = theano.grad(cost, params)
    updates = []
    
    for p, g in zip(params, grads):
        mparam_i = theano.shared(np.zeros(p.get_value().shape, dtype=theano.config.floatX))
        v = momentum * mparam_i - learning_rate * g
        updates.append((mparam_i, v))
        updates.append((p, p + v))

    return updates

def adam(cost, params, momentum1=0.9, momentum2=0.99, eps=1e-6, learn_rate = 0.0005):
    grads = theano.grad(cost, params)
    updates = []

    time = theano.shared(np.asarray(1.0, dtype=theano.config.floatX))
    
    for p, g in zip(params, grads):
        #use window (exponentially decaying) for the gradient norm accumulation
        accu_norm_grad1_tm1 = theano.shared(np.zeros(p.get_value().shape, dtype=theano.config.floatX))
        accu_norm_grad1_t = momentum1 * accu_norm_grad1_tm1 + (1.-momentum1) * g
        updates.append((accu_norm_grad1_tm1, accu_norm_grad1_t))

        accu_norm_grad2_tm1 = theano.shared(np.zeros(p.get_value().shape, dtype=theano.config.floatX))
        accu_norm_grad2_t = momentum2 * accu_norm_grad2_tm1 + (1.-momentum2) * (g**2)
        updates.append((accu_norm_grad2_tm1, accu_norm_grad2_t))

        #bias corrected values
        bc_accu_norm_grad1_t=accu_norm_grad1_t/(1.-momentum1**time)
        bc_accu_norm_grad2_t=accu_norm_grad2_t/(1.-momentum2**time)
        
        #renormalize gradient with respect to norm of historic, element wise (normaly would do that on grad => -g/... instead of v/...)
        update_pt = -learn_rate * bc_accu_norm_grad1_t/(T.sqrt(bc_accu_norm_grad2_t)+eps)
        updates.append((p, p + update_pt))

    updates.append((time,time+1))


    return updates