# -*- coding: utf-8 -*-
"""
Base layer for the implementation of the layers comprising blocks with competing units as 
described in the paper 

The current file contains implementations for feedforward and convolutional layers.

@author: Konstantinos P. Panousis, Dept. of Informatics and Telecommunications, 
         National and Kapodistrian University of Athens, Greece
"""

import tensorflow as tf
from tensorflow.contrib.distributions import  Bernoulli, OneHotCategorical

from utils.distributions import normal_kl, bin_concrete_kl, concrete_kl, kumaraswamy_kl
from utils.distributions import kumaraswamy_sample, bin_concrete_sample, concrete_sample

def variable_on_cpu(name, shape, initializer, dtype=tf.float32, constraint= None, trainable=True):
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer, constraint=constraint, dtype=dtype, trainable=trainable)
    return var

# =============================================================================
# SB LWTA Layer
# =============================================================================
def SB_Layer(inp, K,U, S=128, bias=True, train = True, reuse=False, sbp=False, 
             temp_bern=0.67, temp_cat=0.67, activation='none', name='SB_layer'):
    """
    Dense layer for the SB-LWTA model
    
    Parameters:
        inp: 2d tensor 
            The input to the current layer.
        K: int
            The number of blocks for the current layer
        U: int
            The number of units in each block.
        bias: boolean
            Flag denoting the use of bias. Default False.
        train: boolean
            Flag to alternate between train or test branches.
        reuse: boolean
            Flag to reuse or not the variables of the layer.
        sbp: boolean
            Flag to enable or disable the stick breaking process
        temp_bern: float
            The temperature for the bernoulli relaxation
        temp_cat: float
            The temperature for the categorical relaxation
        activation: String
            Select the activation function for the current layer.
        name: str
            The name of the current layer.
            
    Returns:
        out: 2d tensor
            The output of local competition in the layer after masking with a sample of the IBP,
            application of the linear operation and the addition of bias (if bias==True).
        mW: 2d tensor
            The mean of the weights. Used to load values when calling the compression script
        masked_mw: 2d tensor
            The mean of the weights of the layer, masked with a sample from the IBP.
            Used for calculating the compression ability of the implementation.
        masked_sw: 2d tensor
            The variance of the weights of the layer, masked with a sample from the IBP.
            Used for calculating the compression ability of the implementation.
        activations: 2d tensor
            The activations for the current batch. Used for plotting the probability of activations.
    """
    #cutoff threshold
    tau = 1e-2
    
    name = name+'_'+activation
    with tf.variable_scope(name, reuse = reuse):
    
        # mean and variance of the weights
        mW = variable_on_cpu('mW', [inp.get_shape()[1],K*U], initializer =tf.contrib.layers.xavier_initializer(),
                                    dtype= tf.float32)
        
        
        sW = variable_on_cpu('sW', [inp.get_shape()[1],K*U], 
                                    initializer = tf.initializers.random_normal(-5.,1e-2),
                                    constraint = lambda x: tf.clip_by_value(x, -7.,x),
                                    dtype= tf.float32)
        sW = tf.nn.softplus(sW)
        
        # variables and construction for the stick breaking process (if active)
        if sbp:
            
            # posterior concentration variables for the IBP
            conc1 = variable_on_cpu('sb_t_u_1', [K],
                                      initializer = tf.constant_initializer(K),
                                      constraint=lambda x: tf.clip_by_value(x, -6., x),
                                      dtype = tf.float32)
            
            conc0 = variable_on_cpu('sb_t_u_2', [K],
                                      initializer = tf.constant_initializer(2.),
                                      constraint=lambda x: tf.clip_by_value(x, -6., x),
                                      dtype = tf.float32)
            conc1 = tf.nn.softplus(conc1)
            conc0 = tf.nn.softplus(conc0)
            
            # stick breaking construction
            q_u = kumaraswamy_sample(conc1, conc0, sample_shape = [inp.get_shape()[1].value,K])
            pi = tf.cumprod(q_u)
            
            # posterior probabilities z
            t_pi = variable_on_cpu('sb_t_pi', [inp.get_shape()[1],K], \
                                  initializer =  tf.initializers.random_uniform(-.1, .1),
                                  constraint = lambda x: tf.clip_by_value(x, -5.,600.),\
                                  dtype = tf.float32)
            t_pi = tf.nn.sigmoid(t_pi)
        
        
        biases=0.
        if bias:
            biases = variable_on_cpu('bias', [K*U], tf.constant_initializer(0.1))
    
        # train branch
        if train:
            
            # reparametrizable normal sample
            eps = tf.stop_gradient(tf.random_normal([inp.get_shape()[1].value, K*U]))
            W = mW + eps * sW
            
            z=1.
            # stick breaking process and kl terms
            if sbp:
                
                # sample relaxed bernoulli
                z_sample = bin_concrete_sample(t_pi,temp_bern)
                z = tf.tile(z_sample, [1,U])
                re = z*W
                
                # kl terms for the stick breaking construction
                kl_sticks = tf.reduce_sum(kumaraswamy_kl(tf.ones_like(conc1), tf.ones_like(conc0),
                                                     conc1, conc0, q_u))
                kl_z = tf.reduce_sum(bin_concrete_kl(pi, t_pi,temp_bern, z_sample))
            
                tf.add_to_collection('kl_loss', kl_sticks)
                tf.add_to_collection('kl_loss', kl_z)
                
                tf.summary.scalar('kl_sticks', kl_sticks)
                tf.summary.scalar('kl_z', kl_z)
                
                # cut connections if probability of activation less than tau
                tf.summary.scalar('sparsity', tf.reduce_sum(tf.cast(tf.greater(t_pi/(1.+t_pi), tau), tf.float32))*U)
                
            else:
                re = W
                
            # add the kl for the weights to the collection
            kl_weights = tf.reduce_sum(normal_kl(tf.zeros_like(mW), tf.ones_like(sW), \
                                                 mW, sW,W))
            tf.add_to_collection('kl_loss', kl_weights)
            tf.summary.scalar('kl_weights', kl_weights)
                
            # dense calculation
            lam = tf.matmul(inp, re)  + biases
            
            # activation branches
            if activation=='lwta':
                
                assert U>1, 'The number of competing units should be larger than 1'
                
                # reshape weight for LWTA
                lam_re = tf.reshape(lam, [-1,K,U])
                
                # calculate probability of activation and some stability operations
                prbs = tf.nn.softmax(lam_re) + 1e-4
                prbs /= tf.reduce_sum(prbs, -1, keepdims=True)
                
                # relaxed categorical sample
                xi = concrete_sample(prbs, temp_cat)
                
                #apply activation
                out  = lam_re * xi
                out = tf.reshape(out, tf.shape(lam))
                
                # kl for the relaxed categorical variables
                kl_xi =  tf.reduce_mean(tf.reduce_sum(concrete_kl(  tf.ones([S,K,U])/U, prbs, xi), [1]))
            
                tf.add_to_collection('kl_loss', kl_xi)
                tf.summary.scalar('kl_xi', kl_xi)

                
            elif activation == 'relu':
                
                out = tf.nn.relu(lam)
            
            elif activation=='maxout':
                
                lam_re =  tf.reshape(lam, [-1,K,U])
                out = tf.reduce_max(lam_re, -1)
                
            else:
                
                out = lam
            
            
        # test branch. It follows the train branch, but replacing samples with means
        else:
            
            # we use re for accuracy and z for compression (if sbp is active)
            re  =  1.
            z = 1.
             
            if sbp:
                
                mask = tf.cast(tf.greater(t_pi, tau), tf.float32)
                z = Bernoulli(probs = mask*t_pi, name="q_z_test", dtype=tf.float32).sample()
                z = tf.tile(z, [1,U])
                
                re = tf.tile(mask*t_pi,[1,U])
                

            lam = tf.matmul(inp, re*mW) + biases
            
            if activation == 'lwta':
                
                # reshape and calulcate winners
                lam_re = tf.reshape(lam, [-1,K,U])
                prbs = tf.nn.softmax(lam_re) +1e-4
                prbs /= tf.reduce_sum(prbs, -1, keepdims=True)
                                
                # apply activation
                out = lam_re*concrete_sample(prbs, 0.01)
                out = tf.reshape(out, tf.shape(lam))
                                
            elif activation == 'relu':
                
                out = tf.nn.relu(lam)
                
            elif activation=='maxout':
                
                lam_re =  tf.reshape(lam, [-1,K,U])
                out = tf.reduce_max(lam_re, -1)
                
            else:
                out = lam
                
        return out, mW, z*mW, z*sW**2, z

# =============================================================================
# SB LWTA Convolutional Layer
# =============================================================================
def SB_Conv2d(inp, ksize, S=128, padding='SAME', strides=[1,1,1,1], 
              bias = True, train = True, reuse= False, sbp=False, temp_bern=0.5, temp_cat=0.5, 
              activation='lwta', name='conv'):
    """
    Convolutional layer for the SB-LWTA model, incorporating local competition.
    
    Parameters:
        inp: 4d tensor 
            The input to the current layer.
        ksize: 5d tensor
            The size of the kernels. The last 2 dimensions denote the blocks and units therein.
        padding: str 
            The padding for the conv operation. Default: SAME. (see tf conv documentation).
        strides: 4d tensor 
            The strides for the conv operation. Default: [1,1,1,1] (see tf conv).
        bias: boolean
            Flag denoting the use of bias.
        train: boolean
            Flag to alternate between train or not branches.
        reuse: boolean
            Flag to reuse or not the variables of the layer.
        sbp: boolean
            Flag to enable or disable the stick breaking process
        temp_bern: float
            The temperature for the bernoulli relaxation
        temp_cat: float
            The temperature for the categorical relaxation
        activation: String
            Select the activation function for the current layer.
        name: str
            The name of the current layer.
            
    Returns:
        out: 4d tensor
            The output of the layer after the masked convolution operation, the addition of bias (if bias==True)
            and the LWTA activation.
        mW: 2d tensor
            The mean of the weights. Used to load values when calling the compression script
        masked_mw: 4d tensor
            The mean of the weights of the convolutional kernel masked with a sample from the IBP (if active).
            Used for calculating the compression ability of the implementation.
        masked_sw: 4d tensor
            The variance of the weights of the convolutional kernel masked with a sample from the IBP (if active).
            Used for calculating the compression ability of the implementation.
        activations: 2d tensor
            The activations for the current batch. Used for plotting the probability of activations.
    
    """
    
    K  = ksize[-2]
    U  = ksize[-1]
    tau = 1e-2 
    
    name = name+'_'+activation
    with tf.variable_scope(name, reuse=reuse):
        
        # variables for the weights
        mW = tf.get_variable('mW', [ksize[0], ksize[1], ksize[2], K*U], 
                        initializer=tf.contrib.layers.xavier_initializer(),
                        dtype= tf.float32)

        sW= tf.get_variable('sW', [ksize[0], ksize[1], ksize[2], K*U], 
                       initializer=tf.constant_initializer(-5.),
                        constraint = lambda x: tf.clip_by_value(x, -7., x ),
                        dtype= tf.float32)
        sW = tf.nn.softplus(sW)

        # variables and construction for the stick breaking process
        if sbp:
            
            # posterior concentrations for the Kumaraswamy distribution
            conc1 = variable_on_cpu('sb_t_u_1', [K],
                                      initializer = tf.constant_initializer(3.),
                                      constraint=lambda x: tf.clip_by_value(x, -6., x),
                                      dtype = tf.float32)
            
            conc0 = variable_on_cpu('sb_t_u_2', [K],
                                      initializer = tf.constant_initializer(1.),
                                      constraint=lambda x: tf.clip_by_value(x, -6., x),
                                      dtype = tf.float32)
            conc1 = tf.nn.softplus(conc1)
            conc0 = tf.nn.softplus(conc0)
            
            # stick breaking construction
            q_u = kumaraswamy_sample(conc1, conc0, sample_shape = [inp.get_shape()[1].value,K])
            pi = tf.cumprod(q_u)
            
            # posterior bernooulli (relaxed) probabilities
            t_pi = tf.get_variable('sb_t_pi', [K], \
                                  initializer =  tf.initializers.random_uniform(-5., 1.),
                                  constraint = lambda x: tf.clip_by_value(x, -7., 600.),\
                                  dtype = tf.float32)
            t_pi = tf.nn.sigmoid(t_pi)
        
        biases=0.
        if bias:
            biases = variable_on_cpu('bias', [K*U], tf.constant_initializer(0.0))
            
        z = 1.
        # train branch
        if train:
            
            # reparametrizable normal sample
            eps = tf.stop_gradient(tf.random_normal(mW.get_shape()))
            W = mW + eps*sW
            
            re = tf.ones_like(W)

            # stick breaking kl and operations
            if sbp:
                
                z_sample = bin_concrete_sample(t_pi, temp_bern)
                z = tf.tile(z_sample,[U])
                W *= z
                            
                kl_sticks = tf.reduce_sum(kumaraswamy_kl(tf.ones_like(conc1), tf.ones_like(conc0),
                                                     conc1, conc0, q_u))
                kl_z = tf.reduce_sum(bin_concrete_kl(pi, t_pi,temp_bern, z_sample))
            
                tf.add_to_collection('kl_loss', kl_sticks)
                tf.add_to_collection('kl_loss', kl_z)
                
                tf.summary.scalar('kl_sticks', kl_sticks)
                tf.summary.scalar('kl_z', kl_z)
                
                # if probability of activation is smaller than tau, it's inactive
                tf.summary.scalar('sparsity', tf.reduce_sum(tf.cast(tf.greater(t_pi/(1.+t_pi), tau), tf.float32))*U)
                
                
            # add the kl terms to the collection
            kl_weights = tf.reduce_sum(normal_kl(tf.zeros_like(mW), tf.ones_like(sW), \
                                                 mW, sW, W))
            tf.add_to_collection('losses',  kl_weights)
            tf.summary.scalar('kl_weights', kl_weights)
                        
            # convolution operation
            lam = tf.nn.conv2d(inp, W, strides=strides, padding = padding) + biases

            # choose activation based on input
            if activation == 'lwta':
                
                assert U>1, 'The number of competing units should be larger than 1'
                
                # reshape weight to calculate probabilities
                lam_re = tf.reshape(lam, [-1, lam.get_shape()[1], lam.get_shape()[2], K,U])
    
                prbs = tf.nn.softmax(lam_re) + 1e-5
                prbs /= tf.reduce_sum(prbs, -1, keepdims=True)
                
                # draw relaxed sample and apply activation
                xi = concrete_sample( prbs, temp_cat)
                out = lam_re * xi
                out = tf.reshape(out, tf.shape(lam))
                
                # add the relative kl terms
                kl_xi =  tf.reduce_mean(tf.reduce_sum(concrete_kl(  tf.ones_like(lam_re)/U, prbs, xi), [1]))
            
                tf.add_to_collection('kl_loss', kl_xi)
                tf.summary.scalar('kl_xi', kl_xi)
                
                
            elif activation == 'relu':
                # apply relu
                out = tf.nn.relu(lam)
                
            elif activation == 'maxout':
                #apply maxout activation
                lam_re = tf.reshape(lam, [-1, lam.get_shape()[1], lam.get_shape()[2], K,U])
                out = tf.reduce_max(lam_re, -1, keepdims=False)
            
            else:
                print('Activation:', activation, 'not implemented.')
        
        # test branch, same with train but replace samples with means
        else:
            re = tf.ones_like(mW)
            z = 1.
           
            # if sbp is active calculate mask and draw samples
            if sbp:
                mask = tf.cast(tf.greater(t_pi, tau), tf.float32)
                z = Bernoulli(probs = mask*t_pi, name="q_z_test", dtype=tf.float32).sample()
                z = tf.tile(z, [U])
                re =  tf.tile(mask*t_pi,[U])
                
            # convolution operation
            lam = tf.nn.conv2d(inp, re *mW,  strides=strides, padding = padding) + biases
            
            
            if activation == 'lwta':
                # calculate probabilities of activation
                lam_re = tf.reshape(lam, [-1, lam.get_shape()[1], lam.get_shape()[2], K,U])
                prbs = tf.nn.softmax(lam_re) + 1e-5
                prbs /= tf.reduce_sum(prbs,-1, keepdims=True)
                
                # draw sample for activated units
                out = lam_re * concrete_sample(prbs, 0.01)
                out = tf.reshape(out, tf.shape(lam))
                
            elif activation == 'relu':
                # apply relu
                out = tf.nn.relu(lam)
                
            elif activation=='maxout':
                # apply maxout operation
                lam_re = tf.reshape(lam, [-1, lam.get_shape()[1], lam.get_shape()[2], K,U])
                out = tf.reduce_max(lam_re, -1)

            else:
               print('Activation:', activation,' not implemented.')
               
                
    return out, mW, z * mW, z * sW**2, z


# =============================================================================
# Just a custom conv layer with mean and variance for some checks
# =============================================================================
def customConv2d(inp, ksize, activation=None, reuse= False, train= True,\
                 padding='SAME', strides=[1,1,1,1], bias = True, batch_norm=False, name='conv' ):
    
    with tf.variable_scope(name, reuse = reuse):
        mW = tf.get_variable(name+'mW',
                        [ksize[0],ksize[1],ksize[2],ksize[3]*ksize[4]], 
                        dtype= tf.float32)
        
        sW = tf.get_variable(name+'sW',
                        [ksize[0],ksize[1],ksize[2],ksize[3]*ksize[4]], 
                        initializer = tf.constant_initializer(-5.),
                        dtype= tf.float32)
        
        sW = tf.nn.softplus(sW)
        
        eps = tf.stop_gradient(tf.random_normal(mW.get_shape()))
        W = mW + eps*sW
        
        out = tf.nn.conv2d(inp, W, strides = strides, padding= padding)
        
        if bias:
            bias_conv = tf.get_variable('biases', ksize[-1]*ksize[-2], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
            out = tf.nn.bias_add(out,bias_conv)
            
        if activation:
            out = activation(out)
            
    return out, -1.,-1.
