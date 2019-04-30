# -*- coding: utf-8 -*-
"""
Script file for the implementation of the LeNet-300-100 network for the paper

The network consists of two layers with local competition with 150 blocks with 2 competing units
for the first layer and 50 blocks with 2 competing units for the second, followed by an output layer for
classification.

By changing U, you can choose the competing units, provided that 300 and 100 % U =0

Note: the original LeNet-300-100 network consists of two feedforward fully connected layers
with 300 neurons in the first layer and 100 neurons in the second layer.

@author: Currently anonymous

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import os, time

import numpy as np
#if it doesnt find some files
#change what to append to path depending on the structure of yout proj
import sys
if 'SOURCE_CODE_PATH' in os.environ:
    sys.path.append(os.environ['SOURCE_CODE_PATH'])
else:
    sys.path.append(os.getcwd())
    
import tensorflow as tf

from utils.metrics import elbo, accuracy
from layers.base import  SB_Layer
from utils.bit_precision import compute_reduced_weights

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/MNIST_data', one_hot=False)


# =============================================================================
# DATA
# =============================================================================
S = 128
J = 28
C = 10
num_batches = mnist.train.images.shape[0] // S
trainN = int(S*num_batches)


# =============================================================================
# TRAIN PARAMS
# =============================================================================
lr = 1e-3
epochs = 10
test_time = 5

# number of competing units
U = 2 

# temperature annealing parameters
temp = 1.
tau0 = 1.
ANNEAL_RATE = 0.00001
MIN_TEMP =0.5

# kl scale
kl = 1.

# keep flag to true if you want to compress the weights after training
compress = True

# activate stick breaking process
sbp=False

# =============================================================================
# MODEL
# =============================================================================
def lenet_300_100(inp, C, train=True, reuse=False, temp_bern=0.67, temp_cat=0.5, U=2, sbp=False):
    """
    Construct the feedforward lenet 300-100 network using the LWTA layers.
    
    Parameters:
        inp: 2d matrix
            The input (usually placeholder for images) to the network.
        C: int
            The number of classes for the dataset.
        train: boolean
            Flag denoting the construction of the train or the test branch of the graph.
        reuse: boolean
            Flag denoting the reuse of some variables that may already exist.
        temp_cat: float of placeholder
            The temperature of the categorical relaxation
        temp_bern: float or placeholde
            The temperature of the bernoulli relaxation
        U: int
            The number of competing units.
        sbp: boolean
            Flag denoting the use of the stick breaking process
            

    Outputs:
        net: 2d float
            The output of the layer with chosen activation and bias (if true)
        mw : 2d float, tf variable
            The mean of the weights. We use this variable to load the compressed mean.
        weight: 2d float
            The mean of the weights, masked with a sample from the IBP
        var: 2d float
            The var of the weights, masked with a sample from the IBP
        mask: 2d float
            The mask for the current layer (values are 0. and 1.)
    """
    
    if 300 % U !=0 or 100 % U !=0:
        print('Cannot properly divide neurons in blocks. Please provide a different U.')
        sys.exit(-1)
        
    mws = []
    masked_m = []
    masked_s = []
    masks = []
    
    # first feedforward layer with 300 output units
    net, mw, mm, ms, mask = SB_Layer(inp, int(300/U), U , temp_bern = temp_bern, temp_cat = temp_cat,\
                         activation='lwta', name='lwta_1', train=train, reuse=reuse, sbp=sbp)
    mws.append(mw)
    masked_m.append(mm)
    masked_s.append(ms)
    masks.append(mask)
    
    # second feedforward layer with 100 output units
    net, mw, mm, ms, mask = SB_Layer(net, int(100/U), U ,temp_bern = temp_bern, temp_cat = temp_cat,\
                         activation='lwta', name='lwta_2',train=train, reuse=reuse, sbp=sbp)
    mws.append(mw)
    masked_m.append(mm)
    masked_s.append(ms)
    masks.append(mask)
    
    # classification layer
    net, mw, mm, ms, mask = SB_Layer(net, C, 1, activation='none', name = 'out', train=train, reuse=reuse, sbp=sbp)
    mws.append(mw)
    masked_m.append(mm)
    masked_s.append(ms)
    masks.append(mask)
    
    return net, mws, masked_m, masked_s, masks

#==============================================================================
# COMPRESSION FUNCTION
#==============================================================================
def compression(base_folder, x_ph, y_ph, mws, masked_m ,masked_s, masks):
    """
    Function used to compress the weights and calculate bit precision and accuracy
    
    Inputs:
        base_folder: string
            The base folder of the current experiment.
        x_ph: 2d float placeholder
            The input placeholder for the data.
        y_ph: 2d float placeholder
            The placeholder for the labels.
        mws: list
            list of the means objects of the weights to load the values.
        masked_m: list
            The weights' means masked by a sample from the ibp.
        masked_s: list
            The weights' variances masked by a sample from the ibp.
        masks: list
            The masks from the IBP for each layer.
    
    """
    
    # calculate the masks and evaluate masked weights to numpy
    test_acc_total = 0.
    pruned_m = []
    pruned_s = []
    
    print('\n\nCOMPRESSION:')
    for l in range(len(masked_m)):
        if sbp:
            mask = masks[l].eval()
            print('Remaining feature maps/connections:', int(mask.sum()), '/', int((mask>-1.).sum()))
        pruned_m.append(masked_m[l].eval())
        pruned_s.append(masked_s[l].eval())
        
    print('')
    
    # calculate significant bits and rounded-off weights
    weights, significant_bits, exponent_bits = compute_reduced_weights(pruned_m, pruned_s)
    print('\nExponent:', exponent_bits)
    print('Significant:', significant_bits)
    
    # write bit precision and exponent bits to folder of checkpoint
    with open(base_folder+'bit_precision.txt','w') as f:
        f.write(str(significant_bits))
        f.write(' ')
        f.write(str(exponent_bits))
    
    # load the reduced weights as the mean to test
    for i in range(len(mws)):
        mws[i].load(weights[i], sess)
        
    # test set acc and loss
    test_iter = int(mnist.test.images.shape[0]/S)+1
    for i in range(test_iter):
        batch = mnist.test.next_batch(S)
        batch_test_acc = sess.run([accuracy_op_test], feed_dict={
                                                           x_ph: batch[0],\
                                                           y_ph: batch[1]})[0]
        test_acc_total += batch_test_acc/test_iter
    
    # write the compressed accuracy for easier access
    with open(base_folder+'compressed_acc.txt', 'w') as f:
        f.write(str(test_acc_total))
    
    # print the compressed accuracy
    print('\nCompressed Accuracy: {:0.2f}'.format(test_acc_total*100))
    
    
#==============================================================================
#M MAIN SCRIPT
#==============================================================================
if __name__=='__main__':
    
    with tf.Graph().as_default() as graph, tf.device('/cpu:0'):
        
        # some placeholders
        x_ph = tf.placeholder(tf.float32, [None,784], name='x_ph')
        y_ph = tf.placeholder(tf.int32,[None], name='y_ph')
        kl_scale = tf.placeholder(tf.float32, name='kl_scale')
        temp_bern_ph = tf.placeholder(tf.float32, name = 'temp_bernoulli')
        temp_cat_ph = tf.placeholder(tf.float32, name = 'temp_cat')
        lr_ph = tf.placeholder(tf.float32, name = 'learning_rate')
        
    
        # construct train and test branches of the graph
        logits_op_train, _, _, _, _  = lenet_300_100(x_ph, C,  train=True, reuse=False, temp_bern = temp_bern_ph, temp_cat= temp_cat_ph, U=U, sbp=sbp)
        logits_op_test, mws, masked_m ,masked_s, masks =  lenet_300_100(x_ph, C,  train=False, reuse=True, U=U, sbp=sbp)
        
        # losses for the train and test/validation sets
        loss_op_train = elbo(logits_op_train, y_ph, 50000., kl_weight= kl_scale)
        loss_op_test = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_op_test,
                                                                                     labels=y_ph))
        
        # accuracy operations for classification
        accuracy_op_train = accuracy(logits_op_train, y_ph)
        accuracy_op_test = accuracy(logits_op_test, y_ph)
        
        # the global step is used mainly for summaries
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)

        # get variables and stick breaking variables (easier if you want to try different learning rates)
        variables = list(filter(lambda v: 'optimizer' not in v.name.lower(), tf.trainable_variables()))
        net_variables = list(filter(lambda v: 'sb' not in v.name.lower(), variables))
        sbp_variables = list(filter(lambda v: 'sb' in v.name.lower(), variables))
        grads = optimizer.compute_gradients(loss_op_train, net_variables + sbp_variables)
      
        # the train operation
        train_op = optimizer.apply_gradients(grads, global_step=global_step)
    
    
        # create some summaries that can be loaded on tensorboard for better visualization
        tf.summary.scalar('train_accuracy', accuracy_op_train)
        
        train_summaries = tf.summary.merge_all()
        test_acc = tf.placeholder(tf.float32, shape=[], name='test_acc_placeholder')
        test_acc_summary = tf.summary.scalar('test_accuracy', test_acc)
        test_loss = tf.placeholder(tf.float32, shape=[], name='test_loss_placeholder')
        test_loss_summary = tf.summary.scalar('test_loss', test_loss)
        test_summaries = tf.summary.merge([test_acc_summary, test_loss_summary])
        
        
        # where to write summaries and models
        base_folder =  os.path.dirname(os.path.realpath(__file__))
        base_folder+='\\models_lwta\\'+time.strftime('%d_%m_%H_%M')+'_lr_'+str(lr)+'_U_'+str(U)+'\\'
            
        summaries_dir = base_folder
        train_writer = tf.summary.FileWriter(summaries_dir + '/train',graph)
        test_writer = tf.summary.FileWriter(summaries_dir + '/test', graph)
        val_writer = tf.summary.FileWriter(summaries_dir + '/val', graph)
    
        # a saver to save models
        saver = tf.train.Saver()

        # basic configuration for the session
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    
        with tf.Session(config=config) as sess:
            # init all variables
            sess.run(tf.global_variables_initializer())
            
            
            best_val_acc = 0.0
            
            # train the model for #epochs
            for epoch in range(epochs):
                
                epoch_loss = 0.
                
                # every 10 epochs, anneal the temperature
                if epoch % 10 == 1:
                    temp = np.maximum(tau0*np.exp(-ANNEAL_RATE*((epoch+1)*num_batches)),MIN_TEMP)

                # train operation for all batches
                for i in range(num_batches):
                    
                    batch = mnist.train.next_batch(S)
                    _, summary = sess.run([train_op, train_summaries], feed_dict={lr_ph: lr, kl_scale: kl, temp_bern_ph: temp,\
                                          temp_cat_ph: temp, x_ph: batch[0], y_ph: batch[1]})
                    train_writer.add_summary(summary, global_step.eval())
                    
                 # test time is declared above (default is 5) to reduce the computational cost of evalutation
                if epoch%test_time ==0:
                    
                    val_loss_total, val_acc_total = 0.0, 0.0
                    val_iter = int(mnist.validation.images.shape[0]/S)+1
                    
                    for i in range(val_iter):
                        batch = mnist.validation.next_batch(S)
                        batch_val_acc, batch_val_loss = sess.run([accuracy_op_test, loss_op_test], feed_dict={
                                                                           x_ph: batch[0],\
                                                                           y_ph: batch[1]})
                        val_acc_total += batch_val_acc/val_iter
                        val_loss_total += batch_val_loss/val_iter
                        
                    summary = sess.run([test_summaries], feed_dict={test_acc: val_acc_total, test_loss: val_loss_total})
                         
                    for s in summary:
                        val_writer.add_summary(s, global_step.eval())
                    
                    # keep the best model and we can later load
                    # the best model on the validation set is tested for the test set
                    if val_acc_total >= best_val_acc - 0.001:
                        saver.save(sess, base_folder + 'models\\best_model.ckpt')
                        best_val_acc = val_acc_total
                        
                        test_loss_total, test_acc_total = 0.0, 0.0
                        test_iter = int(mnist.test.images.shape[0]/S)+1
                        
                        for i in range(test_iter):
                            batch = mnist.test.next_batch(S)
                            batch_test_acc, batch_test_loss = sess.run([accuracy_op_test, loss_op_test], feed_dict={
                                                                               x_ph: batch[0],\
                                                                               y_ph: batch[1]})
                            test_acc_total += batch_test_acc/test_iter
                            test_loss_total += batch_test_loss/test_iter
                        
                        summary = sess.run([test_summaries], feed_dict={test_acc: test_acc_total, test_loss: test_loss_total})
                         
                        for s in summary:
                            test_writer.add_summary(s, global_step.eval())
                    
                    # also save the current model
                    saver.save(sess,base_folder+'models\\cur_model.ckpt')
                    
                    # print some stuff to track progress
                    print('Epoch:', epoch, ' Val Acc: {:0.2f}, Valloss: {:0.2f}, Best Val Acc: {:0.2f}, Test Acc: {:0.2f}'.format(val_acc_total*100.,\
                          val_loss_total, best_val_acc*100, test_acc_total*100))
                    
            # if compress flag is True, compress the weights
            if compress:
                compression(base_folder, x_ph, y_ph, mws, masked_m ,masked_s, masks)