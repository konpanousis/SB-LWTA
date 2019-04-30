# -*- coding: utf-8 -*-
"""
Main file to train and test the SB-LWTA implementation of the convnet network
with the CIFAR dataset as described in the paper.

After training the model, the saved checkpoint can be used in cifar_10_train_convnet_compress
to test the compression.

Various variables such as sparsity and kl losses can be viewed with tensorboard


@author: Konstantinos P. Panousis, Dept. of Informatics and Telecommunications, 
         National and Kapodistrian University of Athens, Greece
         
"""

import os
import sys
import time
from tensorflow.python import pywrap_tensorflow

if 'SOURCE_CODE_PATH' in os.environ:
    sys.path.append(os.environ['SOURCE_CODE_PATH'])
else:
    sys.path.append(os.getcwd())

import tensorflow as tf
import urllib, tarfile

from utils.bit_precision import compute_reduced_weights
from utils.graph import build_graph
from utils.metrics import elbo, accuracy

from data import reader
from layers.base import SB_Layer, SB_Conv2d

tf.app.flags.DEFINE_string('dataset', 'cifar10', 'dataset name')
tf.app.flags.DEFINE_string('data_dir', 'C:\\tmp\\cifar10_data', 'Path to data')
tf.app.flags.DEFINE_string('summaries_dir', '', 'global path to summaries directory')
tf.app.flags.DEFINE_string('checkpoints_dir', '', 'global path to checkpoints directory')
tf.app.flags.DEFINE_string('checkpoint', '', 'global path to checkpoint file')
tf.app.flags.DEFINE_integer('batch_size', 128, 'batch size')
tf.app.flags.DEFINE_integer('num_gpus', 1, 'num GPUs')
tf.app.flags.DEFINE_float('initial_learning_rate', 1e-3, 'initial learning rate')
tf.app.flags.DEFINE_float('decay_rate', 0.9, 'the decay rate')
tf.app.flags.DEFINE_integer('epochs', 200, 'number of training epochs')
tf.app.flags.DEFINE_boolean('sbp', False, 'activate stick breaking process')
tf.app.flags.DEFINE_integer('U', 2, 'number of competing units')
tf.app.flags.DEFINE_boolean('compress', True, 'compress after training')

FLAGS = tf.app.flags.FLAGS


DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'

def sb_layer(net, units, is_training, reuse, name, activation, bias, temp_cat=0.5, temp_bern = 0.5, U=2, sbp=True):
    """
    Function to create a dense SB-LWTA layer.
    
    Inputs:
        net: 2d float
            The input data or the output of the previous layer.
        units: int
            The number of output units for the layer
        is_training: boolean
            Flag denoting train or test branches.
        reuse: boolean
            Flag to reuse or not existing variables.
        name: string
            The name of the layer.
        activation: string
            The activation to use for the specific layer (relu, lwta, maxout, none)
        bias: boolean
            Flag to use or not a bias term
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

    net, mw, weight, var, mask = SB_Layer(net, int(units/U), U,  train=is_training, reuse=reuse, \
                              activation=activation, bias=bias, temp_cat=temp_cat, temp_bern=temp_bern, name =name, sbp=sbp)
    
    return net, mw, weight, var, mask

def conv_lwta(net, num_filters, is_training, reuse, name, activation, \
              bias=True, temp_cat=0.5, temp_bern=0.5, U=2, sbp=True):
    
    """
    Function to create a dense SB-LWTA layer.
    
    Inputs:
        net: 2d float
            The input data or the output of the previous layer.
        num_filters: int
            The number of output feature maps for the layer
        is_training: boolean
            Flag denoting train or test branches.
        reuse: boolean
            Flag to reuse or not existing variables.
        name: string
            The name of the layer.
        activation: string
            The activation to use for the specific layer (relu, lwta, maxout, none)
        bias: boolean
            Flag to use or not a bias term
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

    net, mw, weight, var, mask =SB_Conv2d(net, [5, 5,  net.get_shape()[3], int(num_filters/U), U],\
                                        train=is_training, reuse=reuse,\
                                        name=name, padding='SAME', activation=activation, bias=bias, \
                                        temp_cat=temp_cat, temp_bern=temp_bern,
                                        sbp=sbp)
    
    return net, mw, weight, var, mask

#==============================================================================
# COMPRESSION FUNCTION
#==============================================================================
def compression(mws, weights, variances, masks, sess, test_acc_op, test_loss_op, steps_per_test ):
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
    
    weights_np = []
    vars_np =[]
    
    for i in range(len(weights)):
        if FLAGS.sbp:
            mask = masks[i]
            print('Remaining feature maps/connections:', mask.eval().sum())
        weights_np.append(weights[i].eval())
        vars_np.append(variances[i].eval())
        
    weights, significant_bits, exponent_bits = compute_reduced_weights(weights_np, vars_np)
    
    print('Exponent:', exponent_bits)
    print('Significant:', significant_bits)
    

    for i in range(len(mws)):
        mws[i].load(weights[i], sess)
        
    test_loss_total, test_acc_total = 0.0, 0.0
    for step_num in range(steps_per_test):
        batch_test_acc, batch_test_loss = sess.run([test_acc_op, test_loss_op])
        test_acc_total += batch_test_acc/steps_per_test
        test_loss_total += batch_test_loss/steps_per_test
    
    
  
    print(" test accuracy: %.3f test loss: %.3f" % ( test_acc_total, test_loss_total))
    
# =============================================================================
# MODEL
# =============================================================================
def net_convnet(images, nclass, is_training, reuse, temp_cat=0.5, temp_bern=0.5, U=2,  sbp=False):
    """
    Create the ConvNet architecture as described in the paper and in 
    https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10.py
    
    Inputs:
        Images: 4d placeholder
            The images placeholder to feed values during training/test
        nclass: int
            The number of classes of the dataset
        is_training: boolean
            Flag to choose between train and test branches
        reuse:
            Flag to reuse variables
        temp_cat: float of placeholder
            The temperature of the categorical relaxation
        temp_bern: float or placeholde
            The temperature of the bernoulli relaxation
    
    Outputs:
        net: 2d tensor
            The logits for classification
        mws: list
            list containting the tf variables of the weights' mean to assign values
        weights: list
            list containing the masked mean for the weights for each layer
        variances: list
            list containing the masked variances for the weights for each layer
        masks: list
            list containing the masks for each layer
    
    """
    mws= []
    weights = []
    variances = []
    masks = []
    
    net, mw, weight, var, mask = conv_lwta(images, 64, is_training, reuse, 'conv_1', 'lwta', bias=True, U=U, sbp=sbp)
    mws.append(mw)
    weights.append(weight)
    variances.append(var)
    masks.append(mask)
    
    net = tf.nn.max_pool(net, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')
    net = tf.nn.lrn(net, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)
    
    net, mw, weight, var, mask = conv_lwta(net, 64, is_training, reuse,  'conv_2', 'lwta', bias=True, U=U, sbp=sbp)
    mws.append(mw)
    weights.append(weight)
    variances.append(var)
    masks.append(mask)
    
    net = tf.nn.lrn(net, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

    net = tf.nn.max_pool(net, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')
    
    net = tf.reshape(net, [-1, (net.get_shape()[1]*net.get_shape()[2]*net.get_shape()[3]).value])

    net, mw, weight, var, mask = sb_layer(net, 384, is_training, reuse,  name='dense_1',
                             activation='lwta', bias=True, sbp=sbp, U=U)
    mws.append(mw)
    weights.append(weight)
    variances.append(var)
    masks.append(mask)
    
    net, mw, weight, var, mask = sb_layer(net, 192, is_training, reuse,  name='dense_2',
                             activation='lwta', bias=True, sbp=sbp, U=U)
    mws.append(mw)
    weights.append(weight)
    variances.append(var)
    masks.append(mask)
    
    net, mw, weight, var, mask = sb_layer(net, nclass,is_training, reuse, name='dense_3',
                             activation='none',  bias=True, sbp=sbp, U=U)
    
    mws.append(mw)
    weights.append(weight)
    variances.append(var)
    masks.append(mask)
    
    tf.add_to_collection('logits', net)
    
    return net, mws, weights, variances, masks


def main(_):
    
    # the directory for summaries and checkpoints
    summaries_dir = FLAGS.summaries_dir
    base_folder =  os.path.dirname(os.path.realpath(__file__))
    if summaries_dir == '':
        summaries_dir = base_folder+ '\\checkpoints_gnj\\convnet_{}_'.format(FLAGS.dataset)
        summaries_dir += time.strftime('_%d-%m-%Y_%H_%M_%S')
    checkpoints_dir = summaries_dir
        
    # create the graph
    with tf.Graph().as_default() as graph, tf.device('/cpu:0'):
        
        # get data producer train set
        inputs, shape, n_train_examples, nclass = reader.get_producer(FLAGS.dataset, FLAGS.batch_size, training=True,
                                                                      distorted=True, data_dir=FLAGS.data_dir)
        images_train, labels_train = inputs
        
        # get data producer test set
        inputs, shape, n_test_examples, nclass = reader.get_producer(FLAGS.dataset, FLAGS.batch_size, training=False,
                                                                     data_dir=FLAGS.data_dir)
        images_test, labels_test = inputs
    
        # BUILDING GRAPH
        devices = ['/gpu:%d' % i for i in range(FLAGS.num_gpus)]
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        num_epochs_decay = 150.
        decay_steps = n_train_examples/FLAGS.batch_size * num_epochs_decay
        lr = tf.train.exponential_decay(FLAGS.initial_learning_rate, global_step, decay_steps, FLAGS.decay_rate)

        n_epochs = FLAGS.epochs
        
        # create the inferene function based on the convnet function
        inference = lambda images, is_training, reuse: net_convnet(images, nclass, is_training, reuse, sbp=FLAGS.sbp, U =FLAGS.U)
        loss = lambda preds, labels: elbo(preds, labels, n_train_examples, kl_weight=1.)
        train_op, test_acc_op, test_loss_op, _, mws, weights, variances, masks =  build_graph(images_train, labels_train, images_test, labels_test,
                                                                global_step, loss, accuracy,
                                                                inference, lr, devices)
        
        # some summaries operation
        train_summaries = tf.summary.merge_all()
        test_acc = tf.placeholder(tf.float32, shape=[], name='test_acc_placeholder')
        test_acc_summary = tf.summary.scalar('test_accuracy', test_acc)
        test_loss = tf.placeholder(tf.float32, shape=[], name='test_loss_placeholder')
        test_loss_summary = tf.summary.scalar('test_loss', test_loss)
        test_summaries = tf.summary.merge([test_acc_summary, test_loss_summary])

        
        # SUMMARIES WRITERS
        train_writer = tf.summary.FileWriter(summaries_dir + '/train', graph)
        test_writer = tf.summary.FileWriter(summaries_dir + '/test', graph)

        # TRAINING
        steps_per_epoch = int(n_train_examples/(FLAGS.batch_size*FLAGS.num_gpus))+1
        steps_per_test = int(n_test_examples/(FLAGS.batch_size*FLAGS.num_gpus))+1
        
       
        saver = tf.train.Saver()
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allow_growth=True
        
        # if a checkpoint is provided, find the common variables between the model and the checkpoint
        if FLAGS.checkpoint !='':
            var=[]
            readers = pywrap_tensorflow.NewCheckpointReader(FLAGS.checkpoint.replace(".meta", ""))
            var_to_shape_map = readers.get_variable_to_shape_map()
            for key in var_to_shape_map:
                if 'adam' in key or 'global' in key or 'power' in key:
                    continue
                var.append(key+':0')
            net_vars = [v for v in tf.global_variables() if v.name in var]


        with tf.Session(config=config) as sess:
            
            # initialize all variables
            sess.run(tf.global_variables_initializer())

            # restore checkpoints if it's provided
            if FLAGS.checkpoint != '':
                print('restore')
                restorer = tf.train.Saver(net_vars)
                restorer.restore(sess, FLAGS.checkpoint.replace(".meta", ""))
            
            
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            
            
            best_test_acc = 0.0
            
            # train model for n_epochs
            for epoch_num in range(n_epochs):
                
                # parse through all batches
                for step_num in range(steps_per_epoch):
                    
                    _, summary = sess.run([train_op, train_summaries])
                    train_writer.add_summary(summary, global_step.eval())
                    
                # test the model after each epoch
                if epoch_num % 1 ==0:
                    
                    test_loss_total, test_acc_total = 0.0, 0.0
                    
                    for step_num in range(steps_per_test):
                        batch_test_acc, batch_test_loss = sess.run([test_acc_op, test_loss_op])
                        test_acc_total += batch_test_acc/steps_per_test
                        test_loss_total += batch_test_loss/steps_per_test
                    
                    # save the best model
                    if test_acc_total >= best_test_acc:
                        saver.save(sess, checkpoints_dir + '/models/best_model.ckpt')
                        best_test_acc = test_acc_total
                    
                    summary = sess.run([test_summaries], feed_dict={test_acc: test_acc_total, test_loss: test_loss_total})
                    for s in summary:
                        test_writer.add_summary(s, global_step.eval())
                    
                    print("Epoch %d test accuracy: %.3f best: %.3f test loss: %.3f" % (epoch_num, test_acc_total, best_test_acc, test_loss_total))
                    
                # save the current model
                saver.save(sess, checkpoints_dir + '/models/cur_model.ckpt')
            
            # compress the weights and print stats
            if FLAGS.compress:
                compression(mws, weights, variances, masks, sess, test_acc_op, test_loss_op, steps_per_test)
                
            # stop threads and queues
            coord.request_stop()
            coord.join(threads)
            

#==============================================================================
# DOWNLOAD DATASET
#==============================================================================
def maybe_download_and_extract():
  """
  Download and extract the tarball from Alex's website, if not present
  """
  dest_directory = FLAGS.data_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
          float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  extracted_dir_path = os.path.join(dest_directory, 'cifar-10-batches-bin')
  if not os.path.exists(extracted_dir_path):
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)
    
#==============================================================================
# MAIN SCRIPT
#==============================================================================
if __name__ == '__main__':
    maybe_download_and_extract()
    if tf.gfile.Exists(FLAGS.summaries_dir):
        tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.summaries_dir)
    tf.app.run()
