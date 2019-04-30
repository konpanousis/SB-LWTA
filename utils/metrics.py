# -*- coding: utf-8 -*-
"""
The ELBO expression and the accuracy metric.

@author: Currently anonymous
"""
import tensorflow as tf

def elbo(logits, labels,  num_examples=50000., l2_weight=0.0, kl_weight=1.):
    """
    Construct the evidence lower bound described in https://arxiv.org/abs/1805.07624
    using the cross entropy loss.
    
    Parameters:
        logits: 2d tensor
            The output logits of the last layer.
        labels: 1d tensor
            The labels of for the respective datapoints.
        num_examples: float
            The number of examples to scale the likelihood (considering SGD techniques).
        l2_weight: float
            The weight for the l2 regularization losses (if any).
    Returns:
        kl_weight: float
            The weight for the KL terms.
            
        total_loss: scalar
            The total loss (likelihood+KL, etc)
    """
    labels = tf.cast(labels, tf.int32)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits,
                                                                   name='cross_entropy_per_example')
    cross_entropy_mean =  tf.reduce_mean(cross_entropy, name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy_mean)
    
    kl_loss = 0.
    if len(tf.get_collection('kl_loss'))>0:
        kl_loss = tf.add_n(tf.get_collection('kl_loss'))
        
    l2_loss = 0.
    if len(tf.get_collection('l2_loss')) > 0:
        l2_loss = tf.add_n(tf.get_collection('l2_loss'))
        
    total_loss = cross_entropy_mean + kl_weight * kl_loss/num_examples + l2_weight * l2_loss
  
    return total_loss

def accuracy(logits, labels):
    """
    Calculate the accuracy between the provided logits and labels.
    
    Parameters:
        logits: 2d tensor.
            The logits from the test graph.
        labels: 1d tensor
            The labels of the considered datapoints.
    
    Returns:
        accuracy: float
            The mean accuracy for the provided datapoints.
    """
    predicted_labels = tf.cast(tf.argmax(logits, axis=1), dtype=tf.int32)
    return tf.reduce_mean(tf.cast(tf.equal(predicted_labels, labels), tf.float32))