# -*- coding: utf-8 -*-
"""
Some helper functions for the distributions. 

@author: @author: Konstantinos P. Panousis, Dept. of Informatics and Telecommunications, 
         National and Kapodistrian University of Athens, Greece
"""

import tensorflow as tf
import numpy as np
from tensorflow.contrib.distributions import Beta, Normal, RelaxedBernoulli

# =============================================================================
# Some helper functions
# =============================================================================
    
def bin_concrete_sample(a, temp, eps=1e-8):
    """" 
    Sample from the binary concrete distribution with a and temperature temp.
    """
    
    U = tf.random_uniform(tf.shape(a), minval = 0., maxval=1.)
    L = tf.log(U+eps) - tf.log(1.-U+eps) 
    X = tf.nn.sigmoid((L + tf.log(a))/temp)
    
    return tf.clip_by_value(X, 1e-4, 1.-1e-4)

def concrete_sample(a, temp, eps = 1e-8):
    """
    Sample from the Concrete distribution with a and temperature temp.
    """
    
    U = tf.random_uniform(tf.shape(a), minval = 0., maxval=1.)
    G = - tf.log(-tf.log(U+eps)+eps)
    t = (tf.log(a) + G)/temp 
    out = tf.nn.softmax(t,-1)
    out += eps
    out /= tf.reduce_sum(out, -1, keepdims=True)
    return out*tf.stop_gradient(tf.cast(a>0., tf.float32))
    
def bin_concrete_kl(pr_a, post_a, post_temp, post_sample):
    """
    Calculate the binary concrete kl using the sample
    """
    
    p_log_prob = bin_concrete_log_mass(pr_a, post_temp, post_sample)
    q_log_prob = bin_concrete_log_mass(post_a,post_temp, post_sample)
   
    return -(p_log_prob - q_log_prob)
   

def concrete_kl(pr_a, post_a, post_sample):
    """
    Calculate the KL between two relaxed discrete distributions, using MC samples.
    This approach follows " The concrete distribution: A continuous relaxation of 
    discrete random variables" [Maddison et al.] and the rationale for this approximation
    can be found in eqs (20)-(22)
    
    Parameters:
        pr: tensorflow distribution
            The prior discrete distribution.
        post: tensorflow distribution
            The posterior discrete distribution
            
    Returns:
        kl: float
            The KL divergence between the prior and the posterior discrete relaxations
    """

    p_log_prob = tf.log(pr_a)
    q_log_prob = tf.log(post_a+1e-4)
   
    return -(p_log_prob - q_log_prob)


def kumaraswamy_sample(conc1, conc0, sample_shape):
    """
    Sample from the Kumaraswamy distribution with  parameters conc1 and conc0.
    """
    
    x = tf.random_uniform(sample_shape, minval=0.01, maxval=0.99)
        
    q_u = (1-(1-x)**(1./conc0))**(1./conc1)
    
    return q_u

def kumaraswamy_log_pdf(a, b, x):
    """
    Log-pdf for the Kumaraswamy distribution with parameters a and b, evaluated on x.
    """
    
    return tf.log(a) +tf.log(b) + (a-1.)*tf.log(x)+ (b-1.)*tf.log(1.-x**a)

def kumaraswamy_kl(prior_alpha, prior_beta,a,b, x):
    """
    Implementation of the KL distribution between a Beta and a Kumaraswamy distribution.
    Code refactored from the paper "Stick breaking DGMs". Therein they used 10 terms to 
    approximate the infinite taylor series.
    
    Parameters:
        prior_alpha: float/1d, 2d
            The parameter \alpha  of a prior distribution Beta(\alpha,\beta).
        prior_beta: float/1d, 2d
            The parameter \beta of a prior distribution Beta(\alpha, \beta).
        a: float/1d,2d
            The parameter a of a posterior distribution Kumaraswamy(a,b).
        b: float/1d, 2d
            The parameter b of a posterior distribution Kumaraswamy(a,b).
            
    Returns:
        kl: float
            The KL divergence between Beta and Kumaraswamy with given parameters.
    
    """
    
    q_log_prob = kumaraswamy_log_pdf(a, b, x)
    p_log_prob  = Beta(prior_alpha, prior_beta).log_prob(x)

    return -(p_log_prob-q_log_prob)

def normal_kl(m1,s1,m2,s2, sample):
    """
    KL divergence for the Normal distribution using MC sampling.
    """
    
    p_log_prob = Normal(m1, s1).log_prob(sample)
    q_log_prob = Normal(m2, s2).log_prob(sample)
   
    return  -(p_log_prob - q_log_prob)





    v = tf.where(u > u_prime, v_1, v_0)
    v = tf.check_numerics(v, 'v sampling is not numerically stable.')
    v = v + tf.stop_gradient(-v + u)  # v and u are the same up to numerical errors

    return v
