# -*- coding: utf-8 -*-
"""
Some helper functions for the distributions. 

@author: Currently anonymoys
"""

import tensorflow as tf
import numpy as np
from tensorflow.contrib.distributions import Beta, Normal, RelaxedBernoulli

# =============================================================================
# Some helper functions
# =============================================================================
    
def bin_concrete_sample(a, temp, eps=1e-8):
    """" 
    Sample from the binary concrete distribution
    """
    U = tf.random_uniform(tf.shape(a), minval = 0., maxval=1.)
    L = tf.log(U+eps) - tf.log(1.-U+eps) 
    X = tf.nn.sigmoid((L + tf.log(a))/temp)
    
    return tf.clip_by_value(X, 1e-4, 1.-1e-4)

def concrete_sample(a, temp, eps = 1e-8):
    """
    Sample from the Concrete distribution
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
    x = tf.random_uniform(sample_shape, minval=0.01, maxval=0.99)
        
    q_u = (1-(1-x)**(1./conc0))**(1./conc1)
    
    return q_u

def kumaraswamy_log_pdf(a, b, x):
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
    p_log_prob = Normal(m1, s1).log_prob(sample)
    q_log_prob = Normal(m2, s2).log_prob(sample)
   
    return  -(p_log_prob - q_log_prob)

#####################
## EXTRA
#########

def sample_lognormal(mu, sigma):
    U = tf.random_normal(tf.shape(mu))
    normal_sample = mu + U*sigma
    log_normal_sample = tf.exp(normal_sample)
    
    return tf.clip_by_value(log_normal_sample, 1e-3, log_normal_sample)

def lognormal_kl(mu, sigma):
    return 0.5*(mu**2 + sigma**2 -1. ) - 2*tf.log(sigma)

def exponential_sample(rate, eps = 1e-8):
    U = tf.random_uniform(tf.shape(rate), minval = np.finfo(np.float32).tiny, maxval=1.)
    
    return -tf.log(U+eps)/(rate + eps)

def exponential_kl(rate0, rate):
    return tf.log(rate) - tf.log(rate0) + rate0/rate - 1.


def sas_kl(alpha, gamma, mu, sigma ):
    # maybe it's not alpha and it's alpha/alpha+1 and the same for gamma
    safe_one_minus_alpha = tf.clip_by_value(1.-alpha, 1e-3, 1.-1e-3)
    safe_alpha = tf.clip_by_value(alpha, 1e-2, 1.-1e-3)
    
    return 0.5*gamma*(-1.- 2*tf.log(sigma) + tf.square(mu) + tf.square(sigma))\
            + (1.-gamma)* (tf.log(1.-gamma) - tf.log(safe_one_minus_alpha))\
              + gamma*(tf.log(gamma)-tf.log(safe_alpha))
            
def sas_kl_2(mu, sigma, post_sample):
    kl_w = 0.5*post_sample *( -1.- 2*tf.log(sigma) + tf.square(mu) + tf.square(sigma))
    #kl_z = bin_concrete_kl(alpha, 0.5, gamma, 0.67, post_sample )
    
    return kl_w 


def concrete_mass(a, temp, x):
    # it's the log prob of the exp relaxed, so we exp it to take the log prob
    # of the relaxed
    n= tf.cast(tf.shape(a)[-1], tf.float32)
    log_norm = (tf.lgamma(n)
                      + (n - 1.)
                      * tf.log(temp))
    
    log_un = tf.nn.log_softmax(tf.log(a+1e-4) -x*temp)
    log_un = tf.reduce_sum(log_un,-1, keep_dims=True)
    
    pr = tf.clip_by_value(log_norm + log_un, -10., -1e-2)
         
    return tf.exp(pr)



def bin_concrete_log_mass(a, temp, x):
    log_pr = tf.log(temp) + tf.log(a + 1e-4 ) + (-temp-1) * tf.log(x) + (-temp-1)*tf.log(1-x)
    log_pr -= 2 * (tf.log(a + 1e-4) - temp* tf.log(x) - temp*tf.log(1-x))
    
    return log_pr

def beta_function(a,b):
    """
    Calculation of the Beta function using the lgamma (log gamma) implementation of tf.
    
    Parameters:
        a: 1d or 2d tensor
            The first parameter of the beta function
        b: 1d or 2d tensor
            The second parameter of the beta function
            
    Returns:
        out: same as input size
            The calculated beta function for given a and b
    """
    
    return tf.exp(tf.lgamma(a) + tf.lgamma(b) - tf.lgamma(a+b))
def _log_prob(loc, scale, x):
    return _log_unnormalized_prob(loc, scale, x) - _log_normalization(scale)

def _log_unnormalized_prob(loc,scale, x):
    return -0.5 * tf.square(_z(loc, scale, x))

#missing the log(2pi) term since we dont really care
def _log_normalization(scale):
    return  tf.log(scale)

def _z(loc, scale, x):
    """Standardize input `x` to a unit normal."""
    return (x - loc) / scale


def soft_sample(log_a, v, temperature = None):
    z = log_a + tf.log(v) - tf.log(1.-v)
    z /=  temperature
    
    return tf.nn.sigmoid(z)


def hard_sample(log_a, u):
    z = log_a + tf.log(u) - tf.log(1.-u)
    
    #hard thresholding
    H_z = tf.stop_gradient(tf.to_float(z > 0))
    
    return H_z
    
def u_to_v(log_alpha, u, eps=1e-8):
    """Convert u to tied randomness in v.
    Taken from tensorflow lib.
    """
    u_prime = tf.nn.sigmoid(-log_alpha)  # g(u') = 0

    v_1 = (u - u_prime) / tf.clip_by_value(1 - u_prime, eps, 1)
    v_1 = tf.clip_by_value(v_1, 0, 1)
    v_1 = tf.stop_gradient(v_1)
    v_1 = v_1*(1 - u_prime) + u_prime
    v_0 = u / tf.clip_by_value(u_prime, eps, 1)
    v_0 = tf.clip_by_value(v_0, 0, 1)
    v_0 = tf.stop_gradient(v_0)
    v_0 = v_0 * u_prime

    v = tf.where(u > u_prime, v_1, v_0)
    v = tf.check_numerics(v, 'v sampling is not numerically stable.')
    v = v + tf.stop_gradient(-v + u)  # v and u are the same up to numerical errors

    return v
