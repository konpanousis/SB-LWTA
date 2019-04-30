# -*- coding: utf-8 -*-
"""
Functions to transform and plot the activations as described in the paper.

@author: Currently anonymous
"""

import itertools
import numpy as np
import matplotlib.pyplot as plt

import os

def transform_activations(mnist, x_ph, y_ph, activations, sess, C, U):
    acts_l1 = []
    acts_l2 = []
    acts_l3 =[]
    x = mnist.test.images
    y = mnist.test.labels
    

    for i in range(C):
        indices = np.where(y==i)
        acts_1, acts_2, acts_3 = sess.run(activations, feed_dict={x_ph: x[indices], y_ph: y[indices]})
        
        acts_1_mean = acts_1.mean(0)
        acts_1_max = np.max(acts_1_mean, -1, keepdims=True)
        acts_1 = acts_1_mean >= np.tile(acts_1_max, [1,U])
        acts_1 = np.reshape(acts_1, [-1])
        
    
        acts_2_mean = acts_2.mean(0)
        acts_2_max = np.max(acts_2_mean, -1, keepdims=True)
        acts_2 = acts_2_mean >= np.tile(acts_2_max, [1,U])
        acts_2 = np.reshape(acts_2, [-1])
        
        acts_3_mean = acts_3.mean(0)
        acts_3_max = np.max(acts_3_mean, -1, keepdims=True)
        acts_3 = acts_3_mean >= np.tile(acts_3_max, [1,U])
        acts_3 = np.reshape(acts_3, [-1])
        
        acts_l1.append(acts_1)
        acts_l2.append(acts_2)
        acts_l3.append(acts_3)
        
    return [np.array(acts_l1), np.array(acts_l2), np.array(acts_l3)]


def plot_activations(path, prior_activations, activations, C=10, U=2):
    
    combs = []
    for i in range(C):
        for j in range(i, C):
            combs.append((i,j))
            
    a = list(range(C))
    b = list(range(C))
    combs_2 = list(itertools.product(a, b))
    
    if not os.path.exists(path+'svg'):
        os.mkdir(path+'svg')
    
    for i in range(len(activations)):
        percentage = []
        
        for comb in combs_2:
                x = np.where(activations[i][comb[0]]==1)[0]
                y = np.where(activations[i][comb[1]]==1)[0]
                percentage.append(np.mean(x==y))
    
        percentage = np.array(percentage).reshape([10,10])
        
        fig, ax = plt.subplots()
        #fig.suptitle('Common Activations in Layer 1', fontsize=16)
    
        im1 = ax.imshow(percentage, cmap = 'binary')
        fig.colorbar(im1, ax=ax)
        #fig.clim(0, 1);
        plt.xticks(list(range(C)))
        plt.yticks(list(range(C)))
        plt.xlabel('Digits')
        plt.ylabel('Digits')
        
        plt.savefig(path+'svg\\digits_layer_'+str(i)+'.svg', format='svg', dpi=1200)
        #tikz_save(path+'tikz\\layer_'+str(i)+'.tex', figureheight='4cm', figurewidth='6cm')
        plt.close(fig)
    
            
        percentage = []
        percentage_rand = []
        for comb in combs:
          
            x = np.where(activations[i][comb[0]]==1)[0]
            y = np.where(activations[i][comb[1]]==1)[0]
            percentage.append(np.mean(x==y))
            
            x = np.where(prior_activations[i][comb[0]]==1)[0]
            y = np.where(prior_activations[i][comb[1]]==1)[0]
            percentage_rand.append(np.mean(x==y))
                
        percentage = np.array(percentage)
        percentage_rand = np.array(percentage_rand)
        
        fig = plt.figure()
        p1 = plt.bar(np.arange(len(combs)), percentage_rand, color='gray', edgecolor='white', label='Untrained')
        p2 = plt.bar(np.arange(len(combs)), percentage,  color='black', edgecolor='white', label='Trained')
   
        plt.ylabel('% of common neuron activations')
        plt.xlabel('Pairs of digits')
        plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=10,
                   ncol=2, borderaxespad=0., frameon=False)
        plt.tight_layout()
        plt.legend((p1[0], p2[0]), ('Untrained', 'Trained'), frameon=False)
    
        plt.savefig(path+'svg\\pairs_layer_'+str(i)+'.svg', format='svg', dpi=1200)
        #tikz_save(path+'tikz\\layer_'+str(i)+'.tex', figureheight='4cm', figurewidth='6cm')
        plt.close(fig)
#        
        
    # neurns per digit
    for i in range(len(activations)):
        print(activations[i].shape)
        fig = plt.figure(figsize=(10,5))
        acts = activations[i][:,:20]
        plt.imshow(acts, aspect='equal', cmap='binary')
        ax = plt.gca()
        size = acts.shape[-1]
    
        ax.set_xticks(np.arange(-.5, size, U))
        ax.set_yticks(np.arange(0,10,1))
    
        ax.set_xticklabels(np.arange(0,size+1, U))
        ax.set_yticklabels(np.arange(0,10,1))
        
        ax.set_xticks(np.arange(.51, size, 1), minor=True)
        ax.set_yticks(np.arange(.51, 10, 1), minor=True);
    
    
        ax.grid(which='minor', color=(240/255.,240./255,240/255.), linestyle=':', linewidth=1)
        ax.xaxis.grid(True,'major', color='black', linewidth=2)
        ax.yaxis.grid(True,'minor')
    
        
        plt.xlabel('Units')
        plt.ylabel('Digits')
        
        plt.colorbar(fraction=0.05, pad=0.01)
        plt.clim(0, 1);
 
        plt.tight_layout()
        plt.savefig(path+'svg\\_layer_'+str(i)+'all_digits.svg', format='svg', dpi=1200)
        #tikz_save(path+'tikz\\layer_'+str(i)+'all_digits.tex',  figureheight='20cm', figurewidth='20cm')
        plt.close(fig)