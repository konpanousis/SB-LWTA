Code Implementation for the submitted paper "Nonparametric Bayesian Deep Networks with Local Competition".

Project Structure:

Data Folder: Containing files for reading the CIFAR10 dataset.

Layers Folder: Contains 2 files for the considered methods.
	(i) base.py containing the code implementation for dense and convolutional layers of the SB-LWTA method described in the paper.
	(ii) base_bc containing the corresponding code for the implementation of methods described in Bayesian Compression for Deep Learning.

Models Folder: Contains the run scripts for the considered architectures in their respective subfolder:
	(i) Lenet-300-100: lenet_300_100.py Main script for training and testing the lenet 300-100 architecture.
	(ii) Lenet5: lenet5.py Main script for training and testing the lenet5 convolutional architecture.
	(iii) ConvNet: Contains three script
		(a) cifar10_train_convnet.py: Train and test the ConvNet architecture using the SB-LWTA implementation.
		(b) cifar10_train_convnet_gnj.py Train and test the ConvNet architecture using BC-GNJ.
		(c) cifar10_train_convnet_ghs.py Train and test the ConvNet architecture using BC-GHS.

Utils Folder: All helper functions, e.g. reparametrization tricks for distributions, bit precision calculation, e.t.c.
	(i) bit_precision.py All necessary functions for calculating the bit precision. This is the implementation was provided in https://github.com/KarenUllrich/Tutorial_BayesianCompressionForDL.
	(ii) distributions.py All the necessary function for sampling, kl divergences etc for our SB-LWTA implementation.
	(iii) figs.py Functions to plot the activations for the LWTA nonlinearities.
	(iv) graph.py Functions to create graph when training using CIFAR10.
	(v) metrics.py Functions for the evidence lower bound and accuracy metric.

Run instructions:
1) Use the terminal or the command prompt to navigate to the StickBreaking-LWTA-ICML folder.
2) Run the desired script as follows:
	python models\desired_model\desired_script.py
Replace desired model with lenet-300-100, lenet5 or convnet and desired script with their respective script.
Example run: python models\lenet-300-100\lenet_300_100.py

Arguments:
You can change the values of a script to change its behavior.
For example, to run the lenet-300-100 network with relu activations instead of the default LWTA implementation, replace the activation='lwta' occurences with activation='relu'.
The functions call arguments are explained in their respective definition.
