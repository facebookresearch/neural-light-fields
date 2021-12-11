# Code based on Seyoung Park's TensorFlow port of original LPIPS code by Zhang et al. (https://github.com/richzhang/PerceptualSimilarity)

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import sys
import numpy as np
import tensorflow as tf
from collections import namedtuple

from . import networks
from . util import as_tuple, for_each, np_dtype, f32_to_dtype


DATA_DIR = os.path.dirname(os.path.realpath(__file__))


def normalize_tensor(in_feat, eps=1e-10):
	'''Normalizes a tensor to unit length in the depth/feature dimension.'''
	norm_factor = tf.sqrt(tf.reduce_sum(tf.square(in_feat), axis=3, keepdims=True))
	return in_feat / (norm_factor + eps)


	

# Learned perceptual metric.
class PNetLin(object):
	def __init__(self, 
		pnet_type='vgg',           # 'vgg', 'squeeze', 'vgg_ensemble', 'squeeze_ensemble_maxpool'
		use_lpips_dropout=False,   # Score dropout
		use_net_dropout=False,     # Dropout inside VGG/SqueezeNet/etc.
		net_dropout_keep_prob=0.99,
		trainable=False,           # 'lpips', 'net', 'both'
		custom_lpips_weights=None, # Custom linear weights
		custom_net_weights=None,   # Custom VGG/SqueezeNet weights
		dtype=tf.float32           # Data type for VGG/SqueezeNet/etc.
	):
		super(PNetLin, self).__init__()

		self.pnet_type = pnet_type
		
		self.use_lpips_dropout = use_lpips_dropout
		self.use_net_dropout = use_net_dropout
		
		self.chns = [64,128,256,512,512]
		
		self.net_trainable = True if trainable == 'net' or trainable == 'both' else False
		self.lpips_trainable = True if trainable == 'lpips' or trainable == 'both' else False
		
		self.linear_weight_as_dict = custom_lpips_weights if custom_lpips_weights is not None else None
		
		self.dtype = dtype
		
		if pnet_type == 'vgg':
			self.net = networks.vgg16(use_net_dropout=use_net_dropout, net_dropout_keep_prob=net_dropout_keep_prob, trainable=self.net_trainable, custom_net_weights=custom_net_weights, dtype=dtype)
			
			if self.linear_weight_as_dict is None:
				self.linear_weight_as_dict = np.load(os.path.join(DATA_DIR, "vgg_maxpool.npy"), allow_pickle=True).item()
						
		elif pnet_type == 'squeeze':
			self.net = networks.squeezenet1_1(use_net_dropout=use_net_dropout, net_dropout_keep_prob=net_dropout_keep_prob, trainable=self.net_trainable, custom_net_weights=custom_net_weights, dtype=dtype)
			
			if self.linear_weight_as_dict is None:
				self.linear_weight_as_dict = np.load(os.path.join(DATA_DIR, "squeeze.npy"), allow_pickle=True).item()

		elif pnet_type == 'squeeze_ensemble_maxpool':
			self.net = networks.squeezenet1_1_full_maxpool(use_net_dropout=use_net_dropout, net_dropout_keep_prob=net_dropout_keep_prob, trainable=self.net_trainable, custom_net_weights=custom_net_weights, dtype=dtype)
			
			if self.linear_weight_as_dict is None:
				self.linear_weight_as_dict = np.load(os.path.join(DATA_DIR, "squeeze_full_maxpool.npy"), allow_pickle=True).item()
			
		elif pnet_type == 'vgg_ensemble':
			self.net = networks.vgg16_full_avg(use_net_dropout=use_net_dropout, net_dropout_keep_prob=net_dropout_keep_prob, trainable=self.net_trainable, custom_net_weights=custom_net_weights, dtype=dtype)
			
			if self.linear_weight_as_dict is None:
				self.linear_weight_as_dict = np.load(os.path.join(DATA_DIR, "vgg_full_avg.npy"), allow_pickle=True).item()
		
		else:
			raise Exception('Unsupported pnet_type.')
		
		if self.lpips_trainable:
			with tf.variable_scope('lpips_weights'):
				if custom_lpips_weights:
					self.linear_weight_as_dict = {
						key: tf.get_variable(key, dtype=tf.float32, initializer=value, trainable=True) for key, value in self.linear_weight_as_dict.items()
					}
				else:
					self.linear_weight_as_dict = {
						key: tf.get_variable(key, dtype=tf.float32, initializer=tf.zeros_like(value), trainable=True) for key, value in self.linear_weight_as_dict.items()
					}
		else:
			self.linear_weight_as_dict = {key: tf.constant(value, dtype=tf.float32) for key, value in self.linear_weight_as_dict.items()}
		
		self.shift = tf.constant([-.030, -.088, -.188],shape=(1,1,1,3),dtype=np_dtype(dtype))
		self.scale = tf.constant([.458, .448, .450],shape=(1,1,1,3),dtype=np_dtype(dtype))

		
	def _get_mean_of_linear_activation(self, diff, w):
		# Get mean of activation differences over pixels.
		if self.use_lpips_dropout:
			dropout_random = tf.random_uniform(tf.shape(as_tuple(diff)[0]), dtype=tf.float32)
			dropout_weights = tf.cast(tf.less(dropout_random, 0.5), dtype) / 0.5					
			
		def process(x):
			if self.use_lpips_dropout:
				x = dropout_weights * x
			
			x = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
			
			x = tf.nn.conv2d(
				x,
				f32_to_dtype(w, self.dtype),
				strides=[1,1,1,1],
				padding="VALID",
				name="conv"
			)
			return x[:,0,0,0]
			
		result = for_each(diff, process)
		return result


	def _forward_all_linear_activations(self, diffs):
		assert isinstance(diffs, tuple)

		layer_count = len(diffs[0])
		tuple_size = len(diffs)
			
		val = [0.0] * tuple_size
		
		for i in range(layer_count):
			key = 'lin{:d}.model.1.weight'.format(i)

			layer_diffs = tuple((diffs[j][i] for j in range(tuple_size)))
			
			mean_activations = self._get_mean_of_linear_activation(
				layer_diffs,
				self.linear_weight_as_dict[key]
			)
			
			val = [val[j] + mean_activations[j] for j in range(tuple_size)]

		return tuple(val)

	def forward(self, in0, in1):
		'''
		Parameters 'in0' and 'in1' are NHWC tensors containing N images with 3 color channels.
		
		The return value is a vector of size N of perceptual distances between in0[i,:,:,:] and in1[i,:,:,:].
		
		If 'in0' is a tuple, returns a tuple whose i'th element is the distance vector between in0[i] and in1, with each distance evaluation using the same dropout.
		'''
		with tf.name_scope("PNetLin"):
			in0_sc = for_each(as_tuple(in0), lambda X: (X - self.shift) /  self.scale) # convert in0 to tuple, and shift and scale each element
			in1_sc = (in1 - self.shift) /  self.scale 
			
			in0_size = len(in0_sc)
			
			# Collect activations of layers for all of in0 and in1.
			network_layers = list(self.net.forward(in0_sc + (in1_sc,)))
			
			# Normalize all layers in feature direction.
			for i, _ in enumerate(network_layers):
				network_layers[i] = tuple(normalize_tensor(network_layers[i][j]) for j in range(in0_size + 1))
			
			# Collect activation differences between in1 and all of in0.
			diffs = [[] for _ in range(in0_size)]
			
			for j in range(in0_size):
				for i, _ in enumerate(network_layers):
					diffs[j].append((network_layers[i][j] - network_layers[i][-1])**2)
			
			# Evaluate the losses.
			losses = self._forward_all_linear_activations(tuple(diffs))
			
			if isinstance(in0, tuple):
				return losses
			else:
				return losses[0]
