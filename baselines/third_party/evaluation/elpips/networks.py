from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple
import tensorflow as tf
import numpy as np
import os

from . util import as_tuple, for_each, f32_to_dtype


DATA_DIR = os.path.dirname(os.path.realpath(__file__))


def make_trainable(features, trainable, variable_scope_name):
	'''If 'trainable' is True, returns 'features' as a trainable TensorFlow variable in a new variable scope.
	   Otherwise returns 'features' as a TensorFlow constant.'''
	new_features = {}
	for key, value in features.items():
		if key.endswith('bias'):
			new_features[key] = value.reshape([1, 1, 1, -1])
		elif key.endswith('weight'):
			new_features[key] = value
	
	features = new_features
	
	if trainable:
		with tf.variable_scope(variable_scope_name):
			features = {
				key: tf.get_variable(key, dtype=tf.float32, initializer=value, trainable=True) for key, value in features.items()
			}
	else:
		features = {
			key: tf.constant(value) for key, value in features.items()
		}
	
	return features
		
	
class Network(object):
	'''The convolutional part of an image classification network for use by (E-)LPIPS.'''
	
	def __init__(self, use_net_dropout, net_dropout_keep_prob, dtype=tf.float32):
		self.features = None
		self.use_net_dropout = use_net_dropout
		self.net_dropout_keep_prob = net_dropout_keep_prob
		self.dtype = dtype

	def _conv(
		self, 
		tensor,
		weight,
		bias,
		w_shape,
		padding,
		stride=[1,2,2,1],
		data_format='NHWC'
	):
		'''Give 'tensor' as a tuple to run the same dropout to multiple tensors.'''
		if self.use_net_dropout:
			dropout_random = tf.random_uniform(tf.shape(as_tuple(tensor)[0]), dtype=tf.float32)
			dropout_weights = tf.cast(tf.less(dropout_random, self.net_dropout_keep_prob), self.dtype) / float(self.net_dropout_keep_prob)
			
			tensor = for_each(tensor, lambda X: dropout_weights * X)
			
		tensor = for_each(tensor, lambda X: tf.nn.conv2d(X, f32_to_dtype(weight, self.dtype), strides=stride, padding=padding, data_format=data_format) + f32_to_dtype(bias, self.dtype))		
		tensor = for_each(tensor, lambda X: tf.nn.relu(X))
		
		return tensor
			
			
class squeezenet1_1(Network):
	def __init__(self, trainable=False, use_net_dropout=False, net_dropout_keep_prob=0.99, custom_net_weights=None, dtype=tf.float32):
		super(squeezenet1_1, self).__init__(use_net_dropout, net_dropout_keep_prob, dtype=dtype)
		
		feature_path = os.path.join(DATA_DIR, "squeeze_pytorch_transposed_nonlinear_features.npy") 
		self.features = np.load(feature_path, allow_pickle=True).item() if custom_net_weights is None else custom_net_weights

		self.trainable = trainable
		self.features = make_trainable(self.features, trainable, 'squeezenet')

	def _squeeze(self, input, index, ch_in, ch_out):
		return self._conv(
			input,
			self.features['{:d}.squeeze.weight'.format(index)],
			self.features['{:d}.squeeze.bias'.format(index)],
			w_shape=[1,1,ch_in,ch_out],
			padding='VALID',
			stride=[1,1,1,1]
		)

	def _expand(self, input, index, ch_in, ch_out):			
		e1x1 = self._conv(
			input, 
			self.features['{:d}.expand1x1.weight'.format(index)],
			self.features['{:d}.expand1x1.bias'.format(index)],
			w_shape=[1,1,ch_in,ch_out],
			padding='VALID',
			stride=[1,1,1,1]
		)
		e3x3 = self._conv(
			input, 
			self.features['{:d}.expand3x3.weight'.format(index)],
			self.features['{:d}.expand3x3.bias'.format(index)],
			w_shape=[3,3,ch_in,ch_out],
			padding='SAME',
			stride=[1,1,1,1]
		)
		
		if isinstance(input, tuple):
			return tuple((tf.concat([X, Y], 3) for X, Y in zip(e1x1, e3x3)))
		else:
			return tf.concat([e1x1, e3x3], 3)

	def _pool(self, input, ksize, strides, padding, name, data_format):
		def op(X):
			return tf.nn.max_pool(
				X,
				ksize=ksize,
				strides=strides,
				padding=padding,
				name=name,
				data_format=data_format
			)
		
		return for_each(input, op)
	
	def fire_module(self, input, index, ch_in, ch_out_squeeze, ch_out_expand):
		net = self._squeeze(input, index, ch_in, ch_out_squeeze)
		return self._expand(net, index, ch_out_squeeze, ch_out_expand)
		

	def get_slice1(self, input):
		with tf.name_scope("slice1"):
			return self._conv(
				input, 
				self.features['0.weight'],
				self.features['0.bias'],
				w_shape=[3,3,3,64],
				padding='VALID',
				stride=[1,2,2,1]
				)
		   
	def get_slice2(self, input):
		with tf.name_scope("slice2"):
			net = self._pool(
				input,
				ksize=[1,3,3,1],
				strides=[1,2,2,1],
				padding="VALID",
				name="max_pool_slice2",
				data_format='NHWC'
			)

			net = self.fire_module(
				input=net, 
				index=3,
				ch_in=64,
				ch_out_squeeze=16,
				ch_out_expand=64
				)

			return self.fire_module(
				input=net, 
				index=4,
				ch_in=128,
				ch_out_squeeze=16,
				ch_out_expand=64
				)

	def get_slice3(self, input):
		with tf.name_scope("slice3"):
			net = self._pool(
				input,
				ksize=[1,3,3,1],
				strides=[1,2,2,1],
				padding="VALID",
				name="max_pool_slice3",
				data_format='NHWC'
			)

			net = self.fire_module(
				input=net, 
				index=6,
				ch_in=128,
				ch_out_squeeze=32,
				ch_out_expand=128
				)
			return self.fire_module(
				input=net, 
				index=7,
				ch_in=256,
				ch_out_squeeze=32,
				ch_out_expand=128
				)

	def get_slice4(self, input):
		with tf.name_scope("slice4"):
			#net = tf.nn.max_pool(
			net = self._pool(
				input,
				ksize=[1,3,3,1],
				strides=[1,2,2,1],
				padding="VALID",
				name="max_pool_slice4",
				data_format='NHWC'
			)

			return self.fire_module(
				input=net, 
				index=9,
				ch_in=256,
				ch_out_squeeze=48,
				ch_out_expand=192
				)


	def get_slice5(self, input):
		with tf.name_scope("slice5"):
			return self.fire_module(
				input=input, 
				index=10,
				ch_in=384,
				ch_out_squeeze=48,
				ch_out_expand=192
				)

	def get_slice6(self, input):
		with tf.name_scope("slice6"):
			return self.fire_module(
				input=input, 
				index=11,
				ch_in=384,
				ch_out_squeeze=64,
				ch_out_expand=256
				)

	def get_slice7(self, input):
		with tf.name_scope("slice7"):
			return self.fire_module(
				input=input, 
				index=12,
				ch_in=512,
				ch_out_squeeze=64,
				ch_out_expand=256
				)


	def forward(self, X):
		h = self.get_slice1(X)
		h_relu1 = h
		h = self.get_slice2(h)
		h_relu2 = h
		h = self.get_slice3(h)
		h_relu3 = h
		h = self.get_slice4(h)
		h_relu4 = h
		h = self.get_slice5(h)
		h_relu5 = h
		h = self.get_slice6(h)
		h_relu6 = h
		h = self.get_slice7(h)
		h_relu7 = h
		
		squeeze_outputs = namedtuple("SqueezeOutputs", ['relu1', 'relu2', 'relu3', 'relu4', 'relu5', 'relu6', 'relu7'])
		return squeeze_outputs(h_relu1, h_relu2, h_relu3, h_relu4, h_relu5, h_relu6, h_relu7)


class squeezenet1_1_full_maxpool(Network):
	def __init__(self, trainable=False, use_net_dropout=False, net_dropout_keep_prob=0.99, custom_net_weights=None, dtype=tf.float32):
		super(squeezenet1_1_full_maxpool, self).__init__(use_net_dropout, net_dropout_keep_prob, dtype=dtype)
		
		feature_path = os.path.join(DATA_DIR, "squeeze_pytorch_transposed_nonlinear_features.npy") 
		self.features = np.load(feature_path, allow_pickle=True).item() if custom_net_weights is None else custom_net_weights

		self.trainable = trainable
		self.features = make_trainable(self.features, trainable, 'squeezenet_full_avg')

	def _squeeze(self, input, index, ch_in, ch_out):
		return self._conv(
			input,
			self.features['{:d}.squeeze.weight'.format(index)],
			self.features['{:d}.squeeze.bias'.format(index)],
			w_shape=[1,1,ch_in,ch_out],
			padding='VALID',
			stride=[1,1,1,1]
		)

	def _expand(self, input, index, ch_in, ch_out):			
		e1x1 = self._conv(
			input, 
			self.features['{:d}.expand1x1.weight'.format(index)],
			self.features['{:d}.expand1x1.bias'.format(index)],
			w_shape=[1,1,ch_in,ch_out],
			padding='VALID',
			stride=[1,1,1,1]
		)
		e3x3 = self._conv(
			input, 
			self.features['{:d}.expand3x3.weight'.format(index)],
			self.features['{:d}.expand3x3.bias'.format(index)],
			w_shape=[3,3,ch_in,ch_out],
			padding='SAME',
			stride=[1,1,1,1]
		)
		
		if isinstance(input, tuple):
			return tuple((tf.concat([X, Y], 3) for X, Y in zip(e1x1, e3x3)))
		else:
			return tf.concat([e1x1, e3x3], 3)

	def _pool(self, input, ksize, strides, padding, name, data_format):
		def op(X):
			return tf.nn.max_pool(
				X,
				ksize=ksize,
				strides=strides,
				padding=padding,
				name=name,
				data_format=data_format
			)
		
		return for_each(input, op)
	
	def fire_module(self, input, index, ch_in, ch_out_squeeze, ch_out_expand):
		net = self._squeeze(input, index, ch_in, ch_out_squeeze)
		return self._expand(net, index, ch_out_squeeze, ch_out_expand)
		

	def get_slice1(self, input):
		with tf.name_scope("slice1"):
			o1 = input
			
			o2 = self._conv(
				input, 
				self.features['0.weight'],
				self.features['0.bias'],
				w_shape=[3,3,3,64],
				padding='VALID',
				stride=[1,2,2,1]
				)
				
			return o1, o2
		   
	def get_slice2(self, input):
		with tf.name_scope("slice2"):
			net = self._pool(
				input,
				ksize=[1,3,3,1],
				strides=[1,2,2,1],
				padding="VALID",
				name="max_pool_slice2",
				data_format='NHWC'
			)

			o1 = net = self.fire_module(
				input=net, 
				index=3,
				ch_in=64,
				ch_out_squeeze=16,
				ch_out_expand=64
				)

			o2 = self.fire_module(
				input=net, 
				index=4,
				ch_in=128,
				ch_out_squeeze=16,
				ch_out_expand=64
				)
			return o1, o2

	def get_slice3(self, input):
		with tf.name_scope("slice3"):
			net = self._pool(
				input,
				ksize=[1,3,3,1],
				strides=[1,2,2,1],
				padding="VALID",
				name="max_pool_slice3",
				data_format='NHWC'
			)

			o1 = net = self.fire_module(
				input=net, 
				index=6,
				ch_in=128,
				ch_out_squeeze=32,
				ch_out_expand=128
				)
			o2 = self.fire_module(
				input=net, 
				index=7,
				ch_in=256,
				ch_out_squeeze=32,
				ch_out_expand=128
				)
			return o1, o2

	def get_slice4(self, input):
		with tf.name_scope("slice4"):
			net = self._pool(
				input,
				ksize=[1,3,3,1],
				strides=[1,2,2,1],
				padding="VALID",
				name="max_pool_slice4",
				data_format='NHWC'
			)

			o1 = self.fire_module(
				input=net, 
				index=9,
				ch_in=256,
				ch_out_squeeze=48,
				ch_out_expand=192
				)
			return o1


	def get_slice5(self, input):
		with tf.name_scope("slice5"):
			o1 = self.fire_module(
				input=input, 
				index=10,
				ch_in=384,
				ch_out_squeeze=48,
				ch_out_expand=192
				)
			return o1

	def get_slice6(self, input):
		with tf.name_scope("slice6"):
			o1 = self.fire_module(
				input=input, 
				index=11,
				ch_in=384,
				ch_out_squeeze=64,
				ch_out_expand=256
				)
			return o1

	def get_slice7(self, input):
		with tf.name_scope("slice7"):
			o1 = self.fire_module(
				input=input, 
				index=12,
				ch_in=512,
				ch_out_squeeze=64,
				ch_out_expand=256
				)
			return o1


	def forward(self, X):
		o11, o12 = self.get_slice1(X)
		o21, o22 = self.get_slice2(o12)
		o31, o32 = self.get_slice3(o22)
		o41 = self.get_slice4(o32)
		o51 = self.get_slice5(o41)
		o61 = self.get_slice6(o51)
		o71 = self.get_slice7(o61)
		
		squeeze_outputs = namedtuple("SqueezeOutputs", ['o11', 'o12', 'o21', 'o22', 'o31', 'o32', 'o41', 'o51', 'o61', 'o71'])
		return squeeze_outputs(o11, o12, o21, o22, o31, o32, o41, o51, o61, o71)

		
class vgg16(Network):
	def __init__(self, trainable=False, use_net_dropout=False, net_dropout_keep_prob=0.99, custom_net_weights=None, dtype=tf.float32):
		super(vgg16, self).__init__(use_net_dropout, net_dropout_keep_prob, dtype=dtype)
		
		feature_path = os.path.join(DATA_DIR, "vgg16_pytorch_transposed_nonlinear_features.npy") 
		self.features = np.load(feature_path, allow_pickle=True).item() if custom_net_weights is None else custom_net_weights
		
		self.trainable = trainable
		self.features = make_trainable(self.features, trainable, 'vgg')
		
	def _pool(self, input, ksize, strides, padding, name, data_format):
		def op(X): 
			return tf.nn.max_pool(
				X,
				ksize=ksize,
				strides=strides,
				padding=padding,
				name=name,
				data_format=data_format
			)
		
		return for_each(input, op)
	
	def get_slice1(self, input):
		with tf.name_scope("slice1"):
			net = self._conv(
				input,
				self.features['0.weight'],
				self.features['0.bias'],
				w_shape=[3,3,3,64],
				padding='SAME',
				stride=[1,1,1,1]
			)

			return self._conv(
				net,
				self.features['2.weight'],
				self.features['2.bias'],
				w_shape=[3,3,64,64],
				padding='SAME',
				stride=[1,1,1,1]
			)

	def get_slice2(self, input):
		with tf.name_scope("slice2"):
			net = self._pool(
				input,
				ksize=[1,2,2,1],
				strides=[1,2,2,1],
				padding="VALID",
				name="max_pool_slice2",
				data_format='NHWC'
			)

			net = self._conv(
				net,
				self.features['5.weight'],
				self.features['5.bias'],
				w_shape=[3,3,64,128],
				padding='SAME',
				stride=[1,1,1,1]
			)

			return self._conv(
				net,
				self.features['7.weight'],
				self.features['7.bias'],
				w_shape=[3,3,128,128],
				padding='SAME',
				stride=[1,1,1,1]
			)

	def get_slice3(self, input):
		with tf.name_scope("slice3"):
			net = self._pool(
				input,
				ksize=[1,2,2,1],
				strides=[1,2,2,1],
				padding="VALID",
				name="max_pool_slice3",
				data_format='NHWC'
			)

			net = self._conv(
				net,
				self.features['10.weight'],
				self.features['10.bias'],
				w_shape=[3,3,128,256],
				padding='SAME',
				stride=[1,1,1,1]
			)

			net = self._conv(
				net,
				self.features['12.weight'],
				self.features['12.bias'],
				w_shape=[3,3,256,256],
				padding='SAME',
				stride=[1,1,1,1]
			)

			return self._conv(
				net,
				self.features['14.weight'],
				self.features['14.bias'],
				w_shape=[3,3,256,256],
				padding='SAME',
				stride=[1,1,1,1]
			)

	def get_slice4(self, input):
		with tf.name_scope("slice4"):
			net = self._pool(
				input,
				ksize=[1,2,2,1],
				strides=[1,2,2,1],
				padding="VALID",
				name="max_pool_slice4",
				data_format='NHWC'
			)

			net = self._conv(
				net,
				self.features['17.weight'],
				self.features['17.bias'],
				w_shape=[3,3,256,512],
				padding='SAME',
				stride=[1,1,1,1]
			)

			net = self._conv(
				net,
				self.features['19.weight'],
				self.features['19.bias'],
				w_shape=[3,3,512,512],
				padding='SAME',
				stride=[1,1,1,1]
			)

			return self._conv(
				net,
				self.features['21.weight'],
				self.features['21.bias'],
				w_shape=[3,3,512,512],
				padding='SAME',
				stride=[1,1,1,1]
			)

	def get_slice5(self, input):
		with tf.name_scope("slice5"):
			net = self._pool(
				input,
				ksize=[1,2,2,1],
				strides=[1,2,2,1],
				padding="VALID",
				name="max_pool_slice5",
				data_format='NHWC'
			)

			net = self._conv(
				net,
				self.features['24.weight'],
				self.features['24.bias'],
				w_shape=[3,3,512,512],
				padding='SAME',
				stride=[1,1,1,1]
			)

			net = self._conv(
				net,
				self.features['26.weight'],
				self.features['26.bias'],
				w_shape=[3,3,512,512],
				padding='SAME',
				stride=[1,1,1,1]
			)

			return self._conv(
				net,
				self.features['28.weight'],
				self.features['28.bias'],
				w_shape=[3,3,512,512],
				padding='SAME',
				stride=[1,1,1,1]
			)
  
	def forward(self, X):
		h = self.get_slice1(X)
		h_relu1_2 = h
		h = self.get_slice2(h)
		h_relu2_2 = h
		h = self.get_slice3(h)
		h_relu3_3 = h
		h = self.get_slice4(h)
		h_relu4_3 = h
		h = self.get_slice5(h)
		h_relu5_3 = h
		vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3', 'relu5_3'])
		out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)

		return out


class vgg16_full_avg(Network):
	def __init__(self, trainable=False, use_net_dropout=False, net_dropout_keep_prob=0.99, custom_net_weights=None, dtype=tf.float32):
		super(vgg16_full_avg, self).__init__(use_net_dropout, net_dropout_keep_prob, dtype=dtype)
		
		feature_path = os.path.join(DATA_DIR, "vgg16_pytorch_transposed_nonlinear_features.npy") 
		self.features = np.load(feature_path, allow_pickle=True).item() if custom_net_weights is None else custom_net_weights
		
		self.trainable = trainable
		self.features = make_trainable(self.features, trainable, 'vgg_full_avg')

	def _pool(self, input, ksize, strides, padding, name, data_format):
		def op(X): 
			return tf.nn.avg_pool(
				X,
				ksize=ksize,
				strides=strides,
				padding=padding,
				name=name,
				data_format=data_format
			)
		
		return for_each(input, op)
	
		
	def get_slice1(self, input):
		with tf.name_scope("slice1"):
			o1 = input
			
			o2 = net = self._conv(
				input,
				self.features['0.weight'],
				self.features['0.bias'],
				w_shape=[3,3,3,64],
				padding='SAME',
				stride=[1,1,1,1]
			)

			o3 = net = self._conv(
				net,
				self.features['2.weight'],
				self.features['2.bias'],
				w_shape=[3,3,64,64],
				padding='SAME',
				stride=[1,1,1,1]
			)
			
			return [o1, o2, o3]

		
	def get_slice2(self, input):
		with tf.name_scope("slice2"):
			net = self._pool(
				input,
				ksize=[1,2,2,1],
				strides=[1,2,2,1],
				padding="VALID",
				name="avg_pool_slice2",
				data_format='NHWC'
			)

			o1 = net = self._conv(
				net,
				self.features['5.weight'],
				self.features['5.bias'],
				w_shape=[3,3,64,128],
				padding='SAME',
				stride=[1,1,1,1]
			)

			o2 = net = self._conv(
				net,
				self.features['7.weight'],
				self.features['7.bias'],
				w_shape=[3,3,128,128],
				padding='SAME',
				stride=[1,1,1,1]
			)
			
			return [o1, o2]

	def get_slice3(self, input):
		with tf.name_scope("slice3"):
			net = self._pool(
				input,
				ksize=[1,2,2,1],
				strides=[1,2,2,1],
				padding="VALID",
				name="avg_pool_slice3",
				data_format='NHWC'
			)

			o1 = net = self._conv(
				net,
				self.features['10.weight'],
				self.features['10.bias'],
				w_shape=[3,3,128,256],
				padding='SAME',
				stride=[1,1,1,1]
			)

			o2 = net = self._conv(
				net,
				self.features['12.weight'],
				self.features['12.bias'],
				w_shape=[3,3,256,256],
				padding='SAME',
				stride=[1,1,1,1]
			)

			o3 = net = self._conv(
				net,
				self.features['14.weight'],
				self.features['14.bias'],
				w_shape=[3,3,256,256],
				padding='SAME',
				stride=[1,1,1,1]
			)
			
			return [o1, o2, o3]


	def get_slice4(self, input):
		with tf.name_scope("slice4"):
			net = self._pool(
				input,
				ksize=[1,2,2,1],
				strides=[1,2,2,1],
				padding="VALID",
				name="avg_pool_slice4",
				data_format='NHWC'
			)

			o1 = net = self._conv(
				net,
				self.features['17.weight'],
				self.features['17.bias'],
				w_shape=[3,3,256,512],
				padding='SAME',
				stride=[1,1,1,1]
			)

			o2 = net = self._conv(
				net,
				self.features['19.weight'],
				self.features['19.bias'],
				w_shape=[3,3,512,512],
				padding='SAME',
				stride=[1,1,1,1]
			)

			o3 = net = self._conv(
				net,
				self.features['21.weight'],
				self.features['21.bias'],
				w_shape=[3,3,512,512],
				padding='SAME',
				stride=[1,1,1,1]
			)
			
			return [o1, o2, o3]
			
	def get_slice5(self, input):
		with tf.name_scope("slice5"):
			net = self._pool(
				input,
				ksize=[1,2,2,1],
				strides=[1,2,2,1],
				padding="VALID",
				name="avg_pool_slice5",
				data_format='NHWC'
			)

			o1 = net = self._conv(
				net,
				self.features['24.weight'],
				self.features['24.bias'],
				w_shape=[3,3,512,512],
				padding='SAME',
				stride=[1,1,1,1]
			)

			o2 = net = self._conv(
				net,
				self.features['26.weight'],
				self.features['26.bias'],
				w_shape=[3,3,512,512],
				padding='SAME',
				stride=[1,1,1,1]
			)

			o3 = net = self._conv(
				net,
				self.features['28.weight'],
				self.features['28.bias'],
				w_shape=[3,3,512,512],
				padding='SAME',
				stride=[1,1,1,1]
			)
  
			return [o1, o2, o3]

	def forward(self, X):
		o11, o12, o13 = self.get_slice1(X)
		o21, o22 = self.get_slice2(o13)
		o31, o32, o33 = self.get_slice3(o22)
		o41, o42, o43 = self.get_slice4(o33)
		o51, o52, o53 = self.get_slice5(o43)
		vgg_outputs = namedtuple("VggOutputs", ['o11', 'o12', 'o13', 'o21', 'o22', 'o31', 'o32', 'o33', 'o41', 'o42', 'o43', 'o51', 'o52', 'o53'])
		out = vgg_outputs(o11, o12, o13, o21, o22, o31, o32, o33, o41, o42, o43, o51, o52, o53)

		return out
