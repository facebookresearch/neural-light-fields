from . import elpips
import tensorflow as tf


Config = elpips.Config

def elpips_squeeze_maxpool(batch_size=1, n=1,dtype=tf.float32):
	'''E-LPIPS-SQUEEZENET-MAXPOOL configuration with all input transformations and dropout with p=0.99. Returns the average result over n samples. Does not use average pooling since SqueezeNet would require re-training for that. Experimental!
	Warning: Some versions of TensorFlow might have bugs that make n > 1 problematic due to the tf.while_loop used internally when n > 1.
'''
	config = Config()
	config.metric = 'squeeze_ensemble_maxpool'
	config.batch_size = batch_size
	config.average_over = n
	config.dtype = dtype
	
	return config

def elpips_vgg(batch_size=1, n=1, dtype=tf.float32):
	'''E-LPIPS-VGG configuration with all input transformations and dropout with p=0.99. Returns the average result over n samples.
	Warning: Some versions of TensorFlow might have bugs that make n > 1 problematic due to the tf.while_loop used internally when n > 1.'''
	config = Config()
	config.metric = 'vgg_ensemble'
	config.batch_size = batch_size
	config.average_over = n
	config.dtype = dtype
			
	return config

def lpips_squeeze(batch_size=1, dtype=tf.float32):
	'''Plain LPIPS-SQUEEZE configuration.'''
	config = Config()
	config.metric = 'squeeze'
	config.enable_dropout = False
	config.enable_offset = False
	config.enable_flip = False
	config.enable_swap = False
	config.enable_color_permutation = False
	config.enable_color_multiplication = False
	config.enable_scale = False
	config.batch_size = batch_size
	config.average_over = 1
	config.dtype = dtype
	return config

def lpips_vgg(batch_size=1, dtype=tf.float32):
	'''No augmentations.'''
	config = Config()
	config.metric = 'vgg'
	config.enable_dropout = False
	config.enable_offset = False
	config.enable_flip = False
	config.enable_swap = False
	config.enable_color_permutation = False
	config.enable_color_multiplication = False
	config.enable_scale = False
	config.batch_size = batch_size
	config.average_over = 1
	config.dtype = dtype
	return config


def get_config(config_name, batch_size=1, n=1, dtype=tf.float32):
	'''Returns a config name by string.'''
	
	if config_name == 'elpips_vgg':
		return elpips_vgg(batch_size=batch_size, n=n, dtype=dtype)
	elif config_name == 'elpips_squeeze_maxpool':
		config = elpips_squeeze_maxpool(batch_size=batch_size, n=n, dtype=dtype)
		return config
	elif config_name == 'lpips_squeeze':
		return lpips_squeeze(batch_size=batch_size, dtype=dtype)
	elif config_name == 'lpips_vgg':
		return lpips_vgg(batch_size=batch_size, dtype=dtype)
	else:
		raise Exception('Unknown config_name.')
		
	
Metric = elpips.Metric