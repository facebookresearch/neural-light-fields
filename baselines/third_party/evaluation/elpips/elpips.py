import tensorflow as tf
import numpy as np
import itertools 
import numbers

from . import networks
from . import pnetlin
from . util import switch_case_cond, switch_case_where, for_each, as_tuple


### Configuring E-LPIPS.
	
class Config:
	def __init__(self):
		self.metric = 'vgg_ensemble'
		
		self.enable_dropout = True
		self.dropout_keep_prob = 0.99
		
		self.enable_offset = True
		self.offset_max = 7
		
		self.enable_flip = True
		self.enable_swap = True
		self.enable_color_permutation = True
		
		self.enable_color_multiplication = True
		self.color_multiplication_mode = 'color' # 'brightness'
		
		self.enable_scale = True
		self.set_scale_levels(8)
		
		# Enables cropping instead of padding. Faster but may randomly skip edges of the input.
		self.fast_and_approximate = False
		
		self.batch_size = 1 
		self.average_over = 1  # How many runs to average over.
	
		self.dtype = tf.float32
		
	def set_scale_levels(self, num_scales):
		# Crop_size / num_scales should be at least 64.
		self.num_scales = num_scales
		self.scale_probabilities = [1.0 / float(i)**2 for i in range(1, self.num_scales + 1)]
		
	def set_scale_levels_by_image_size(self, image_h, image_w):
		'''Sets the number of scale levels based on the image size.'''
		image_size = min(image_h, image_w)
		self.set_scale_levels(max(1, image_size // 64))
		
	def validate(self):
		assert self.metric in ('vgg_ensemble', 'vgg', 'squeeze', 'squeeze_ensemble_maxpool')
		assert self.color_multiplication_mode in ('color', 'brightness')
		assert self.num_scales == len(self.scale_probabilities)
		

### Ensemble sampling and application to images.

def sample_ensemble(config):
	'''Samples a random transformation according to the config.
	   Uses Latin Hypercube Sampling when batch size is greater than 1.'''
	
	N = config.batch_size

	# Offset randomization.
	offset_xy = tf.random_uniform([N, 2], minval=0, maxval=config.offset_max + 1, dtype=tf.int32)
			
	# Sample scale level.
	cumulative_sum = np.cumsum(config.scale_probabilities)
	u = cumulative_sum[-1] * tf.random_uniform([])
		
	scale_level = switch_case_cond(
		[(tf.less(u, x), (lambda j=i: tf.constant(j+1))) for i, x in enumerate(cumulative_sum[:-1])],
		lambda: tf.constant(len(cumulative_sum))
	)
	scale_level = tf.clip_by_value(scale_level, 1, config.num_scales)		
	
	# Scale randomization.
	scale_offset_xy = tf.random_uniform([2], minval=0, maxval=scale_level, dtype=tf.int32)
	
	# Sample flips.
	flips = tf.range((N + 3)//4*4, dtype=tf.int32)
	flips = tf.floormod(flips, 4)
	flips = tf.random_shuffle(flips)
	flips = flips[:N]
		
	# Sample transposing.
	swap_xy = tf.random_uniform([], minval=0, maxval=2, dtype=tf.int32)

	# Color multiplication.
	def sample_colors():
		color = tf.random_uniform([N], minval=0.0, maxval=1.0, dtype=config.dtype)
		color += tf.cast(tf.range(N), config.dtype)
		color /= tf.cast(N, config.dtype)
		return tf.random_shuffle(color)
	colors_r = tf.reshape(sample_colors(), [-1, 1, 1, 1])
	colors_g = tf.reshape(sample_colors(), [-1, 1, 1, 1])
	colors_b = tf.reshape(sample_colors(), [-1, 1, 1, 1])
	
	if config.color_multiplication_mode == 'color':
		color_factors = tf.concat([colors_r, colors_g, colors_b], axis=3)
	elif config.color_multiplication_mode == 'brightness':
		color_factors = tf.concat([colors_r, colors_r, colors_r], axis=3)
	else:
		raise Exception('Unknown color multiplication mode.')
	
	color_factors = 0.2 + 0.8 * color_factors
	
	# Sample permutations.
	permutations = np.asarray(list(itertools.permutations(range(3))), dtype=np.int32)
	repeat_count = (N + len(permutations) - 1) // len(permutations)
	permutations = tf.tile(tf.convert_to_tensor(permutations), tf.constant([repeat_count, 1]))
	permutations = tf.reshape(tf.random_shuffle(permutations)[:N, :], [-1])
			
	base_indices = 3 * tf.reshape(tf.tile(tf.reshape(tf.range(N), [-1, 1]), [1, 3]), [-1]) # [0, 0, 0, 3, 3, 3, 6, 6, 6, ...]
	permutations += base_indices
						
	return (offset_xy, flips, swap_xy, color_factors, permutations, scale_offset_xy, scale_level)
	
	
def apply_ensemble(config, sampled_ensemble_params, X):
	'''Applies the sampled random transformation to image X.'''
	offset_xy, flips, swap_xy, color_factors, permutations, scale_offset_xy, scale_level = sampled_ensemble_params
	
	shape = tf.shape(X)
	N, H, W, C = shape[0], shape[1], shape[2], shape[3]

	# Resize image.
	if config.enable_scale:		
		def downscale_nx_impl(image, scale):
			shape = tf.shape(image)
			N, H, W, C = shape[0], shape[1], shape[2], shape[3]
		
			image = tf.reshape(image, tf.stack([N, H//scale, scale, W//scale, scale, C]))
			image = tf.reduce_mean(image, axis=[2, 4])
			return image
			
		def downscale_1x():
			return X
		
		def downscale_nx():
			nonlocal X

			if config.fast_and_approximate:
				# Crop to a multiple of scale_level.
				crop_left = scale_offset_xy[1]
				full_width = (W - scale_level + 1) // scale_level * scale_level
				crop_right = crop_left + full_width
			
				crop_bottom = scale_offset_xy[0]
				full_height = (H - scale_level + 1) // scale_level * scale_level
				crop_top = crop_bottom + full_height
				
				X = X[:, crop_bottom:crop_top, crop_left:crop_right, :]
			else:
				# Pad to a multiple of scale_level.
				pad_left = scale_offset_xy[1]
				full_width = (scale_level - 1 + W + scale_level - 1) // scale_level * scale_level
				pad_right = full_width - W - pad_left
			
				pad_bottom = scale_offset_xy[0]
				full_height = (scale_level - 1 + H + scale_level - 1) // scale_level * scale_level
				pad_top = full_height - H - pad_bottom
				
				X = tf.pad(X, [(0, 0), (pad_bottom, pad_top), (pad_left, pad_right), (0, 0)], 'reflect')
			return downscale_nx_impl(X, scale_level)
		
		X = tf.cond(tf.equal(scale_level, 1), downscale_1x, downscale_nx)
	
	# Pad image.
	if config.enable_offset:
		L = []

		shape = tf.shape(X)
		N, H, W, C = shape[0], shape[1], shape[2], shape[3]

		for i in range(config.batch_size):
			if config.fast_and_approximate:
				# Crop.
				crop_bottom = offset_xy[i, 0]
				crop_left = offset_xy[i, 1]
				crop_top = H - config.offset_max + crop_bottom
				crop_right = W - config.offset_max + crop_left
			
				L.append(X[i, crop_bottom:crop_top, crop_left:crop_right, :])
			else:
				# Pad.
				pad_bottom = config.offset_max - offset_xy[i, 0]
				pad_left = config.offset_max - offset_xy[i, 1]
				pad_top = offset_xy[i, 0]
				pad_right = offset_xy[i, 1]
			
				L.append(tf.pad(X[i,:,:,:], tf.convert_to_tensor([(pad_bottom, pad_top), (pad_left, pad_right), (0, 0)], dtype=np.int32), 'reflect'))
		X = tf.stack(L, axis=0)
		
	# Apply flips.		
	if config.enable_flip:
		def flipX(X):
			return X[:, :, ::-1, :]
		def flipY(X):
			return X[:, ::-1, :, :]
		def flipXandY(X):
			return X[:, ::-1, ::-1, :]
		X = switch_case_where(
			[(tf.equal(flips, 0), flipX(X)),
			(tf.equal(flips, 1), flipY(X)),
			(tf.equal(flips, 2), flipXandY(X))],
			X
		)
	
	# Apply transpose.
	if config.enable_swap:
		def swapXY(X):
			return tf.transpose(X, perm=tf.constant((0, 2, 1, 3)))
		X = tf.cond(tf.equal(swap_xy, 1), lambda: swapXY(X), lambda: X)
				
	# Apply color permutations.
	if config.enable_color_permutation:
		def permuteColor(X, perms):
			shape = tf.shape(X)
			N, H, W, C = shape[0], shape[1], shape[2], shape[3]

			X = tf.transpose(X, [0, 3, 1, 2]) # NHWC -> NCHW
			X = tf.reshape(X, [N * C, H, W])  # (NC)HW
			X = tf.gather(X, perms)           # Permute rows (colors)
			X = tf.reshape(X, [N, C, H, W])   # NCHW
			X = tf.transpose(X, [0, 2, 3, 1]) # NCHW -> NHWC
			return X

		X = permuteColor(X, permutations)
	
	if config.enable_color_multiplication:
		X = X * tf.reshape(color_factors, [config.batch_size, 1, 1, 3])

	return X
	
	
### E-LPIPS implementation.
	
class Metric:
	def __init__(self, config,
	             back_prop=True,
	             trainable=False, use_lpips_dropout=False,
	             custom_lpips_weights=None, custom_net_weights=None,
				 custom_sample_ensemble=None):
		'''Perceptual image distance metric.
		
		   PARAMS:
		       config: Metric configuration. One of: elpips.elpips_vgg(), elpips.elpips_squeeze_maxpool(), elpips.lpips_vgg(), elpips.lpips_squeeze(). 
			   back_prop: Whether to store data for back_prop.
			   
			   trainable: Whether to make weights trainable. Options: 'lpips', 'net', 'both'.
			   use_lpips_dropout: Whether to use dropout for activation differences. Potentially useful for training LPIPS weights.
			   custom_lpips_weights: Custom NumPy array of LPIPS weights to use.
			   custom_net_weights: Custom NumPy array of internal network weights to use. (For VGG, SqueezeNet, etc.)
			   custom_sample_ensemble: Replace the input transformation sampling with something else. May be useful for e.g. variance reduction or deterministic input transformations.
		'''
		assert trainable in ('lpips', 'net', 'both', False)
		
		if trainable and back_prop != True:
			raise Exception('Enable back_prop for training.')
		
		config.validate()
		self.config = config
		
		if config.metric in ('vgg', 'squeeze', 'vgg_ensemble', 'squeeze_ensemble_maxpool'):
			self.network = pnetlin.PNetLin(
				pnet_type=config.metric,
				use_lpips_dropout=use_lpips_dropout,
				use_net_dropout=self.config.enable_dropout,
				net_dropout_keep_prob=self.config.dropout_keep_prob,
				trainable=trainable,
				custom_lpips_weights=custom_lpips_weights,
				custom_net_weights=custom_net_weights,
				dtype=config.dtype
			)
		else:
			raise Exception('Unknown metric type \'{}\''.format(config.metric))
			
		self.back_prop = back_prop
		self.sample_ensemble = custom_sample_ensemble if custom_sample_ensemble else sample_ensemble
		
	def forward(self, image, reference):
		'''Evaluates distances between images in 'image' and 'reference' (data in NHWC order).
		   Returns an N-element distance vector.
		   
		   If 'image' is a tuple, evaluates all the images in the tuple with the same input transformations
		   and dropout as 'reference'. A different set of input transformations for each would result in
		   unnecessary uncertainty in determining which of the images is closest to the reference. The
		   returned value is a tuple of N-element distance vectors.'''
		  
		if isinstance(image, list):
			raise Exception('Parameter \'image\' must be a tensor or a tuple of tensors.')
		
		image_in = as_tuple(image)
		
		def cond(i, loss_sum):
			return tf.less(i, tf.cast(self.config.average_over, tf.int32))
		
		def body(i, loss_sum):
			ensemble = self.sample_ensemble(self.config)
			
			ensemble_X = for_each(image_in, lambda X: apply_ensemble(self.config, ensemble, X))
			ensemble_X = for_each(ensemble_X, lambda X: 2.0 * X - 1.0)

			ensemble_R = apply_ensemble(self.config, ensemble, reference)			
			ensemble_R = 2.0 * ensemble_R - 1.0
			
			loss = self.network.forward(ensemble_X, ensemble_R)
			loss_sum += tf.stack(loss, axis=0)
			
			loss_sum.set_shape([len(image_in), self.config.batch_size])
			
			return i+1, loss_sum

		if isinstance(self.config.average_over, numbers.Number) and self.config.average_over == 1:
			# Skip tf.while for trivial single iterations.
			_, loss_sum = body(0, tf.zeros([len(image_in), self.config.batch_size], dtype=self.config.dtype))
		else:
			# Run multiple times for any other average_over count.
			_, loss_sum = tf.while_loop(cond, body, (0, tf.zeros([len(image_in), self.config.batch_size], dtype=self.config.dtype)), back_prop=self.back_prop)
			loss_sum /= tf.cast(self.config.average_over, self.config.dtype)

		
		if isinstance(image, tuple):
			return tuple((loss_sum[i, :] for i in range(len(image))))
		else:
			return tf.reshape(loss_sum, [self.config.batch_size])
