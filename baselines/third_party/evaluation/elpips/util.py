import tensorflow as tf
import numpy as np

	
def switch_case_cond(cases, default_case):
	if cases:
		condition, effect = cases[0]
		return tf.cond(condition, effect, lambda: switch_case_cond(cases[1:], default_case))
	return default_case()

def switch_case_where(cases, default_case):
	if cases:
		condition, effect = cases[0]
		return tf.where(condition, effect, switch_case_where(cases[1:], default_case))
	return default_case


def np_dtype(tf_dtype):
	if tf_dtype == tf.float32:
		return np.float32
	if tf_dtype == tf.float64:
		return np.float64
	raise Exception('Unsupported dtype')

def f32_to_dtype(x, dtype):
	if dtype == tf.float32:
		return x
	return tf.cast(x, dtype)
	

def as_tuple(x):
	'''Formats x as a tuple. If x is already a tuple returns it as is, otherwise returns a 1-tuple containing x.'''
	if isinstance(x, tuple):
		return x
	else:
		return (x,)

def for_each(x, func):
	'''Runs 'func' for x, or each item of x if x is a tuple. Returns the results in the same format.'''
	if isinstance(x, tuple):
		return tuple((func(s) for s in x))
	else:
		return func(x)

def for_tuple(x, func):
	'''Runs 'func' for as_tuple(x). Returns the results in the original format. Assumes 'func' returns tuple when given tuple.'''
	if isinstance(x, tuple):
		return func(x)
	else:
		return func(as_tuple(x))[0]
