# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Copyright 2017 Modifications Clement Godard.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import absolute_import, division, print_function
import tensorflow as tf

def bilinear_sampler_2d(input_images, x_offset ,wrap_mode='border', name='bilinear_sampler', **kwargs):
    def _repeat(x, n_repeats):
        with tf.variable_scope('_repeat'):
            rep = tf.tile(tf.expand_dims(x, 1), [1, n_repeats])
            return tf.reshape(rep, [-1])

    def _interpolate(im, x, y):
        with tf.variable_scope('_interpolate'):

            # handle both texture border types
            _edge_size = 0
            if _wrap_mode == 'border':
                _edge_size = 1
                im = tf.pad(im, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT')
                x = x + _edge_size
                y = y + _edge_size
            elif _wrap_mode == 'edge':
                _edge_size = 0
            else:
                return None

            x = tf.clip_by_value(x, 1.0,  _width_f - 2 + 2 * _edge_size)
            y = tf.clip_by_value(y, 1.0,  _height_f - 2 + 2 * _edge_size)

            x0_f = tf.floor(x)
            y0_f = tf.floor(y)
            x1_f = x0_f + 1
            y1_f = y0_f + 1

            x0 = tf.cast(x0_f, tf.int32)
            y0 = tf.cast(y0_f, tf.int32)
            
            x1 = tf.cast(tf.minimum(x1_f,  _width_f - 1 + 2 * _edge_size), tf.int32)
            y1 = tf.cast(tf.minimum(y1_f,  _height_f - 1 + 2 * _edge_size), tf.int32)

            dim2 = (_width + 2 * _edge_size)
            dim1 = (_width + 2 * _edge_size) * (_height + 2 * _edge_size)
            
            base = _repeat(tf.range(_num_batch) * dim1, _height * _width)
            base_y0 = base + y0 * dim2
            base_y1 = base + y1 * dim2
            
            idx_00 = base_y0 + x0
            idx_01 = base_y0 + x1
            
            idx_10 = base_y1 + x0
            idx_11 = base_y1 + x1

            im_flat = tf.reshape(im, tf.stack([-1, _num_channels]))

            pix_00 = tf.gather(im_flat, idx_00)
            pix_01 = tf.gather(im_flat, idx_01)
            pix_10 = tf.gather(im_flat, idx_10)
            pix_11 = tf.gather(im_flat, idx_11)
            

            weight_x0 = tf.expand_dims(x - x0_f, 1)
            weight_y0 = tf.expand_dims(y - y0_f, 1)
            weight_x1 = tf.expand_dims(x1_f - x, 1)
            weight_y1 = tf.expand_dims(y1_f - y, 1)

            
            weight = pix_00 * weight_y1*weight_x1 + pix_01 *weight_y1*weight_x0 + pix_10 * weight_y0*weight_x1 + pix_11 *weight_y0* weight_x0

            return weight

    def _transform(input_images, x_offset):
        with tf.variable_scope('transform'):
            # grid of (x_t, y_t, 1), eq (1) in ref [1]
            x_t, y_t = tf.meshgrid(tf.linspace(0.0,   _width_f - 1.0,  _width),
                                   tf.linspace(0.0 , _height_f - 1.0 , _height))

            x_t_flat = tf.reshape(x_t, (1, -1))
            y_t_flat = tf.reshape(y_t, (1, -1))

            x_t_flat = tf.tile(x_t_flat, tf.stack([_num_batch, 1]))
            y_t_flat = tf.tile(y_t_flat, tf.stack([_num_batch, 1]))

            x_t_flat = tf.reshape(x_t_flat, [-1])
            y_t_flat = tf.reshape(y_t_flat, [-1])
         
            x_shift = x_offset[:,:,:,0:1]
            y_shift = x_offset[:,:,:,1:2]
            
            x_t_flat = x_t_flat + tf.reshape(x_shift, [-1])  * _width_f 
            y_t_flat = y_t_flat + tf.reshape(y_shift, [-1]) * _width_f
          


            input_transformed = _interpolate(input_images, x_t_flat, y_t_flat)

            output = tf.reshape(
                input_transformed, tf.stack([_num_batch, _height, _width, _num_channels]))
            return output

    with tf.variable_scope(name):
        _num_batch    = tf.shape(input_images)[0]
        _height       = tf.shape(input_images)[1]
        _width        = tf.shape(input_images)[2]
        _num_channels = tf.shape(input_images)[3]

        _height_f = tf.cast(_height, tf.float32)
        _width_f  = tf.cast(_width,  tf.float32)

        _wrap_mode = wrap_mode

        output = _transform(input_images, x_offset)
        return output
