
import tensorflow as tf
from tensorlayer.layers import PadLayer,Conv2d,UpSampling2dLayer,InputLayer,ConcatLayer
import numpy as np

def upsampling_factor_padding(h_res,w_res):
    
    res_temp = h_res
    py =[res_temp%2]
    while res_temp!=1:
        res_temp = res_temp//2
        py.append(res_temp%2)
    
    del py[-1]    
    py = np.flip(py)
    
    res_temp = w_res
    px =[res_temp%2]
    while res_temp!=1:
        res_temp = res_temp//2
        px.append(res_temp%2)
    
    del px[-1]    
    px = np.flip(px)
    
    lx = len(px)
    ly = len(py)
    up_x = 2*np.ones((lx))
    up_y = 2*np.ones((ly))
    
    if lx > ly:
        py = np.append(py,[0]*(lx-ly))
        up_y = np.append(up_y,[1]*(lx-ly))
    
        
    if ly > lx:
        px = np.append(px,[0]*(ly-lx))
        up_x = np.append(up_x,[1]*(ly-lx))
    
    return px,py,up_x,up_y


def gen_flow(batch_input,num_out):
    padding_d  = [[0,0],[1,1],[1,1],[0,0]]
    batch_input = PadLayer(batch_input,padding_d,"REFLECT")
    network = Conv2d(batch_input, n_filter=num_out, filter_size=(3, 3),strides=(1, 1), act = tf.tanh, padding='VALID',W_init=tf.random_normal_initializer(0, 0.02),b_init = tf.constant_initializer(value=0.0))
    return network.outputs

def conv_layer(batch_input, out_channels,padding_d,fs):
    batch_input = PadLayer(batch_input,padding_d,"REFLECT")
    network = Conv2d(batch_input, n_filter=out_channels, filter_size=(fs,fs),strides=(1, 1), act=tf.nn.leaky_relu, padding='VALID',W_init=tf.random_normal_initializer(0, 0.02),b_init = tf.constant_initializer(value=0.0))
    return network


def Flow(input_coordinates,h_res,w_res,num_out,ngf,min_,max_):

    
    # we calculated the amount of padding for each layer and 
    # the total number of upsampling in each dimension to output the resolution h_res*w_res.
    padx,pady,up_x,up_y = upsampling_factor_padding(h_res,w_res)
    
    num_l = len(padx) 
    layer_specs = [ngf*16, ngf*16 , ngf*16 , ngf*8 , ngf*8 , ngf*8 , ngf*4 ]
    layer_specs.extend([ngf*4]*(num_l-len(layer_specs))) 
    
    
    # coordconv layer
    coordconv    = tf.constant([[[[min_, min_],
                                  [max_, min_]], 
                                 [[min_, max_], 
                                  [max_, max_]]]],dtype=tf.float32)
 
    coordconv_tl = InputLayer(tf.tile(coordconv,[input_coordinates.shape[0],1,1,1]))
    output = InputLayer(input_coordinates)

    for num,num_filter in enumerate(layer_specs):

        with tf.variable_scope("layer_%d" % (num)):
            
            upsampled =  UpSampling2dLayer(output,(up_y[num],up_x[num]),True,0,True)           
            if num == 0:
                  padding  = [[0,0],[0,pady[num]],[0,padx[num]],[0,0]]
                  output = conv_layer(upsampled,num_filter,padding,1)
                  coordconv_tl = PadLayer(coordconv_tl,padding,"REFLECT")
                  # concatenating the coordconv layer
                  output = ConcatLayer([output,coordconv_tl],-1)
            else:
                  padding  = [[0,0],[1,1 + pady[num]],[1,1 + padx[num]],[0,0]]
                  output = conv_layer(upsampled,num_filter,padding,3)
                  
    with tf.variable_scope("outputs_flows"):
        flows = gen_flow(output,num_out)

        
    return  flows

