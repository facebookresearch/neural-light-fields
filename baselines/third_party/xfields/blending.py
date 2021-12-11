# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 14:03:37 2020

@author: mbemana
"""


import tensorflow as tf
from bilinear_sampler import bilinear_sampler_2d
epsilon = 0.00001

def Blending_train(inputs,
                   Neighbors,
                   flows,
                   albedo,
                   h_res,w_res,
                   args):
    
    
    
    if args.type == ['light','view','time']:
        
        light_flow =  flows[:1,:,:,0:2]
        view_flow  =  flows[:1,:,:,2:4]
        time_flow  =  flows[:1,:,:,4:6]
        
        light_flow_neighbor =  flows[1:,:,:,0:2]
        view_flow_neighbor  =  flows[1:,:,:,2:4]
        time_flow_neighbor  =  flows[1:,:,:,4:6]
        
        coord_in       = inputs[:1,::]
        coord_neighbor = inputs[1:,::]
     
        delta = tf.tile(coord_in - coord_neighbor,[1,h_res,w_res,1])
        delta_light = delta[:,:,:,0:1]
        delta_view  = delta[:,:,:,1:2]
        delta_time  = delta[:,:,:,2:3]
    
    
        flag = tf.dtypes.cast(tf.abs(delta_light)>0,tf.float32)
        offset_forward = delta_light*light_flow + delta_view*view_flow + delta_time*time_flow 
        shading = flag*Neighbors/albedo + (1-flag)*Neighbors
    
        warped_shading    = bilinear_sampler_2d(shading,offset_forward)
        warped_view_flow  = bilinear_sampler_2d(view_flow_neighbor,offset_forward)
        warped_time_flow  = bilinear_sampler_2d(time_flow_neighbor,offset_forward)
        warped_light_flow = bilinear_sampler_2d(light_flow_neighbor,offset_forward)
           
        warped_image = flag*warped_shading*albedo  + (1-flag)*warped_shading
        
        offset_backward = delta_light*warped_light_flow + delta_view*warped_view_flow+ delta_time*warped_time_flow 
         
        dist              = tf.reduce_sum(tf.abs(offset_forward-offset_backward),-1,keepdims=True)
        weight            = tf.exp(-args.sigma*w_res*dist)
        weight_normalized = weight/(tf.reduce_sum(weight,0,keepdims=True) + epsilon)
        interpolated      = tf.reduce_sum(tf.multiply(warped_image,weight_normalized),0,keepdims=True)
        
        
    elif args.type == ['view'] or args.type == ['light'] or args.type == ['time']:
            
        
        flow           = flows[:1,::]
        Neighbors_flow = flows[1:,::]
        
        coord_in       = inputs[:1,::]
        coord_neighbor = inputs[1:,::]
     
        delta = tf.tile(coord_in - coord_neighbor,[1,h_res,w_res,1])
        
        offset_forward = delta*flow
        shading        = Neighbors/albedo
        
        warped_shading = bilinear_sampler_2d(shading,offset_forward)
        warped_flow    = bilinear_sampler_2d(Neighbors_flow,offset_forward)
    
    
        warped_image    = warped_shading*albedo
        offset_backward =  delta*warped_flow
    
        dist              = tf.reduce_sum(tf.abs(offset_backward-offset_forward),-1,keepdims=True)
        weight            = tf.exp(-args.sigma*w_res*dist)
        weight_normalized = weight/(tf.reduce_sum(weight,0,keepdims=True)+ epsilon)
        interpolated      = tf.reduce_sum(tf.multiply(warped_image,weight_normalized),0,keepdims=True)
        
    
    return interpolated


def Blending_test(coord_in,
                  coord_neighbor,
                  Neighbors_im,
                  Neighbors_flow,
                  flows,
                  albedo,
                  h_res,w_res,
                  args):
  
    if args.type == ['light','view','time']:

    
        light_flow =  flows[:1,:,:,0:2]
        view_flow  =  flows[:1,:,:,2:4]
        time_flow  =  flows[:1,:,:,4:6]
        
        light_flow_neighbor  = Neighbors_flow[:,:,:,0:2]
        view_flow_neighbor   = Neighbors_flow[:,:,:,2:4]
        time_flow_neighbor   = Neighbors_flow[:,:,:,4:6]
            
        delta = tf.tile(coord_in - coord_neighbor,[1,h_res,w_res,1])
        delta_light = delta[:,:,:,0:1]
        delta_view  = delta[:,:,:,1:2]
        delta_time  = delta[:,:,:,2:3]
    
    
        forward_shading = delta_view*view_flow + delta_time*time_flow + delta_light*light_flow 
        forward_albedo  = delta_view*view_flow + delta_time*time_flow
        shading         = Neighbors_im/albedo
    
    
        warped_shading    = bilinear_sampler_2d(shading             ,forward_shading)
        warped_view_flow  = bilinear_sampler_2d(view_flow_neighbor  ,forward_shading)
        warped_time_flow  = bilinear_sampler_2d(time_flow_neighbor  ,forward_shading)
        warped_light_flow = bilinear_sampler_2d(light_flow_neighbor ,forward_shading)
        warped_albedo     = bilinear_sampler_2d(albedo              ,forward_albedo)
    
        
        backward_shading = delta_view*warped_view_flow + delta_time*warped_time_flow + delta_light*warped_light_flow 
        backward_albedo  = delta_view*warped_view_flow + delta_time*warped_time_flow 
    
        
        
        dist_shading       = tf.reduce_sum(tf.abs(backward_shading-forward_shading),-1,keepdims=True)
        weight_shading     = tf.exp(-args.sigma*w_res*dist_shading)
        weight_occ_shading = weight_shading/(tf.reduce_sum(weight_shading,0,keepdims=True) + epsilon)
        multiplied         = tf.multiply(warped_shading,weight_occ_shading)
        novel_shading      = tf.reduce_sum(multiplied,0,keepdims=True)
        
        
        dist_albedo        = tf.reduce_sum(tf.abs(backward_albedo-forward_albedo),-1,keepdims=True)
        weight_albedo      = tf.exp(-args.sigma*w_res*dist_albedo)
        weight_albedo_sum  = tf.reduce_sum(weight_albedo,0,keepdims=True) + epsilon
        weight_occ_albedo  = weight_albedo/(weight_albedo_sum)
        multiplied         = tf.multiply(warped_albedo,weight_occ_albedo)
        novel_albedo       = tf.reduce_sum(multiplied,0,keepdims=True)
        
        
        interpolated = novel_shading*novel_albedo
        
        
        
    elif args.type == ['view'] or args.type == ['light'] or args.type == ['time']:
            
        flow           = flows[:1,::]
        delta = tf.tile(coord_in - coord_neighbor,[1,h_res,w_res,1])
        
        offset_forward = delta*flow
        shading        = Neighbors_im/albedo
        
        warped_shading = bilinear_sampler_2d(shading,offset_forward)
        warped_flow    = bilinear_sampler_2d(Neighbors_flow,offset_forward)
    
    
        warped_image    = warped_shading*albedo
        offset_backward =  delta*warped_flow
    
        dist              = tf.reduce_sum(tf.abs(offset_backward-offset_forward),-1,keepdims=True)
        weight            = tf.exp(-args.sigma*w_res*dist)
            
        weight_normalized = weight/(tf.reduce_sum(weight,0,keepdims=True)+ epsilon)
        interpolated      = tf.reduce_sum(tf.multiply(warped_image,weight_normalized),0,keepdims=True)
        
  
    
    return interpolated

