
import numpy as np
import cv2,os,time
import flow_vis
from blending import Blending_train
from load_imgs import load_imgs
from model import Flow
import argparse
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset',  type=str,
                    help='path to dataset',        default = 'dataset/view_light_time/pomegranate')
parser.add_argument('--type',     type=str, nargs= "*",
                    help='xfield type',            default = ['light','view','time'])
parser.add_argument('--dim',      type=int, nargs= "*",
                    help='dimension of input xfields',   default = [5,5])
parser.add_argument('--factor',   type=int,
                    help='downsampling factor',      default = 1)
parser.add_argument('--nfg',      type=int,
                    help='capacity multiplier',    default = 16)
parser.add_argument('--num_n',    type=int,
                    help='number of neighbors',    default = 4)
parser.add_argument('--iters',       type=float,
                    help='learning rate',          default = 250000)
parser.add_argument('--lr',       type=float,
                    help='learning rate',          default = 0.0001)
parser.add_argument('--sigma',    type=float,
                    help='bandwidth parameter',    default = 0.5)
parser.add_argument('--br',      type=float,
                    help='baseline ratio',         default = 1)
parser.add_argument('--load_pretrained', type=bool,
                    help='loading pretrained_model',default = False)
parser.add_argument('--savepath', type=str,
                    help='saving path',             default = 'results/')

args = parser.parse_args()


def run_training(args):

    print('---------- Perform Training ----------')

    savedir = args.savepath
    if not os.path.exists(savedir):
        os.mkdir( savedir )

    head_tail = os.path.split(args.dataset)
    savedir = os.path.join(savedir,head_tail[1])

    if not os.path.exists(savedir):
        os.mkdir( savedir )

    if not os.path.exists(os.path.join(savedir,"trained_model") ):
        os.mkdir( os.path.join(savedir,"trained_model") )
        print('creating directory %s'%(os.path.join(savedir,"trained_model")))

    if not os.path.exists(os.path.join(savedir,"saved_training") ):
        os.mkdir( os.path.join(savedir,"saved_training") )
        print('creating directory %s'%(os.path.join(savedir,"saved_training")))



    print('XField type: %s'%(args.type))
    print( 'Dimension of input xfield: %s'%(args.dim))


    #loading images
    images,coordinates,all_pairs,h_res,w_res = load_imgs(args)

    dims = args.dim
    num_n = args.num_n # number of neighbors
    min_ = np.min(coordinates)
    max_ = np.max(coordinates)

    print('\n ------- Creating the model -------')


    # batch size is num_n + 1 (number of neighbors + target)
    inputs = tf.placeholder(tf.float32,shape=[num_n+1,1,1,len(dims)])


    # Jacobian network
    num_output = len(args.type)*2

    with tf.variable_scope("gen_flows"):
        flows = Flow(inputs,h_res,w_res,num_output,args.nfg,min_,max_)

    nparams_decoder = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables() if v.name.startswith("gen_flows")])
    print('Number of learnable parameters (decoder): %d' %(nparams_decoder))


    # learnt albedo
    # The albedos are initialized with constant 1.0
    if args.type == ['light','view','time']:


        with tf.variable_scope("gen_flows"):

            # For light-view-time interpolation, we consider num_views*num_times albedos
            albedos = tf.Variable(tf.constant(1.0, shape=[dims[1]*dims[2],h_res, w_res,3]), name='albedo')
            index_albedo = tf.placeholder(tf.int32,shape=(1,))
            albedo =   tf.gather(albedos,index_albedo,0)

        nparams = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables() if v.name.startswith("gen_flows")])
        print('Number of learnable parameters (%d albedos with res %d x %d ): %d' %(dims[1]*dims[2],h_res,w_res,nparams-nparams_decoder))

    elif args.type == ['light']:

        with tf.variable_scope("gen_flows"):
            # For light interpolation, we consider just one albedo
            albedo = tf.Variable(tf.constant(1.0, shape=[1,h_res, w_res,3]), name='albedo')

        nparams = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables() if v.name.startswith("gen_flows")])
        print('Number of learnable parameters (%d albedos with res %d x %d ): %d' %(1,h_res,w_res,nparams-nparams_decoder))

    else:
        # For view and time interpolation, we do not train for albedo, we consider it as a constant non-learnable parameter
        albedo =   tf.constant(1.0, shape=[1,h_res, w_res,3])


    Neighbors = tf.placeholder(tf.float32,shape=[num_n,h_res,w_res,3])

    # soft blending
    interpolated = Blending_train(inputs,Neighbors,flows,albedo,h_res,w_res,args)

    Reference = tf.placeholder(tf.float32,shape=[1,h_res,w_res,3])

    # L1 loss
    loss  = tf.reduce_mean((tf.abs(interpolated-Reference)))

    gen_tvars = [var for var in tf.trainable_variables() if var.name.startswith("gen_flows")]
    learning_rate = tf.placeholder(tf.float32,shape=())
    gen_optim = tf.train.AdamOptimizer(learning_rate)
    gen_grads = gen_optim.compute_gradients(loss, var_list=gen_tvars)
    gen_train = gen_optim.apply_gradients(gen_grads)

    saver = tf.train.Saver(max_to_keep = 1000)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    if args.load_pretrained:
        ckpt=tf.train.get_checkpoint_state("%s/trained_model"%(savedir))
        if ckpt:
                print('\n loading pretrained_model  '+ckpt.model_checkpoint_path)
                saver.restore(sess,ckpt.model_checkpoint_path)

    print('------------ Start Training ------------')

    lr = args.lr
    print('Starting learning rate with %0.4f'%(lr))

    stop_l1_thr = 0.0001
    iter_end = args.iters                   # total number of iterations

    indices = np.array([i for i in range(len(all_pairs))])
    if len(indices)<500: # we considered around 500 iterations per each epoch
      indices = np.repeat(indices,500//len(indices))

    epoch_size = len(indices)
    epoch_end = int(iter_end//epoch_size)    # total number of epochs

    if args.type == ['light','view','time']:

        st = time.time()
        min_loss = 1000
        l1_loss_t = 1
        epoch = 0

        while l1_loss_t > stop_l1_thr and epoch <= epoch_end:

               l1_loss_t = 0
               np.random.shuffle(indices)


               for id in range(epoch_size):

                   pair =  all_pairs[indices[id],::]

                   input_coords = coordinates[pair[:num_n+1],::]
                   reference_img = images[pair[:1],::]
                   Neighbors_img = images[pair[1:num_n+1],::]
                   _index = [pair[-1]]

                   _,l1loss = sess.run([gen_train,loss],feed_dict={inputs:input_coords,
                                                                   Reference:reference_img,
                                                                   Neighbors: Neighbors_img,
                                                                   learning_rate:lr,
                                                                   index_albedo:_index
                                                                   })
                   l1_loss_t = l1_loss_t + l1loss

                   print('\r Epoch %3.0d  Iteration %3.0d of %3.0d   Cumulative L1 loss = %3.3f'%(epoch,id+1,epoch_size,l1_loss_t),end=" " )



               l1_loss_t = l1_loss_t/epoch_size
               print(" elapsed time %3.1f m  Averaged L1 loss = %3.5f "%((time.time()-st)/60,l1_loss_t))

               if l1_loss_t < min_loss:
                      saver.save(sess,"%s/trained_model/model.ckpt"%(savedir))
                      min_loss = l1_loss_t

               center = np.prod(dims)//2
               cv2.imwrite("%s/saved_training/reference.png"%(savedir),np.uint8(images[center,::]*255))


               pair =  all_pairs[3*center + 0,::]

               out_img,flows_out = sess.run([interpolated,flows],feed_dict={inputs      :coordinates[pair[:num_n+1],::],
                                                                            Neighbors   :images[pair[1:num_n+1],::],
                                                                            index_albedo:[pair[-1]]})

               out_img = np.minimum(np.maximum(out_img,0.0),1.0)
               cv2.imwrite("%s/saved_training/recons_light.png"%(savedir),np.uint8(out_img[0,::]*255))

               flow_color = flow_vis.flow_to_color(flows_out[0,:,:,0:2], convert_to_bgr=False)
               cv2.imwrite("%s/saved_training/flow_light.png"%(savedir),np.uint8(flow_color))

               flow_color = flow_vis.flow_to_color(flows_out[0,:,:,2:4], convert_to_bgr=False)
               cv2.imwrite("%s/saved_training/flow_view.png"%(savedir),np.uint8(flow_color))

               flow_color = flow_vis.flow_to_color(flows_out[0,:,:,4:6], convert_to_bgr=False)
               cv2.imwrite("%s/saved_training/flow_time.png"%(savedir),np.uint8(flow_color))


               pair =  all_pairs[3*center + 1,::]
               out_img = sess.run(interpolated,feed_dict={inputs      :coordinates[pair[:num_n+1],::],
                                                          Neighbors   :images[pair[1:num_n+1],::],
                                                          index_albedo:[pair[-1]]})

               out_img = np.minimum(np.maximum(out_img,0.0),1.0)
               cv2.imwrite("%s/saved_training/recons_view.png"%(savedir),np.uint8(out_img[0,::]*255))


               pair =  all_pairs[3*center + 2,::]
               out_img = sess.run(interpolated,feed_dict={inputs      :coordinates[pair[:num_n+1],::],
                                                          Neighbors   :images[pair[1:num_n+1],::],
                                                          index_albedo:[pair[-1]]})

               out_img = np.minimum(np.maximum(out_img,0.0),1.0)
               cv2.imwrite("%s/saved_training/recons_time.png"%(savedir),np.uint8(out_img[0,::]*255))
               epoch  = epoch + 1


               if epoch == epoch_end//2:
                   lr =  0.00005





    if args.type == ['view'] or args.type ==['time'] or args.type ==['light'] :


        st = time.time()
        img_mov = cv2.VideoWriter('%s/saved_training/epoch_recons.mp4'%(savedir),cv2.VideoWriter_fourcc(*'mp4v'), 10, (w_res,h_res))
        flow_mov = cv2.VideoWriter('%s/saved_training/epoch_flows.mp4'%(savedir),cv2.VideoWriter_fourcc(*'mp4v'), 10, (w_res,h_res))

        min_loss = 1000
        l1_loss_t = 1
        epoch = 0

        while l1_loss_t > stop_l1_thr and epoch <= epoch_end:

               l1_loss_t = 0
               np.random.shuffle(indices)



               for id in range(epoch_size):

                   pair          =  all_pairs[indices[id],::]
                   input_coords  = coordinates[pair[:num_n+1],::]
                   reference_img = images[pair[:1],::]
                   Neighbors_img = images[pair[1:num_n+1],::]

                   _,l1loss = sess.run([gen_train,loss],feed_dict={inputs:input_coords,
                                                                   Reference:reference_img,
                                                                   Neighbors: Neighbors_img,
                                                                   learning_rate:lr,
                                                                   })

                   l1_loss_t = l1_loss_t + l1loss
                   print('\r Epoch %3.0d  Iteration %3.0d of %3.0d   Cumulative L1 loss = %3.3f'%(epoch,id+1,epoch_size,l1_loss_t),end=" " )


               l1_loss_t = l1_loss_t/epoch_size
               print(" elapsed time %3.1f m  Averaged L1 loss = %3.5f"%((time.time()-st)/60,l1_loss_t))

               if l1_loss_t < min_loss:
                      saver.save(sess,"%s/trained_model/model.ckpt"%(savedir))
                      min_loss = l1_loss_t


               if args.type == ['light']:

                   albedo_out = np.minimum(np.maximum(sess.run(albedo),0.0),1.0)
                   cv2.imwrite("%s/saved_training/albedo.png"%(savedir),np.uint8(albedo_out[0,:,:,:]*255))




               center = np.prod(dims)//2
               cv2.imwrite("%s/saved_training/reference.png"%(savedir),np.uint8(images[center,::]*255))

               pair =  all_pairs[(len(all_pairs)//len(images)) *center,::]

               out_img,flows_out = sess.run([interpolated,flows],feed_dict={inputs      :coordinates[pair[:num_n+1],::],
                                                                            Neighbors   :images[pair[1:num_n+1],::]})

               out_img = np.minimum(np.maximum(out_img,0.0),1.0)
               cv2.imwrite("%s/saved_training/recons.png"%(savedir),np.uint8(out_img[0,::]*255))

               flow_color = flow_vis.flow_to_color(flows_out[0,:,:,0:2], convert_to_bgr=False)
               cv2.imwrite("%s/saved_training/flow.png"%(savedir),np.uint8(flow_color))
               img_mov.write(np.uint8(out_img[0,::]*255))
               flow_mov.write(np.uint8(flow_color))
               epoch  = epoch + 1


               if  epoch == epoch_end//2:
                   lr =  0.00005




        img_mov.release()
        flow_mov.release()





if __name__=='__main__':


    run_training(args)
