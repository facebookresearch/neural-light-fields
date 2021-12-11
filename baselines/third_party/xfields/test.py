
import tensorflow as tf
import numpy as np
import cv2,os
from blending import Blending_test
from load_imgs import load_imgs
from model import Flow
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')


parser = argparse.ArgumentParser()
parser.add_argument("--render_spiral", action='store_true',
                    help='Render spiral path')
parser.add_argument("--render_row", action='store_true',
                    help='Render row of light field')
parser.add_argument('--dataset',  type=str,
                    help='path to dataset',        default = 'dataset/view_light_time/pomegranate')
parser.add_argument('--type',     type=str, nargs= "*",
                    help='xfield type',            default = ['light','view','time'])
parser.add_argument('--dim',      type=int, nargs= "*",
                    help='dimension of input Xfields',   default = [5,5])
parser.add_argument('--factor',   type=int,
                    help='downsampling factor',      default = 1)
parser.add_argument('--nfg',      type=int,
                    help='capacity multiplier',    default = 16)
parser.add_argument('--disp_row',      type=int,
                    help='capacity multiplier',    default = 8)
parser.add_argument('--num_n',    type=int,
                    help='number of neighbors',    default = 8)
parser.add_argument('--epoch_end',       type=int,
                    help='learning rate',          default = 500)
parser.add_argument('--sigma',    type=float,
                    help='bandwidth parameter',    default = 0.5)
parser.add_argument('--spiral_scale',    type=float,
                    help='bandwidth parameter',    default = 0.5)
parser.add_argument('--br',      type=float,
                    help='baseline ratio',         default = 1)
parser.add_argument('--savepath', type=str,
                    help='saving path',      default = 'results/')
parser.add_argument('--head_tail', type=str,
                    help='Experiment name tail',             default = 'bulldozer')
parser.add_argument('--scale',      type=int,
                    help='number of intermediate points',     default = 17)
parser.add_argument('--fps',      type=float,
                    help='output video frame rate',  default = 20)

args = parser.parse_args()


def run_test(args):


    print('---------- Perform Testing ----------')

    savedir = args.savepath
    if not os.path.exists(savedir):
        os.mkdir( savedir )
    savedir = os.path.join(savedir,args.head_tail)

    if not os.path.exists(savedir):
        raise NameError('There is no directory:\n %s'%(savedir))

    if not os.path.exists(os.path.join(savedir,"rendered_videos") ):
        os.mkdir( os.path.join(savedir,"rendered_videos") )
        print('creating directory %s'%(os.path.join(savedir,"rendered_videos")))


    print('XField type: %s'%(args.type))
    print( 'Dimension of input xfield: %s'%(args.dim))
    print('output video fps: %d'%(args.fps))
    print('number of intermediate points for interpolation: %d'%(args.scale))

    images,coordinates,all_pairs,h_res,w_res = load_imgs(args)
    min_ = np.min(coordinates)
    max_ = np.max(coordinates)

    dims = args.dim
    num_n = args.num_n

    if num_n > np.prod(dims):
            num_n = np.prod(dims)

    input = tf.placeholder(tf.float32,shape=[1,1,1,len(dims)])


    num_output = len(args.type)*2
    with tf.variable_scope("gen_flows"):
        flows = Flow(input,h_res,w_res,num_output,args.nfg,min_,max_)

    if args.type == ['light','view','time']:

        with tf.variable_scope("gen_flows"):
                albedos = tf.Variable(tf.constant(1.0, shape=[dims[1]*dims[2],h_res, w_res,3]), name='albedo')
                index_albedo   = tf.placeholder(tf.int32,shape=(num_n,))
                albedo         = tf.gather(albedos,index_albedo,0)

    elif args.type == ['light']:

        with tf.variable_scope("gen_flows"):
            albedo = tf.Variable(tf.constant(1.0, shape=[1,h_res, w_res,3]), name='albedo')

    else:
        albedo =   tf.constant(1.0, shape=[1,h_res, w_res,3])





    input_N        = tf.placeholder(tf.float32,shape=[num_n,1,1,len(dims)])
    Neighbors_img  = tf.placeholder(tf.float32,shape=[num_n,h_res,w_res,3])
    Neighbors_flow = tf.placeholder(tf.float32,shape=[num_n,h_res,w_res,len(args.type)*2])


    interpolated = Blending_test(input,input_N,Neighbors_img,Neighbors_flow,flows,albedo,h_res,w_res,args)

    saver = tf.train.Saver(max_to_keep=1000)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    ckpt=tf.train.get_checkpoint_state("%s/trained_model/"%(savedir))
    if ckpt:
        print('\n loading pretrained_model  '+ckpt.model_checkpoint_path)
        saver.restore(sess,ckpt.model_checkpoint_path)
    else:
        raise NameError('There is no pretrained_model located at dir:\n %s/trained_model/'%(savedir))

    precomputed_flows = []

    for i in range(len(coordinates)):
        flows_out = sess.run(flows,feed_dict={input:coordinates[[i],::]})
        precomputed_flows.append(flows_out[0,::])

    precomputed_flows = np.stack(precomputed_flows,0)



    if args.type == ['view'] or args.type == ['light'] or args.type == ['time']:
        if args.render_spiral:
            N = 120
            rows = 17
            cols = 17
            rots = 2

            spiral_scale = args.spiral_scale
            disp_row = args.disp_row

            all_xs = []
            all_ys = []

            for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
                s_idx = (np.cos(theta) * spiral_scale + 1) / 2.0 * (cols - 1)
                t_idx = np.sin(theta) * spiral_scale / 2.0 * (rows - 1) + ((rows - 1) - disp_row)

                s = (s_idx / (cols - 1))
                t = (t_idx / (rows - 1))

                all_xs.append(s)
                all_ys.append(t)

            X = np.array(all_xs)
            Y = np.array(all_ys)
        elif args.render_row:
            disp_row = args.disp_row
            supersample = 4
            rows = 17
            cols = 17

            X = np.linspace(0, 1, cols * supersample)[::-1]
            Y = np.linspace(0, 1, rows)[::-1][disp_row][None]
            X, Y = np.meshgrid(X, Y, indexing='xy')
        else:
            rows = 17
            cols = 17

            X = np.linspace(0, 1, cols)
            Y = np.linspace(0, 1, rows)
            X, Y = np.meshgrid(X, Y, indexing='xy')

        X = X.reshape(-1)
        Y = Y.reshape(-1)


        if args.type == ['view'] or args.type == ['light'] :

            X = X*(dims[1]-1)
            Y = Y*(dims[0]-1)
            rendering_path = np.transpose([X,Y])


        if args.type == ['time']:

            rendering_path = np.transpose([X*(dims[0]-1)])



        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('%s/rendered_videos/rendered.mp4'%(savedir),fourcc, args.fps, (w_res,h_res))
        os.makedirs(f'{savedir}/rendered_videos/rendered_images', exist_ok=True)

        for id in range(len(X)):

                input_coord = np.array([[[rendering_path[id,:]]]])
                indices = np.argsort(np.sum(np.square(input_coord[0,0,0,:]-coordinates[:,0,0,:]),-1))[:num_n]


                input_coord_N   = coordinates[indices,::]
                input_Neighbors = images[indices,::]
                input_flows     = precomputed_flows[indices,::]


                im_out = sess.run(interpolated,feed_dict={        input         :input_coord,
                                                                  input_N       :input_coord_N,
                                                                  Neighbors_img :input_Neighbors,
                                                                  Neighbors_flow:input_flows,
                                                                          })
                im_out = np.minimum(np.maximum(im_out[0,::],0.0),1.0)
                out.write(np.uint8(im_out*255))
                cv2.imwrite(f'{savedir}/rendered_videos/rendered_images/{id:04d}.png', np.uint8(im_out*255))

                print('\r interpolated image %d of %d'%(id+1,len(rendering_path)),end=" " )


        out.release()


    if args.type == ['light','view','time']:

        print('\n number of neighbors for interpolation: %d'%(num_n))
        max_L = dims[0]-1
        max_V = dims[1]-1
        max_T = dims[2]-1

        X_L = np.linspace(0,max_L,max_L*args.scale)
        X_L = np.append(X_L,np.flip(X_L))
        X_V = np.linspace(0,max_V,max_V*args.scale)
        X_V = np.append(X_V,np.flip(X_V))
        X_T = np.linspace(0,max_T,max_T*args.scale)
        X_T = np.append(X_T,np.flip(X_T))
        middle_X_L = max_L*0.5*np.ones_like(X_L)
        middle_X_V = max_V*0.5*np.ones_like(X_V)
        middle_X_T = max_T*0.5*np.ones_like(X_T)

        all_dimensions = {'light'      :np.stack([   X_L,    middle_X_V,middle_X_T],1),
                          'view'       :np.stack([middle_X_L,    X_V,   middle_X_T],1),
                          'time'       :np.stack([middle_X_L,middle_X_V       ,X_T],1),
                          'light_view' :np.stack([X_L,            X_V,   middle_X_T],1),
                          'light_time' :np.stack([X_L,        middle_X_V,       X_T],1),
                          'view_time'  :np.stack([middle_X_L,    X_V,           X_T],1),
                          'light_view_time': np.stack([X_L,       X_V,           X_T],1) }


        for case, rendering_path in all_dimensions.items():

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter('%s/rendered_videos/rendered_%s.mp4'%(savedir,case),fourcc, args.fps, (w_res,h_res))

            print('\n --------- %s interpolation ---------'%(case))

            for id in range(len(rendering_path)):

                input_coord = np.array([[[rendering_path[id,:]]]])
                indices = np.argsort(np.sum(np.square(input_coord[0,0,0,:]-coordinates[:,0,0,:]),-1))[:num_n]

                input_coord_N   = coordinates[indices,::]
                input_Neighbors = images[indices,::]
                input_flows     = precomputed_flows[indices,::]

                time_idx  = indices //(dims[0]*dims[1])
                rest  = indices % (dims[0]*dims[1])
                view_idx  = rest % dims[1]
                albedo_index    = view_idx*dims[1] + time_idx

                im_out = sess.run(interpolated,feed_dict={ input         :input_coord,
                                                           input_N       :input_coord_N,
                                                           Neighbors_img :input_Neighbors,
                                                           Neighbors_flow:input_flows,
                                                           index_albedo  :albedo_index,
                                                                          })

                im_out = np.minimum(np.maximum(im_out[0,::],0.0),1.0)
                out.write(np.uint8(im_out*255))

                print('\r interpolated image %d of %d'%(id+1,len(rendering_path)),end=" " )

            out.release()



if __name__=='__main__':

    run_test(args)
    print('\n The interpolation result is located in the \'rendered_videos\' folder.')
