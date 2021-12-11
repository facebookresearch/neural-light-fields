


import numpy as np
import cv2,os


def load_imgs(args):


    print("Dataset path: %s"%(args.dataset))
    path_images = sorted(os.listdir(args.dataset))


    img =  cv2.imread(os.path.join(args.dataset, path_images[0]))
    h_res,w_res,_ = img.shape
    h_res = h_res//args.factor
    w_res = w_res//args.factor
    num_n = args.num_n

    print('Downsampling factor: %d'%(args.factor))
    print('Output resolution: %d x %d'%(h_res,w_res))
    dims = args.dim

    if len(path_images) != np.prod(dims):
        rows = 17
        cols = 17
        step = 4

        new_path_images = []

        for t_idx in range(0, rows, 1):
            for s_idx in range(0, cols, 1):
                if ((t_idx % step) == 0) and ((s_idx % step) == 0):
                    idx = t_idx * cols + s_idx
                    new_path_images.append(path_images[idx])
                else:
                    continue

        path_images = new_path_images

    coordinates = []
    all_pairs   = []
    images      = []
    if args.type == ['light','view','time']:

        for id in range(len(path_images)):

           print('\r Reading image %d out of %d '%(id+1,len(path_images)),end=" " )

           img =  cv2.imread(os.path.join(args.dataset, path_images[id]))
           img = cv2.resize(img,None,fx=1/args.factor,fy=1/args.factor, interpolation = cv2.INTER_AREA)

           img = np.float32(img)/255.0

           time  = id //(dims[0]*dims[1])
           rest  = id % (dims[0]*dims[1])
           view  = rest % dims[1]
           light = rest // dims[1]

           coordinates.append(np.array([[[light,view,time]]]))
           images.append(img)


           pair = np.array([light,light-1,light+1,view,view,view,time,time,time])
           pair = np.where(pair < 0, 2, pair)
           pair = np.where(pair > dims[0]-1 , dims[0]-3, pair)
           all_pairs.append(pair)


           pair = np.array([light,light,light,view,view-1,view+1,time,time,time])
           pair = np.where(pair < 0, 2, pair)
           pair = np.where(pair > dims[1]-1 , dims[1]-3, pair)
           all_pairs.append(pair)


           pair = np.array([light,light,light,view,view,view,time,time-1,time+1])
           pair = np.where(pair < 0, 2, pair)
           pair = np.where(pair > dims[2]-1 , dims[2]-3, pair)
           all_pairs.append(pair)


        stack = np.stack(all_pairs,0)
        img_index    = stack[:,0:3]*dims[1] + stack[:,3:6] + stack[:,6:9]*dims[0]*dims[1]
        albedo_index = stack[:,3:4]*dims[1] + stack[:,6:7]
        training_pairs = np.concatenate((img_index,albedo_index),-1)
        images = np.stack(images,0)
        coordinates = np.stack(coordinates,0)



    if args.type == ['view'] or args.type == ['light']:

        if num_n > np.prod(dims):
            num_n = np.prod(dims)-1

        for id in range(len(path_images)):

           print('\r Reading image %d out of %d '%(id+1,len(path_images)),end=" " )

           img =  cv2.imread(os.path.join(args.dataset, path_images[id]))
           img = cv2.resize(img,None,fx=1/args.factor,fy=1/args.factor, interpolation = cv2.INTER_AREA)

           img = np.float32(img)/255.0



           cx  = id % dims[1]
           cy  = id // dims[1]

           coordinates.append(np.array([[[cx,args.br*cy]]]))
           images.append(img)


        images = np.stack(images,0)
        coordinates = np.stack(coordinates,0)

        for id in range(len(path_images)):

            dist = np.sum(np.square(coordinates[id,0,0,:] - coordinates[:,0,0,:]),-1)
            idx = np.argsort(dist)[:num_n+1]
            all_pairs.append(idx)

        training_pairs = np.stack(all_pairs,0)


    if args.type == ['time']:


        if num_n > np.prod(dims):
            num_n = np.prod(dims)-1

        for id in range(len(path_images)):

           print('\r Reading image %d out of %d '%(id+1,len(path_images)),end=" " )

           img =  cv2.imread(os.path.join(args.dataset, path_images[id]))
           img = cv2.resize(img,None,fx=1/args.factor,fy=1/args.factor, interpolation = cv2.INTER_AREA)

           img = np.float32(img)/255.0


           coordinates.append(np.array([[[id]]]))
           images.append(img)

        images = np.stack(images,0)
        coordinates = np.stack(coordinates,0)

        for id in range(len(path_images)):

            dist = np.abs(coordinates[id,0,0,0] - coordinates[:,0,0,0])
            idx = np.argsort(dist)[:num_n+1]
            all_pairs.append(idx)

        training_pairs = np.stack(all_pairs,0)

    return images,coordinates,training_pairs,h_res,w_res
