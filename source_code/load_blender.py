import os
from re import L
import torch
import numpy as np
import imageio 
import json
import torch.nn.functional as F
import cv2


trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


def load_blender_data(basedir, half_res=False, testskip=1):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]
    all_sub_imgs = []
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        sub_imgs = []
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip
            
        for frame in meta['frames'][::skip]:
            if frame['file_path'][-4:] == '.png':
                fname = os.path.join(basedir, frame['file_path'])
                sub_fname = os.path.join(basedir, frame['file_path'][:-4] + '_mask.png')    # for contour optimization
            else:    
                fname = os.path.join(basedir, frame['file_path'] + '.png')
                sub_fname = os.path.join(basedir, frame['file_path'] + '_mask.png')         # for contour optimization
            if os.path.exists(fname):
                imgs.append(imageio.imread(fname))
            else:
                # magic code
                imgs.append(imageio.imread('data/nerf_synthetic/multihuman/render/000.png'))
            
            # for contour optimization
            if os.path.exists(sub_fname):
                sub_imgs.append(imageio.imread(sub_fname))
            else:
                sub_imgs.append(imageio.imread('data/nerf_synthetic/multihuman/render/000_mask.png'))
            poses.append(np.array(frame['transform_matrix']))
        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA), stack all the img_info together
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
        #  -----------------------------------------------------for contour optimization------------------------------------------------
        if s == 'train':
            sub_imgs = (np.array(sub_imgs) / 255.).astype(np.float32)   #(imgs.shape[0], H, W, 3)
            all_sub_imgs = sub_imgs

    No, h, w, three = np.nonzero(all_sub_imgs)
    No_, h_, w_, three_ = np.where(all_sub_imgs==0)
    # print(np.where(all_sub_imgs==0))
    non_zero = []
    zeros = []
    inside_coords = []   # (counts[1], ..., 2)
    outside_coords = []
    for i in range (0, counts[1]):
        inside_coords.append([])
        outside_coords.append([])
    for i in range (0, len(No), 3):
        non_zero.append([No[i], h[i], w[i]])
    for i in range (0, len(No_), 3):
        zeros.append([No_[i], h_[i], w_[i]])
    for i in range (0, len(non_zero)):
        inside_coords[non_zero[i][0]].append([non_zero[i][1], non_zero[i][2]])
    for i in range (0, len(zeros)):
        outside_coords[zeros[i][0]].append([zeros[i][1], zeros[i][2]])
    # inside_coords = torch.Tensor(inside_coords)
    # print("outside:", len(outside_coords[16]))
    #  -----------------------------------------------------for contour optimization------------------------------------------------
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    
    H, W = imgs[0].shape[:2]
    if not meta.__contains__("camera_angle_x"):
        focal = meta["focal"]
    else:
        camera_angle_x = float(meta['camera_angle_x'])
        focal = .5 * W / np.tan(.5 * camera_angle_x)
    
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
    
    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (H, W), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()

        
    return imgs, poses, render_poses, [H, W, focal], i_split, inside_coords, outside_coords


