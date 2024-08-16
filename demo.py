import sys
sys.path.append('droid_slam')

from tqdm import tqdm
import numpy as np
import torch
import lietorch
import cv2
import os
import glob 
import time
import argparse
import geom.projective_ops as pops

from torch.multiprocessing import Process
from droid import Droid
from pathlib import Path
from lietorch import SE3
from PIL import Image
import torch.nn.functional as F
torch.manual_seed(0)

def show_image(image, imageR = None):
    if imageR is not None:
        disp_image = torch.concat((image, imageR), 1)
    else:
        disp_image = image
    disp_image = disp_image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', disp_image / 255.0)
    cv2.waitKey(1)

def image_stream(imagedir, calib, stride, depthdir = None, rimagedir=None, stereorelpose=None):
    """ image generator """

    calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]

    stereo_rel_pose = None
    if stereorelpose is not None:
        stereo_rel_pose = torch.as_tensor(np.loadtxt(stereorelpose, delimiter=" "))

    K = np.eye(3)
    K[0,0] = fx
    K[0,2] = cx
    K[1,1] = fy
    K[1,2] = cy

    image_list = sorted(os.listdir(imagedir))[::stride]
    rimage_list = sorted(os.listdir(rimagedir))[::stride] if rimagedir else None
    depth_list = sorted(os.listdir(depthdir))[::stride] if depthdir else None
    for t, imfile in enumerate(image_list):
        image = cv2.imread(os.path.join(imagedir, imfile))
        if len(calib) > 4:
            image = cv2.undistort(image, K, calib[4:])

        h0, w0, _ = image.shape
        h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
        w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

        image = cv2.resize(image, (w1, h1))
        image = image[:h1-h1%8, :w1-w1%8]
        images = [image]
        # image = torch.as_tensor(image).permute(2, 0, 1)

        if rimage_list:
            imager = cv2.imread(os.path.join(rimagedir, imfile))
            if len(calib) > 4:
                imager = cv2.undistort(imager, K, calib[4:])

            imager = cv2.resize(imager, (w1, h1))
            imager = imager[:h1-h1%8, :w1-w1%8]
            images += [imager]
            # imager = torch.as_tensor(imager).permute(2, 0, 1)
        else:
            imager = None
        # image = image + torch.randn_like(image, dtype=torch.float32) * 10.0
        images = torch.from_numpy(np.stack(images, 0))
        images = images.permute(0, 3, 1, 2).to("cuda:0", dtype=torch.float32)

        intrinsics = torch.as_tensor([fx, fy, cx, cy])
        intrinsics[0::2] *= (w1 / w0)
        intrinsics[1::2] *= (h1 / h0)
        if depth_list:
            if depth_list[t].endswith('.npy'):
                depth = np.load(os.path.join(depthdir, depth_list[t]))
            else:
                depth = np.array(Image.open(os.path.join(depthdir, depth_list[t])))

            depth = depth.squeeze()
            depth = depth.astype(np.float32) / 10.0
            depth = torch.as_tensor(depth)
            depth = F.interpolate(depth[None,None], (h1, w1)).squeeze()
            depth = depth[:h1-h1%8, :w1-w1%8]
            depth_img = depth / depth.max()
            cv2.imshow('depth', depth_img.numpy())
            cv2.waitKey(1)
        else:
            depth = None

        yield t, images, intrinsics, depth, stereo_rel_pose

def save_reconstruction(droid, reconstruction_path):

    import random
    import string

    t = droid.video.counter.value
    tstamps = droid.video.tstamp[:t].cpu().numpy()
    images = droid.video.images[:t].cpu().numpy()
    disps = droid.video.disps_up[:t].cpu().numpy()
    poses_torch = droid.video.poses[:t]
    print(f'Projecting all poses.')
    poses_mat = SE3(poses_torch).matrix()
    # Undo depth normalization
    # poses_torch[:, :3] /= 10.0
    rcm = pops.find_rcm(poses_torch).squeeze()
    poses_torch_rcm = pops.project_out_rcm(poses_torch, rcm)
    poses_world = SE3(poses_torch_rcm).inv()
    poses_world.data[:, :3] -= rcm.unsqueeze(0)
    poses_new = poses_world.inv()
    #new_translations = -poses.matrix()[0, :, :3, :3] @ rcm
    # poses_torch_rcm = pops.optimize_out_rcm(poses_torch, rcm)
    poses = droid.video.poses[:t].cpu().numpy()
    print(f'num poses: {t}')
    poses_mat = poses_mat.cpu().numpy()
    intrinsics = droid.video.intrinsics[:t].cpu().numpy()
    Path("reconstructions/{}".format(reconstruction_path)).mkdir(parents=True, exist_ok=True)
    np.save("reconstructions/{}/tstamps.npy".format(reconstruction_path), tstamps)
    np.save("reconstructions/{}/images.npy".format(reconstruction_path), images)
    np.save("reconstructions/{}/disps.npy".format(reconstruction_path), disps)
    np.save("reconstructions/{}/poses.npy".format(reconstruction_path), poses)
    np.save("reconstructions/{}/poses_mat.npy".format(reconstruction_path), poses_mat)
    np.save("reconstructions/{}/intrinsics.npy".format(reconstruction_path), intrinsics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagedir", type=str, help="path to image directory")
    parser.add_argument("--rimagedir", type=str, help="path to image directory for stereo case.", default=None)
    parser.add_argument("--depthdir", type=str, help="path to depth directory")
    parser.add_argument("--calib", type=str, help="path to calibration file")
    parser.add_argument("--stereo_pose", type=str, help="relative pose of stereo camleft w.r.t. camright.", default="calib/rel_pose_miti.txt")
    parser.add_argument("--t0", default=0, type=int, help="starting frame")
    parser.add_argument("--stride", default=3, type=int, help="frame stride")

    parser.add_argument("--weights", default="droid.pth")
    parser.add_argument("--buffer", type=int, default=512)
    parser.add_argument("--image_size", default=[240, 320])
    parser.add_argument("--disable_vis", action="store_true")

    parser.add_argument("--beta", type=float, default=0.3, help="weight for translation / rotation components of flow")
    parser.add_argument("--filter_thresh", type=float, default=1.0, help="how much motion before considering new keyframe")
    parser.add_argument("--warmup", type=int, default=8, help="number of warmup frames")
    parser.add_argument("--keyframe_thresh", type=float, default=1.0, help="threshold to create a new keyframe")
    parser.add_argument("--frontend_thresh", type=float, default=16000.0, help="add edges between frames whithin this distance")
    parser.add_argument("--frontend_window", type=int, default=70, help="frontend optimization window")
    parser.add_argument("--frontend_radius", type=int, default=2, help="force edges between frames within radius")
    parser.add_argument("--frontend_nms", type=int, default=1, help="non-maximal supression of edges")

    parser.add_argument("--backend_thresh", type=float, default=22.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)
    parser.add_argument("--upsample", action="store_true")
    parser.add_argument("--reconstruction_path", help="path to saved reconstruction")
    parser.add_argument("--max_images", type=int, default=-1)
    parser.add_argument("--rcm_reg", type=float, default=0.0)

    args = parser.parse_args()

    args.stereo = True if args.rimagedir is not None else False
    torch.multiprocessing.set_start_method('spawn')
    droid = None

    # need high resolution depths
    if args.reconstruction_path is not None:
        args.upsample = True

    for (t, images, intrinsics, depth, stereo_rel_pose) in tqdm(image_stream(args.imagedir, args.calib, args.stride, args.depthdir, args.rimagedir, args.stereo_pose)):
        if t < args.t0:
            continue

        if not args.disable_vis:
            if args.stereo:
                show_image(images[0], images[1])
            else:
                show_image(images[0])

        if droid is None:
            args.image_size = [images.shape[2], images.shape[3]]
            droid = Droid(args)
        
        droid.track(t, images, intrinsics=intrinsics, depth=depth, stereo_rel_pose=stereo_rel_pose)
        if args.max_images > 0 and args.max_images < t:
            print('break')
            break
    
    if args.reconstruction_path is not None:
        save_reconstruction(droid, args.reconstruction_path)

    traj_est = droid.terminate(image_stream(args.imagedir, args.calib, args.stride))
    if args.reconstruction_path is not None:
        from evo.tools import file_interface
        np.save("reconstructions/{}/traj_est.npy".format(args.reconstruction_path), traj_est)

        traj_est = SE3(torch.from_numpy(traj_est)).matrix().cpu().numpy()
        traj_est = [a for a in traj_est]
        tstamps = np.arange(len(traj_est)).astype(np.float64)

        from evo.core.trajectory import PoseTrajectory3D
        traj_est = PoseTrajectory3D(poses_se3=traj_est, timestamps=tstamps)
        
        file_interface.write_tum_trajectory_file("reconstructions/{}/traj_est.txt".format(args.reconstruction_path), traj_est, confirm_overwrite=False)
