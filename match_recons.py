import numpy as np
import argparse, os
from lietorch import SE3
import torch
import open3d as o3d
import matplotlib.pyplot as plt 
import torch.nn.functional as F
import droid_slam.geom.projective_ops as pops

def load_reshape_gt_depths(depth_list, h, w):
    gt_depths = []
    for t in range(len(depth_list)):
        gt_depth = np.load(os.path.join(gt_depth_dir, depth_list[t]))
        gt_depth = gt_depth.astype(np.float32)
        gt_depth = gt_depth.astype(np.float32)
        gt_depth = torch.as_tensor(gt_depth)
        gt_depth = F.interpolate(gt_depth[None,None], (h, w)).squeeze()
        gt_depths.append(gt_depth.numpy())
    return np.stack(gt_depths)

def correct_recon_scaling(pred_depths, pred_poses, gt_depths):
    scaling = np.median((gt_depths[2:, 10:-10, 10:-10] / pred_depths[2:, 10:-10, 10:-10]))
    pred_depths *= scaling
    pred_poses /= scaling
    return pred_depths, pred_poses

parser = argparse.ArgumentParser()
parser.add_argument("recon_folder", type=str)
args = parser.parse_args()

pose_path = os.path.join(args.recon_folder, 'poses.npy')
depths_path = os.path.join(args.recon_folder, 'disps.npy')
timestamps_path = os.path.join(args.recon_folder, 'tstamps.npy')
gt_depth_dir = '/home/ahmedabbas/data/slam_rcm_data/Shell_rcm/depths'

depth_list = sorted(os.listdir(gt_depth_dir))[::2]

pred_poses = np.load(pose_path)
pred_inv_depths = torch.from_numpy(np.load(depths_path))
pred_depths = torch.where(pred_inv_depths>0, 1.0 / pred_inv_depths, pred_inv_depths).numpy()

timestamps = np.load(timestamps_path)
gt_depths = load_reshape_gt_depths(depth_list, pred_depths.shape[1], pred_depths.shape[2])
pred_depths, pred_poses = correct_recon_scaling(pred_depths, pred_poses, gt_depths)

poses_rcm = pops.project_out_rcm(torch.from_numpy(pred_poses)).numpy()
pred_poses_matrix = SE3(torch.from_numpy(pred_poses)).matrix()
pred_poses_rcm_matrix = SE3(torch.from_numpy(poses_rcm)).matrix()

np.save(os.path.join(args.recon_folder, 'poses_scaled.npy'), pred_poses_matrix)
np.save(os.path.join(args.recon_folder, 'poses_rcm_scaled.npy'), pred_poses_rcm_matrix)
# for t in range(len(pred_depths)):
#     pred_depth = pred_depths[t]
#     gt_depth = gt_depths[t]
#     scaling = (pred_depth / gt_depth)
#     print(f'scaling: min: {scaling.min()}, {scaling.max()}, {scaling.mean()}')

#     plt.subplot(131)
#     plt.imshow(gt_depth)
#     plt.subplot(132)
#     plt.imshow(pred_depth, vmin = gt_depth.min(), vmax = gt_depth.max())
#     plt.subplot(133)
#     plt.imshow(scaling, vmin = 0.9, vmax = 1.1)
#     plt.show()