import torch, os
from lietorch import SE3
import numpy as np
import evo
from evo.core.trajectory import PoseTrajectory3D
from evo.tools import file_interface
from evo.core import sync
import evo.main_ape as main_ape
import evo.main_rpe as main_rpe
from evo.core.metrics import PoseRelation
import argparse
from scipy.spatial.transform import Rotation as R

def invert_traj(traj):
    xyz = traj.positions_xyz
    # shift -1 column -> w in back column
    quat = np.roll(traj.orientations_quat_wxyz, -1, axis=1)
    mat = np.column_stack((xyz, quat))
    mat_inv = SE3(torch.from_numpy(mat)).inv().data.cpu().numpy()
    inv_traj = PoseTrajectory3D(positions_xyz=mat_inv[:, :3], orientations_quat_wxyz=np.concatenate((mat_inv[:, -1:], mat_inv[:, 3:6]), 1), timestamps=traj.timestamps)
    return inv_traj

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_poses_path", help="path to saved reconstruction", default='/home/ahmedabbas/data/slam_rcm_data_zdisp_motion_blur/Shell_rcm/cam2world/')
    parser.add_argument("--reconstruction_path", help="path to saved reconstruction")
    args = parser.parse_args()

    gt_mats = []
    gt_xyz = []
    gt_quat = []
    for posefile in sorted(os.listdir(args.gt_poses_path)):
        gt_mat = np.loadtxt(os.path.join(args.gt_poses_path, posefile))
        gt_xyz.append(gt_mat[:3, 3])
        gt_quat.append(R.from_matrix(gt_mat[:3, :3]).as_quat())
        gt_mats.append(gt_mat)
        
    # gt_xyz = np.stack(gt_xyz)
    # gt_quat = np.stack(gt_quat)
    # gt_poses = np.concatenate((gt_xyz, gt_quat), 1)
    # gt_liet = SE3(torch.from_numpy(gt_poses)).inv().data.cpu().numpy()
    # gt_xyz = gt_liet[:, :3]
    # gt_quat = gt_liet[:, 3:]

    tstamps = np.arange(len(gt_xyz)).astype(np.float64)
    traj_est = np.load("reconstructions/{}/traj_est.npy".format(args.reconstruction_path))
    traj_est = SE3(torch.from_numpy(traj_est)).matrix().cpu().numpy()

    # traj_est[:, 3:] = np.concatenate((traj_est[:,-1:], traj_est[:,3:6]), 1)
    traj_est = [a for a in traj_est]

    traj_est = PoseTrajectory3D(poses_se3=traj_est, timestamps=tstamps)
    traj_ref = PoseTrajectory3D(poses_se3=gt_mats, timestamps=tstamps)
    
    traj_ref, traj_est = sync.associate_trajectories(traj_ref, traj_est)
    result = main_ape.ape(traj_ref, traj_est, est_name='traj', 
        pose_relation=PoseRelation.translation_part, align=True, correct_scale=True)

    print(result)


    # traj_est = invert_traj(traj_est)
    # traj_ref = invert_traj(traj_ref)
    
    # file_interface.write_tum_trajectory_file('pred_traj_baseline.txt', traj_est, confirm_overwrite=False)
    file_interface.write_tum_trajectory_file('ref_traj_motion_blur.txt', traj_ref, False)