import numpy as np
import torch
import lietorch
import droid_backends

from torch.multiprocessing import Process, Queue, Lock, Value
from collections import OrderedDict

from droid_net import cvx_upsample
import geom.projective_ops as pops
import geom.ba as bapy
import os
from lietorch import SE3

class DepthVideo:
    def __init__(self, image_size=[480, 640], buffer=1024, stereo=False, device="cuda:0"):
                
        # current keyframe count
        self.counter = Value('i', 0)
        self.ready = Value('i', 0)
        self.ht = ht = image_size[0]
        self.wd = wd = image_size[1]

        ### state attributes ###
        self.tstamp = torch.zeros(buffer, device="cuda", dtype=torch.float).share_memory_()
        self.images = torch.zeros(buffer, 3, ht, wd, device="cuda", dtype=torch.uint8)
        self.dirty = torch.zeros(buffer, device="cuda", dtype=torch.bool).share_memory_()
        self.red = torch.zeros(buffer, device="cuda", dtype=torch.bool).share_memory_()
        self.poses = torch.zeros(buffer, 7, device="cuda", dtype=torch.float).share_memory_()
        self.disps = torch.ones(buffer, ht//8, wd//8, device="cuda", dtype=torch.float).share_memory_()
        self.disps_sens = torch.zeros(buffer, ht//8, wd//8, device="cuda", dtype=torch.float).share_memory_()
        self.disps_up = torch.zeros(buffer, ht, wd, device="cuda", dtype=torch.float).share_memory_()
        self.intrinsics = torch.zeros(buffer, 4, device="cuda", dtype=torch.float).share_memory_()

        self.stereo = stereo
        c = 1 if not self.stereo else 2

        ### feature attributes ###
        self.fmaps = torch.zeros(buffer, c, 128, ht//8, wd//8, dtype=torch.half, device="cuda").share_memory_()
        self.nets = torch.zeros(buffer, 128, ht//8, wd//8, dtype=torch.half, device="cuda").share_memory_()
        self.inps = torch.zeros(buffer, 128, ht//8, wd//8, dtype=torch.half, device="cuda").share_memory_()

        # initialize poses to identity transformation
        self.poses[:] = torch.as_tensor([0, 0, 0, 0, 0, 0, 1], dtype=torch.float, device="cuda")
        if 'RCM' in os.environ:
            print('depth_video.py: Initializing with RCM. ')
            self.poses[:, 2] = -3 # -1.10 # -3.13
        self.stereo_rel_pose = torch.zeros(buffer, 7, device="cuda", dtype=torch.float).share_memory_()
        self.stereo_rel_pose[:] = torch.as_tensor([-0.1, 0, 0, 0, 0, 0, 1.0], dtype=torch.float, device="cuda")

    def get_lock(self):
        return self.counter.get_lock()

    def __item_setter(self, index, item):
        if isinstance(index, int) and index >= self.counter.value:
            self.counter.value = index + 1
        
        elif isinstance(index, torch.Tensor) and index.max().item() > self.counter.value:
            self.counter.value = index.max().item() + 1

        # self.dirty[index] = True
        self.tstamp[index] = item[0]
        self.images[index] = item[1]

        if item[2] is not None:
            self.poses[index] = item[2]

        if item[3] is not None:
            self.disps[index] = item[3]

        if item[4] is not None:
            depth = item[4][3::8,3::8]
            self.disps_sens[index] = torch.where(depth>0, 1.0/depth, depth)

        if item[5] is not None:
            self.intrinsics[index] = item[5]

        if len(item) > 6:
            self.fmaps[index] = item[6]

        if len(item) > 7:
            self.nets[index] = item[7]

        if len(item) > 8:
            self.inps[index] = item[8]

        if len(item) > 9:
            if item[9] is not None:
                self.stereo_rel_pose[index] = item[9]

    def __setitem__(self, index, item):
        with self.get_lock():
            self.__item_setter(index, item)

    def __getitem__(self, index):
        """ index the depth video """

        with self.get_lock():
            # support negative indexing
            if isinstance(index, int) and index < 0:
                index = self.counter.value + index

            item = (
                self.poses[index],
                self.disps[index],
                self.intrinsics[index],
                self.fmaps[index],
                self.nets[index],
                self.inps[index])

        return item

    def append(self, *item):
        with self.get_lock():
            self.__item_setter(self.counter.value, item)


    ### geometric operations ###

    @staticmethod
    def format_indicies(ii, jj):
        """ to device, long, {-1} """

        if not isinstance(ii, torch.Tensor):
            ii = torch.as_tensor(ii)

        if not isinstance(jj, torch.Tensor):
            jj = torch.as_tensor(jj)

        ii = ii.to(device="cuda", dtype=torch.long).reshape(-1)
        jj = jj.to(device="cuda", dtype=torch.long).reshape(-1)

        return ii, jj

    def upsample(self, ix, mask):
        """ upsample disparity """

        disps_up = cvx_upsample(self.disps[ix].unsqueeze(-1), mask)
        self.disps_up[ix] = disps_up.squeeze()

    def normalize(self):
        """ normalize depth and poses """
        with self.get_lock():
            s = self.disps[:self.counter.value].mean()
            self.disps[:self.counter.value] /= s
            self.poses[:self.counter.value,:3] *= s
            self.dirty[:self.counter.value] = True


    def reproject(self, ii, jj):
        """ project points from ii -> jj """
        ii, jj = DepthVideo.format_indicies(ii, jj)
        Gs = lietorch.SE3(self.poses[None])

        coords, valid_mask = \
            pops.projective_transform(Gs, self.disps[None], self.intrinsics[None], ii, jj, self.stereo_rel_pose[0])

        return coords, valid_mask

    def distance(self, ii=None, jj=None, beta=0.3, bidirectional=True):
        """ frame distance metric """

        return_matrix = False
        if ii is None:
            return_matrix = True
            N = self.counter.value
            ii, jj = torch.meshgrid(torch.arange(N), torch.arange(N))
        
        ii, jj = DepthVideo.format_indicies(ii, jj)

        if bidirectional:

            poses = self.poses[:self.counter.value].clone()

            d1 = droid_backends.frame_distance(
                poses, self.disps, self.intrinsics[0], ii, jj, beta)

            d2 = droid_backends.frame_distance(
                poses, self.disps, self.intrinsics[0], jj, ii, beta)

            d = .5 * (d1 + d2)

        else:
            d = droid_backends.frame_distance(
                self.poses, self.disps, self.intrinsics[0], ii, jj, beta)

        if return_matrix:
            return d.reshape(N, N)

        return d
    
    def project_out_rcm(self):
        from lietorch import SE3
        poses_torch = self.poses[:, :self.counter.value]
        rcm = pops.find_rcm(poses_torch).squeeze()
        poses_torch_rcm = pops.project_out_rcm(poses_torch, rcm)
        poses_world = SE3(poses_torch_rcm).inv()
        poses_world.data[:, :3] -= rcm.unsqueeze(0)
        self.poses[:, :self.counter.value] = poses_world.inv().data

    def ba(self, target, weight, eta, ii, jj, t0=1, t1=None, itrs=2, lm=1e-4, ep=0.1, motion_only=False, rcm_regularizer_strength=0.0):
        """ dense bundle adjustment (DBA) """

        with self.get_lock():

            # [t0, t1] window of bundle adjustment optimization
            if t1 is None:
                t1 = max(ii.max().item(), jj.max().item()) + 1

            # poses_py = SE3(self.poses.unsqueeze(0))[:, :t1]
            # depths_py = self.disps.clone().unsqueeze(0)[:, :t1]
            # intrinsics_py = self.intrinsics.clone().unsqueeze(0)[:, :t1]
            # target_py = torch.movedim(target.clone().unsqueeze(0), 2, -1).contiguous()
            # weight_py = torch.movedim(weight.clone().unsqueeze(0), 2, -1).contiguous()
            # for itr in range(itrs):
            #     if not motion_only:
            #         poses_py, depths_py = bapy.BA(target_py, weight_py, eta, poses_py, depths_py, intrinsics_py, ii, jj, stereo_rel_pose=self.stereo_rel_pose[0], fixedp=t0, rig=1, rcm_regularizer_strength=rcm_regularizer_strength)
            #     else:
            #         poses_py = bapy.MoBA(target_py, weight_py, eta, poses_py, depths_py, intrinsics_py, ii, jj, stereo_rel_pose=self.stereo_rel_pose[0], fixedp=t0, rig=1, rcm_regularizer_strength=rcm_regularizer_strength)

            # depths_py = depths_py.squeeze(0)
            # poses_py = poses_py.data.squeeze(0)
            # self.poses[:t1] = poses_py[:t1]
            # self.disps[:t1] = depths_py[:t1] 

            dx, dz = droid_backends.ba(self.poses, self.disps, self.intrinsics[0], self.stereo_rel_pose[0], self.disps_sens,
                target, weight, eta, ii, jj, t0, t1, itrs, lm, ep, motion_only, True)

            # # diff = self.poses - prev_poses
            # tx_var = torch.var(self.poses[:t1, 0])
            # ty_var = torch.var(self.poses[:t1, 1])
            # tz_var = torch.var(self.poses[:t1, 2])
            # print(f'tx_var: {tx_var:.4f}, ty_var: {ty_var:.4f}, tz_var: {tz_var:.4f}')

            self.disps.clamp_(min=0.001)
            # if self.counter.value > 20:
            #     poses_torch = self.poses[:self.counter.value]
            #     rcm = pops.find_rcm(poses_torch).squeeze()
            #     poses_torch_rcm = pops.project_out_rcm(poses_torch, rcm)
            #     poses_world = SE3(poses_torch_rcm).inv()
            #     poses_world.data[:, :3] -= rcm.unsqueeze(0)
            #     self.poses[:self.counter.value] = poses_world.inv().data
            #     print(f'projected RCM @ {rcm}')
            # print(poses_new.data[:, :3])

            # warmup_buffer = 20
            # ignore_last = 10
            # rcm_on_atleast = 10
            # if self.counter.value > warmup_buffer + rcm_on_atleast + ignore_last:
            #     end_frame = self.counter.value - ignore_last
            #     start_frame = end_frame - rcm_on_atleast
            #     rcm = pops.find_rcm(self.poses[:warmup_buffer]).unsqueeze(0).unsqueeze(-1)
            #     self.poses[: -ignore_last] = pops.project_out_rcm(self.poses[: -ignore_last], rcm)