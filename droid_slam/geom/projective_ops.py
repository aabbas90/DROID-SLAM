import torch
import torch.nn.functional as F

from lietorch import SE3, Sim3, SO3
import numpy as np

MIN_DEPTH = 0.2

def extract_intrinsics(intrinsics):
    return intrinsics[...,None,None,:].unbind(dim=-1)

def coords_grid(ht, wd, **kwargs):
    y, x = torch.meshgrid(
        torch.arange(ht).to(**kwargs).float(),
        torch.arange(wd).to(**kwargs).float())

    return torch.stack([x, y], dim=-1)

def iproj(disps, intrinsics, jacobian=False):
    """ pinhole camera inverse projection """
    ht, wd = disps.shape[2:]
    fx, fy, cx, cy = extract_intrinsics(intrinsics)
    
    y, x = torch.meshgrid(
        torch.arange(ht).to(disps.device).float(),
        torch.arange(wd).to(disps.device).float())

    i = torch.ones_like(disps)
    X = (x - cx) / fx
    Y = (y - cy) / fy
    pts = torch.stack([X, Y, i, disps], dim=-1)

    if jacobian:
        J = torch.zeros_like(pts)
        J[...,-1] = 1.0
        return pts, J

    return pts, None

def proj(Xs, intrinsics, jacobian=False, return_depth=False):
    """ pinhole camera projection """
    fx, fy, cx, cy = extract_intrinsics(intrinsics)
    X, Y, Z, D = Xs.unbind(dim=-1)

    Z = torch.where(Z < 0.5*MIN_DEPTH, torch.ones_like(Z), Z)
    d = 1.0 / Z

    x = fx * (X * d) + cx
    y = fy * (Y * d) + cy
    if return_depth:
        coords = torch.stack([x, y, D*d], dim=-1)
    else:
        coords = torch.stack([x, y], dim=-1)

    if jacobian:
        B, N, H, W = d.shape
        o = torch.zeros_like(d)
        proj_jac = torch.stack([
             fx*d,     o, -fx*X*d*d,  o,
                o,  fy*d, -fy*Y*d*d,  o,
                # o,     o,    -D*d*d,  d,
        ], dim=-1).view(B, N, H, W, 2, 4)

        return coords, proj_jac

    return coords, None

def actp(Gij, X0, jacobian=False, optimize_tx_ty=True):
    """ action on point cloud """
    X1 = Gij[:,:,None,None] * X0
    
    if jacobian:
        X, Y, Z, d = X1.unbind(dim=-1)
        o = torch.zeros_like(d)
        B, N, H, W = d.shape

        if optimize_tx_ty:
            if isinstance(Gij, SE3):
                Ja = torch.stack([
                    d,  o,  o,  o,  Z, -Y,
                    o,  d,  o, -Z,  o,  X, 
                    o,  o,  d,  Y, -X,  o,
                    o,  o,  o,  o,  o,  o,
                ], dim=-1).view(B, N, H, W, 4, 6)

            elif isinstance(Gij, Sim3):
                Ja = torch.stack([
                    d,  o,  o,  o,  Z, -Y,  X,
                    o,  d,  o, -Z,  o,  X,  Y,
                    o,  o,  d,  Y, -X,  o,  Z,
                    o,  o,  o,  o,  o,  o,  o
                ], dim=-1).view(B, N, H, W, 4, 7)
        else:
            if isinstance(Gij, SE3):
                Ja = torch.stack([
                    o,  o,  o,  o,  Z, -Y,
                    o,  o,  o, -Z,  o,  X, 
                    o,  o,  d,  Y, -X,  o,
                    o,  o,  o,  o,  o,  o,
                ], dim=-1).view(B, N, H, W, 4, 6)

            elif isinstance(Gij, Sim3):
                Ja = torch.stack([
                    o,  o,  o,  o,  Z, -Y,  X,
                    o,  o,  o, -Z,  o,  X,  Y,
                    o,  o,  d,  Y, -X,  o,  Z,
                    o,  o,  o,  o,  o,  o,  o
                ], dim=-1).view(B, N, H, W, 4, 7)
        return X1, Ja

    return X1, None

def projective_transform(poses, depths, intrinsics, ii, jj, stereo_rel_pose, jacobian=False, return_depth=False, optimize_tx_ty=True):
    """ map points from ii->jj """

    # inverse project (pinhole)
    X0, Jz = iproj(depths[:,ii], intrinsics[:,ii], jacobian=jacobian)
    # transform
    Gij = poses[:,jj] * poses[:,ii].inv()

    # Gij.data[:,ii==jj] = torch.as_tensor([-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], device="cuda")
    Gij.data[:,ii==jj] = stereo_rel_pose # torch.as_tensor([-5.98142,-0.0955256,0.0978925,0.00368414,-0.0107144,-0.00150446,0.999935], device="cuda")
    X1, Ja = actp(Gij, X0, jacobian=jacobian, optimize_tx_ty=optimize_tx_ty)
    
    # project (pinhole)
    x1, Jp = proj(X1, intrinsics[:,jj], jacobian=jacobian, return_depth=return_depth)

    # exclude points too close to camera
    valid = ((X1[...,2] > MIN_DEPTH) & (X0[...,2] > MIN_DEPTH)).float()
    valid = valid.unsqueeze(-1)

    if jacobian:
        # Ji transforms according to dual adjoint
        Jj = torch.matmul(Jp, Ja) # chain rule
        Ji = -Gij[:,:,None,None,None].adjT(Jj) # chain rule relative pose

        Jz = Gij[:,:,None,None] * Jz
        Jz = torch.matmul(Jp, Jz.unsqueeze(-1))
        return x1, valid, (Ji, Jj, Jz)

    return x1, valid

def induced_flow(poses, disps, intrinsics, ii, jj, stereo_rel_pose):
    """ optical flow induced by camera motion """

    ht, wd = disps.shape[2:]
    y, x = torch.meshgrid(
        torch.arange(ht).to(disps.device).float(),
        torch.arange(wd).to(disps.device).float())

    coords0 = torch.stack([x, y], dim=-1)
    coords1, valid = projective_transform(poses, disps, intrinsics, ii, jj, stereo_rel_pose, False)

    return coords1[...,:2] - coords0, valid

def find_rcm(poses):
    lines = []
    for pose in poses:
        pose_matrix = SE3(pose).matrix()
        rot = (pose_matrix[:3, :3])
        transl = (pose_matrix[:3, 3])
        camera_center = -torch.matmul(rot.T, transl)
        camera_negative_viewing_direction = torch.matmul(rot.T, torch.FloatTensor([0, 0, 1]).to(rot.device))
        lines.append((camera_center.cpu().numpy(), camera_negative_viewing_direction.cpu().numpy()))
    
    A = []
    B = []
    
    for (P, d) in lines:
        P = np.array(P)
        d = np.array(d) / np.linalg.norm(d)  # Ensure direction vector is normalized
        
        a = np.eye(3) - np.outer(d, d)
        b = a @ P
        
        A.append(a)
        B.append(b)
    
    A = np.sum(A, axis=0)
    B = np.sum(B, axis=0)
    
    rcm = np.linalg.solve(A, B)
    rcm = torch.from_numpy(rcm).to(poses.device).to(torch.float32)
    return rcm

def optimize_out_rcm(poses, rcm):
    poses.requires_grad=True
    opt = torch.optim.Adam([poses], lr=1e-3)
    num_itr = 100
    for itr in range(num_itr):
        poses_se3 = SE3(poses)
        original_centers_ref = poses_se3.inv() * torch.tensor([0.0,0.0,0.0], device=poses.device).unsqueeze(0)
        new_centers_ref = original_centers_ref - rcm.squeeze(2)
        new_translation_ref = - (poses_se3 * new_centers_ref - poses_se3.translation()[:, :3])
        translation_x_error = torch.abs(new_translation_ref[:, 0]).mean()
        translation_y_error = torch.abs(new_translation_ref[:, 1]).mean()
        print(f'translation_x_error: {translation_x_error.item()}, translation_y_error: {translation_y_error.item()}')
        loss = translation_x_error + translation_y_error
        loss.backward()
        opt.step()
        opt.zero_grad()
    return poses

def project_out_rcm(poses, rcm):
    poses_se3 = SE3(poses)
    PM = poses_se3.matrix()
    PM_inv = poses_se3.inv().matrix()
    # original_centers_ref = -torch.matmul(PM_inv[:, :3, :3], PM[:, :3, 3:4])
    original_centers_ref = poses_se3.inv() * torch.tensor([0.0,0.0,0.0], device=poses.device).unsqueeze(0)
    new_centers_ref = original_centers_ref - rcm.unsqueeze(0)
    # new_translation_ref = -torch.matmul(PM[:, :3, :3], new_centers_ref)
    new_translation_ref = - (poses_se3 * new_centers_ref - poses_se3.translation()[:, :3])
    translation_x_error = torch.abs(new_translation_ref[:, 0]).mean().item()
    translation_y_error = torch.abs(new_translation_ref[:, 1]).mean().item()
    print(f'translation_x_error: {translation_x_error}, translation_y_error: {translation_y_error}')
    new_translation_ref[:, :2] = 0
    new_translation_original_coords_ref = -torch.matmul(PM[:, :3, :3], -torch.matmul(PM_inv[:, :3, :3], new_translation_ref.unsqueeze(2)) + rcm.unsqueeze(0).unsqueeze(2)).squeeze(2)
    new_poses = poses.clone()
    new_poses[:, :3] = new_translation_original_coords_ref
    return new_poses

    # poses_se3 = SE3(poses)
    # translation_original = poses_se3.translation()[:, :4].unsqueeze(1).clone()
    # translation_original[:, :, -1] = 1.0

    # poses_wo_translation = poses.clone()
    # poses_wo_translation[:, :3] = 0.0
    
    # poses_so3 = SE3(poses_wo_translation)
    # poses_so3_inv = poses_so3.inv()

    # original_centers = -(poses_so3_inv * translation_original.squeeze(1))[:, :3]
    # new_centers = original_centers - rcm.squeeze(-1)
    # new_translation = -(poses_so3 * new_centers) 

    # # print(f'new_translation: {new_translation.shape} \n {new_translation}')
    # # set x and y translation to 0.
    # new_translation[:, :2] = 0
    # new_translation_original_coords = - (poses_so3 * (-(poses_so3_inv * new_translation) + rcm.squeeze(-1)))
    # PM[:, :3, 3:4] = -torch.matmul(PM[:, :3, :3], -torch.matmul(PM_inv[:, :3, :3], new_translation) + rcm)
