from lietorch import SE3
import torch
import numpy as np
import scipy.spatial
from torch_scatter import scatter_sum


def get_random_rot(max_angle = 20):
    # R = scipy.spatial.transform.Rotation.random().as_matrix()
    # I = np.eye(3)
    # R[0, :] = I[0, :]
    # R[1, :] = I[1, :]
    # R[2, :] = I[2, :]
    angles = np.random.rand(3) * max_angle
    q = scipy.spatial.transform.Rotation.from_euler('xyz', angles, degrees=True).as_quat()
    return q

orig_q = get_random_rot(80)
orig_t = np.random.rand(3)*10.0
data = np.concatenate((orig_t, orig_q), 0)
orig_pose = SE3(torch.from_numpy(data).to('cuda').unsqueeze(0))

init_q = get_random_rot(5)
init_t = np.zeros(3)
init_t[2] = 2.0
data = np.concatenate((init_t, init_q), 0)

delta = (torch.from_numpy(data).to('cuda').unsqueeze(0))
delta.requires_grad=True
opt = torch.optim.Adam([delta], 1e-1)

print(f'orig_pose.data: {orig_pose.data}')
for i in range(1000):
    new_pose = orig_pose.retr(delta)
    translation_error = torch.square(new_pose.data[:, :2] - orig_pose.data[:, :2]).sum()
    translation_error.backward()
    opt.step()
    opt.zero_grad()
    print(f'translation_error: {translation_error.item()}')

print(f'orig_pose.data: {orig_pose.data}')
print(f'new_pose.data: {new_pose.data}')
print(f'diff: {new_pose.data - orig_pose.data}')
