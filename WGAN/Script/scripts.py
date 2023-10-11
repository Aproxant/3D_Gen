import trimesh
import numpy as np
import nrrd
import os
import torch
from config import cfg

def vizualizer_stanford_data(vox_color):
    vox_color=np.array(vox_color)
    vox_color = np.rollaxis(vox_color, 0,4)
    ind=vox_color[:,:,:]!=[0,0,0,0]
    vox = ind[:, :, :,0]
    z=trimesh.voxel.VoxelGrid(vox)
    l=z.as_boxes(colors=vox_color)
    return l.show()

def augment_voxel_tensor(voxel_tensor, max_noise=10):
    augmented_voxel_tensor = torch.clone(voxel_tensor)
    if (voxel_tensor.ndim == 4) and (voxel_tensor.shape[0] != 1) and (max_noise > 0):
        noise_val = float(np.random.randint(-max_noise, high=(max_noise + 1))) / 255
        augmented_voxel_tensor[:3, :, :, :] += noise_val
        augmented_voxel_tensor = np.clip(augmented_voxel_tensor, 0., 1.)
    return augmented_voxel_tensor

def sample_z():
    """Returns a numpy array batch of the sampled noise. Call this to get a noise batch sample."""

    if cfg.GAN_NOISE_DIST == 'gaussian':
        return torch.Tensor(np.random.normal(loc=cfg.GAN_NOISE_MEAN,
                                scale=cfg.GAN_NOISE_STDDEV,
                                size=[1, cfg.GAN_NOISE_SIZE]))
    elif cfg.GAN_NOISE_DIST== 'uniform':
        return torch.Tensor(np.random.uniform(low=-cfg.GAN_NOISE_UNIF_ABS_MAX,
                                 high=cfg.GAN_NOISE_UNIF_ABS_MAX,
                                 size=[1, cfg.GAN_NOISE_SIZE]))
    else:
        raise ValueError('Sample distribution must be uniform or gaussian.')
