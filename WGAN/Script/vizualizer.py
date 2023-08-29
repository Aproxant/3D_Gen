import trimesh
import numpy as np
import nrrd
import os
def vizualizer_stanford_data(vox_color):
    vox_color=np.array(vox_color)
    vox_color = np.rollaxis(vox_color, 0,4)
    ind=vox_color[:,:,:]!=[0,0,0,0]
    vox = ind[:, :, :,0]
    z=trimesh.voxel.VoxelGrid(vox)
    l=z.as_boxes(colors=vox_color)
    return l.show()
