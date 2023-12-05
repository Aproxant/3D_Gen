import trimesh
import numpy as np
import nrrd
import os
import torch
from config import cfg

import trimesh
from trimesh import voxel
from Encoder.dataEmbedding.dataEmbedding import Read_Load_BuildBatch
from IPython.display import display
import point_cloud_utils as pcu
from tqdm import tqdm


def vizualizer_stanford_data(vox_color):
    vox_color=np.array(vox_color)
    vox_color = np.rollaxis(vox_color, 0,4)
    ind=vox_color[:,:,:]!=[0,0,0,0]
    vox = ind[:, :, :,0]
    z=trimesh.VoxelGrid(vox)
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

def evaluateShape(real_shape,fake_shape):
    _,w,d,h=real_shape.shape
    condition1 = fake_shape[0, :] < 0.5
    condition2 = fake_shape[0, :] >= 0.5
        

    fake_shape[0,condition1] = 0.0

    fake_shape[0,condition2] = 1.0


    real_shape=real_shape.clone()[3].to(torch.int).to(cfg.DEVICE)
    fake_shape=fake_shape.clone().squeeze(0).to(torch.int)



    hamming_distance = torch.sum(torch.bitwise_xor(real_shape, fake_shape))
    
    hamming_distance = hamming_distance /(h*w*d)

    intersection = torch.sum(torch.logical_and(real_shape, fake_shape))
    union = torch.sum(torch.logical_or(real_shape, fake_shape))
    jaccard_similarity = intersection / union

    mask_real = (real_shape != 0)
    mask_fake=  (fake_shape != 0)
    compare_ten=torch.sum(mask_fake==mask_real)
    return compare_ten/(w*d*h), hamming_distance,jaccard_similarity
    

def loadOneElem(idx,data):


    model_id=data[idx][0]
    learned_embedding = data[idx][3]
    stanData=Read_Load_BuildBatch(cfg.EMBEDDING_BATCH_SIZE)

    values = [stanData.dict_idx2word[key] for key in data[idx][4].tolist() if key!=0]

    print(' '.join(values))

  
    
    voxel,_=nrrd.read(os.path.join(cfg.GAN_VOXEL_FOLDER,model_id,model_id+'.nrrd'))
    voxel = torch.FloatTensor(voxel)
    voxel /=255.

    learned_embedding=torch.Tensor(learned_embedding)
    
    learned_embedding=torch.cat((learned_embedding.unsqueeze(0),sample_z().to(cfg.DEVICE)),1).squeeze(0)


    return model_id,learned_embedding,voxel

def vizualizer_stanford_data(vox_color,data):

    if data=='fake':
        condition1 = vox_color[0, :] < 0.5
        condition2 = vox_color[0, :] >= 0.5
        

        vox_color[0,condition1] = 0.0

        vox_color[0,condition2] = 1.0

        vox_color=vox_color.expand(4, -1, -1, -1)
    

    vox_color=np.array(vox_color)
    vox_color = np.rollaxis(vox_color, 0,4)
    ind=vox_color[:,:,:]!=[0,0,0,0]
    vox = ind[:, :, :,0]
    z=voxel.VoxelGrid(vox)

    l=z.as_boxes()



    l.apply_transform(trimesh.transformations.rotation_matrix(
    np.radians(90.), [0, 0, 1], point=None
    ))

    l.apply_transform(trimesh.transformations.rotation_matrix(
    np.radians(-90.0), [0, 1, 0], point=None
    ))
    scene = trimesh.Scene([l])

    viewer = scene.show()
    display(viewer)

def evalTestSet(data,gen):
    #stanData=Read_Load_BuildBatch(cfg.EMBEDDING_BATCH_SIZE)
    
    metrics={"hausdorff_distance_one_side": [],
            "hausdorff_distance": [],
            "chamfer_distance": [],   
            "modelId": [],
            "cap": []        
            }
    for elem in tqdm(data):
        model_id=elem[0]
        learned_embedding = elem[3]
        cap=elem[4]
        #values = [stanData.dict_idx2word[key] for key in elem[4].tolist() if key!=0]
   
        voxel,_=nrrd.read(os.path.join(cfg.GAN_VOXEL_FOLDER,model_id,model_id+'.nrrd'))
        real_shape = torch.FloatTensor(voxel)
        real_shape /=255.

        learned_embedding=torch.Tensor(learned_embedding)
    
        learned_embedding=torch.cat((learned_embedding.unsqueeze(0),sample_z().to(cfg.DEVICE)),1).squeeze(0)

        p=gen(learned_embedding.unsqueeze(0))
        fake_shape=p['sigmoid_output'].squeeze(0).cpu().detach()
    
        condition1 = fake_shape[0, :] < 0.5
        condition2 = fake_shape[0, :] >= 0.5
        

        fake_shape[0,condition1] = 0.0

        fake_shape[0,condition2] = 1.0


        fake = fake_shape.to(torch.int).squeeze(0).numpy()
        real = real_shape.to(torch.int)[3].squeeze(0).numpy()


        # Get coordinates of non-zero elements
        coords1 = np.array(list(zip(*np.nonzero(real))),dtype=float)
        coords2 = np.array(list(zip(*np.nonzero(fake))),dtype=float)

        metrics['chamfer_distance'].append(pcu.chamfer_distance(coords1, coords2))
        metrics['hausdorff_distance_one_side'].append(pcu.one_sided_hausdorff_distance(coords2, coords1))
        metrics['hausdorff_distance'].append(pcu.hausdorff_distance(coords2, coords1))
        metrics['modelId'].append(model_id)
        metrics['cap'].append(cap)

        #metrics["EMD"].append(pcu.earth_movers_distance(coords1,coords2))
    return  metrics
