from torch.utils.data import Dataset
import torch
import nrrd
import os
from config import cfg
import numpy as np
from Script.scripts import augment_voxel_tensor,sample_z
        

class GANLoader(Dataset):
    def __init__(self,data,indexes,phase):
        self.data=data
        self.indexes=indexes
        self.solid_file=cfg.GAN_VOXEL_FOLDER
        self.phase=phase


    def __len__(self):
        return len(self.indexes)

    def __getitem__(self, idx):
        id= self.indexes[idx]
        elem1=self.data[id[0]]
        elem2=self.data[id[1]]
        model_id=elem1[0]

        learned_embedding = elem2[1]

    
        voxel,_=nrrd.read(os.path.join(self.solid_file,model_id,model_id+'.nrrd'))
        voxel = torch.FloatTensor(voxel)
        voxel /=255.
        if self.phase=='train':
            voxel = augment_voxel_tensor(voxel,max_noise=cfg.GAN_TRAIN_AUGMENT_MAX)
        learned_embedding=torch.Tensor(learned_embedding)
        learned_embedding=torch.cat((learned_embedding.unsqueeze(0),sample_z()),1).squeeze(0)


        return model_id,learned_embedding,voxel
    """
    def __getitem__(self, idx):
        model_id = self.embeddings[idx][0]
        label = self.embeddings[idx][1]
        learned_embedding = self.embeddings[idx][2]
        raw_caption = self.embeddings[idx][3]

    
        voxel,_=nrrd.read(os.path.join(self.solid_file,model_id,model_id+'.nrrd'))
        voxel = torch.FloatTensor(voxel)
        voxel /=255.
        if self.phase=='train':
            voxel = augment_voxel_tensor(voxel,max_noise=cfg.GAN_TRAIN_AUGMENT_MAX)
        
        learned_embedding=torch.cat((learned_embedding.unsqueeze(0),sample_z()),1).squeeze(0)


        return model_id, label, torch.Tensor(learned_embedding), raw_caption , voxel
    
    """

def check_dataset(dataset, batch_size):
    flag = False
    if len(dataset) % batch_size != 0:
        flag = True
    
    return flag


    
    

        
