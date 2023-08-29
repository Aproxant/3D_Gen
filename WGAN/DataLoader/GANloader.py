import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from torch.utils.data import Dataset
import torch
import nrrd
import os
from itertools import groupby
from config import cfg
import numpy as np
import pickle 

class GANLoader(Dataset):
    def __init__(self,vox_path,phase):
        with open(getattr(cfg,"GAN_{}_SPLIT".format(phase.upper())), 'rb') as pickle_file:
            self.embeddings=pickle.load(pickle_file)
        self.solid_file=vox_path



    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        model_id = self.embeddings[idx][0]
        label = self.embeddings[idx][1]
        captions = self.embeddings[idx][2]
            
    
        voxel,_=nrrd.read(os.path.join(self.solid_file,model_id,model_id+'.nrrd'))
        voxel = torch.FloatTensor(voxel)
        voxel /=255.

        

        return model_id, label, captions, len(captions) , voxel
    


def check_dataset(dataset, batch_size):
    flag = False
    if len(dataset) % batch_size != 0:
        flag = True
    
    return flag

def collate_embedding(data):
    model_ids, labels, captions, lengths, voxels = zip(*data)

    voxels = torch.stack(voxels, 0)

        
    
    return model_ids,  torch.Tensor(labels), captions, torch.Tensor(list(lengths)),voxels

    
    
    

        
