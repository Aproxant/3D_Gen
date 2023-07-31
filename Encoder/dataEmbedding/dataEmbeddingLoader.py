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

class GenerateDataLoader(Dataset):
    def __init__(self, input_data,path,dict_word2idx,mode):
        self.embedding_data = input_data
        self.solid_file=path
        self.dict_word2idx=dict_word2idx
        self.mode=mode
        

    def __len__(self):
        return len(self.embedding_data)

    def __getitem__(self, idx):
        model_id = self.embedding_data[idx][0]
        label = self.embedding_data[idx][1]
        captions = self.embedding_data[idx][2]
            
        indices = []
            
        for word in captions:
            if word in self.dict_word2idx.keys():
                indices.append(int(self.dict_word2idx[word]))
            else:
                indices.append(int(self.dict_word2idx["<UNK>"]))
        
        if len(indices)>cfg.EMBEDDING_CAPTION_LEN:
            indices=indices[:cfg.EMBEDDING_CAPTION_LEN]

        length = len(indices)            


        voxel,_=nrrd.read(os.path.join(self.solid_file,model_id,model_id+'.nrrd'))
        voxel = torch.FloatTensor(voxel)
        voxel /=255.



        return model_id, label, indices, length, voxel
    


def check_dataset(dataset, batch_size):
    flag = False
    if len(dataset) % batch_size != 0:
        flag = True
    
    return flag

def collate_embedding(data):
    model_ids, labels, captions, lengths, voxels = zip(*data)
    voxels = torch.stack(voxels, 0)

    merge_caps = torch.zeros(len(captions), cfg.EMBEDDING_CAPTION_LEN).long()

    for i, cap in enumerate(captions):
        end = int(lengths[i])
        merge_caps[i, :end] = torch.LongTensor(cap[:end])
    
    return model_ids,  torch.Tensor(labels), merge_caps, torch.Tensor(list(lengths)),voxels

def take_most_important_words(cap,length):
    max_w=[]
    for i,elem in enumerate(cap):
        max_w.append((elem,i))
    
    max_w=[i for i in max_w if i[0]!='1']

    max_w=[next(g) for _, g in groupby(max_w, key=lambda x:x[0])]

    max_w.sort(key=lambda tup: tup[0], reverse=False) 
    

    if len(max_w)>length:
        max_w=max_w[:length]
        max_w.sort(key=lambda tup: tup[1], reverse=False) 
        max_w=[i[0] for i in max_w]
        return max_w,length
    
    max_w.sort(key=lambda tup: tup[1], reverse=False)

    return [i[0] for i in max_w], len(max_w)

    
    
    

        
