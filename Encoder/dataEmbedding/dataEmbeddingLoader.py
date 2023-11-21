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
from transformers import AlbertTokenizer


class GenerateDataLoader(Dataset):
    def __init__(self, input_data,dict_word2idx):
        self.embedding_data = input_data
        self.dict_word2idx=dict_word2idx

        

    def __len__(self):
        return len(self.embedding_data)

    def __getitem__(self, idx):
        model_id = self.embedding_data[idx][0]
        main_category=self.embedding_data[idx][1]
        label = self.embedding_data[idx][2]
        caption = self.embedding_data[idx][3]
            
        
        indices = []
            
        if cfg.EMBEDDING_ALBERT:
            indices=' '.join(caption)
        else:
            for word in caption:
                if word in self.dict_word2idx.keys():
                    indices.append(int(self.dict_word2idx[word]))

            if len(indices)>cfg.EMBEDDING_CAPTION_LEN:
                indices=indices[:cfg.EMBEDDING_CAPTION_LEN]

        return model_id,main_category, label, indices
    


def check_dataset(dataset, batch_size):
    flag = False
    if len(dataset) % batch_size != 0:
        flag = True
    
    return flag

def collate_embedding(data):
    model_ids, main_categories,labels, captions= zip(*data)
    if cfg.EMBEDDING_ALBERT:
        tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2',truncation_side='right')
        merge_caps = tokenizer(captions, padding=True,max_length=cfg.EMBEDDING_CAPTION_LEN,truncation=True,return_tensors='pt')
    else:

        merge_caps = torch.zeros(len(captions), cfg.EMBEDDING_CAPTION_LEN).long()

        for i, cap in enumerate(captions):
            end = int(len(captions[i]))
            merge_caps[i, :end] = torch.LongTensor(cap[:end])

    return model_ids, torch.IntTensor(main_categories), torch.IntTensor(labels), merge_caps


    
    

        
