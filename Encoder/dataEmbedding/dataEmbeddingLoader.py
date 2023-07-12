import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from torch.utils.data import Dataset
import torch
import nrrd
import os
from itertools import groupby

from config import cfg

class GenerateDataLoader(Dataset):
    def __init__(self, input_data,dataset_name,path,labelEnc,dict_word2idx,mode):
        self.embedding_data = input_data
        self.dataset=dataset_name
        self.solid_file=path
        self.labelEnc=labelEnc
        self.dict_word2idx=dict_word2idx
        self.mode=mode
        
        self.sw_nltk = stopwords.words('english')
        
        self.lemmatizer = WordNetLemmatizer() 


    def __len__(self):
        return len(self.embedding_data)

    def __getitem__(self, idx):
        model_id = self.embedding_data[idx][0]
        label = self.embedding_data[idx][1]
        caption = self.embedding_data[idx][2]
        if self.mode=='train':           
            encodedLabel=self.embedding_data[idx][3]
        else:
            encodedLabel=self.labelEnc[label]
            words = [self.lemmatizer.lemmatize(word) for word in caption.split() if word.lower() not in self.sw_nltk]
            caption = " ".join(words)
            

            
        indices = []
            
        for word in caption.split():
            if word in self.dict_word2idx.keys():
                indices.append(int(self.dict_word2idx[word]))
            else:
                indices.append(int(self.dict_word2idx["<UNK>"]))
        caption=indices

        length = len(caption)            

        if self.dataset== 'stanford_data':
            voxel,_=nrrd.read(os.path.join(self.solid_file,model_id,model_id+'.nrrd'))
            voxel = torch.FloatTensor(voxel)
            voxel /= 255.

        elif self.dataset == 'primitives':
            voxel,_ = nrrd.read(os.path.join(self.solid_file,'primitives.v2',label,label+'_'+model_id+'.nrrd'))
            voxel = torch.FloatTensor(voxel)
            voxel /= 255.   


        return model_id, label, encodedLabel, caption, length, voxel
    




def check_dataset(dataset, batch_size):
    flag = False
    if len(dataset) % batch_size != 0:
        flag = True
    
    return flag

def collate_embedding(data):
    model_ids, labels,encodedLabel, captions, lengths, voxels = zip(*data)
    voxels = torch.stack(voxels, 0)  
    curr_len=max(lengths)

    if cfg.EMBEDDING_CAPTION_LEN<max(lengths):
        max_len=cfg.EMBEDDING_CAPTION_LEN
    else:
        max_len=curr_len
    
    merge_caps = torch.zeros(len(captions), max_len).long()

    for i, cap in enumerate(captions):
        #tokens = tokenizer.tokenize(cap,padding='max_length', truncation=True,max_length=max_len)
        #merge_caps[i] = torch.LongTensor(tokenizer.convert_tokens_to_ids(tokens))
        end = int(lengths[i])
        if end>max_len:
            cap,end=take_most_important_words(cap,max_len)
            
        merge_caps[i, :end] = torch.LongTensor(cap[:end])
    
    return model_ids,  labels,torch.Tensor(encodedLabel), merge_caps, torch.Tensor(list(lengths)),voxels

def take_most_important_words(cap,length):
    max_w=[]
    for i,elem in enumerate(cap):
        max_w.append((elem,i))
    
    
    max_w=[i for i in max_w if i[0]!='1']
    """
    max_w=[next(g) for _, g in groupby(max_w, key=lambda x:x[0])] #to nie dizała
    print(max_w)
    #max_w.sort(key=lambda tup: tup[0], reverse=False) 
    print(max_w)
    """

    if len(max_w)>length:
        #max_w=max_w[:length]
        max_w.sort(key=lambda tup: tup[0], reverse=False) 

        max_w=max_w[:length]

        max_w.sort(key=lambda tup: tup[1], reverse=False) 

        return [i[0] for i in max_w],length
    
    max_w.sort(key=lambda tup: tup[1], reverse=False)


    return [i[0] for i in max_w], len(max_w)

    
    
    

        
