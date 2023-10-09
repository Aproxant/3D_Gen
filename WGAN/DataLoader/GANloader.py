import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from torch.utils.data import Dataset, Sampler
import torch
import nrrd
import os
from itertools import groupby
from config import cfg
import numpy as np
import pickle 
from Script.scripts import augment_voxel_tensor,sample_z

class GANDataGenerator():
    def __init__(self):
        for i in ['train','test','val']:
            with open(getattr(cfg,"GAN_{}_SPLIT".format(i.upper())), 'rb') as pickle_file:
                tmp=pickle.load(pickle_file)

                setattr(self, "{}_data".format(i), tmp)

        self.generate_fake_match_batch('train')
        self.get_real_match_batch('train')
        self.get_real_mismatch_batch('train')

    def get_real_mismatch_batch(self, phase):
        raw_embedding_list=[]

        learned_embedding_list = []
        label_list = []
        model_list = []

        for ind in getattr(self,"{}_data".format(phase)):

            caption_data = ind
            print(ind)
            curr_label=caption_data[1]
            curr_model_id=caption_data[0]


            while True:
                db_ind_mismatch = np.random.randint(self.num_data)
                caption_data =self.embeddings[db_ind_mismatch]
                cur_raw_embedding = caption_data[3]
                cur_learned_embedding = caption_data[2]
                cur_label_mismatch = caption_data[1]
                cur_model_id_mismatch = caption_data[0]


                if cur_model_id_mismatch == curr_model_id:  
                    continue  
                break

            raw_embedding_list.append(cur_raw_embedding)
            learned_embedding_list.append(cur_learned_embedding)
            model_list.append(curr_model_id)
            label_list.append(curr_label)



        
        setattr(self, "{}_real_mis".format(phase), (
            model_list,
            label_list,
            learned_embedding_list,
            raw_embedding_list
        ))
    
    def generate_fake_match_batch(self,phase):
        learned_embedding_list = []
        raw_embedding_list=[]
        label_list = []
        model_list = []
        self.li=[]
        getattr(self,"{}_data".format(phase))
        for ind in getattr(self,"{}_data".format(phase)):
            caption_data = ind

            model_list.append(caption_data[0])
            label_list.append(caption_data[1])
            learned_embedding_list.append(caption_data[2])
            raw_embedding_list.append(caption_data[3])

        setattr(self, "{}_fake_mis".format(phase), (
            model_list,
            label_list,
            learned_embedding_list,
            raw_embedding_list
        ))

    def get_real_match_batch(self,phase):
        raw_embedding_list=[]

        learned_embedding_list = []
        label_list = []
        model_list = []

        for ind in getattr(self,"{}_data".format(phase)):
            caption_data = ind
            model_list.append(caption_data[0])
            label_list.append(caption_data[1])
            learned_embedding_list.append(caption_data[2])
            raw_embedding_list.append(caption_data[3])



        setattr(self, "{}_real_match".format(phase), (
            model_list,
            label_list,
            learned_embedding_list,
            raw_embedding_list
        ))
        

class GANLoader(Dataset):
    def __init__(self,embeddings,phase,noise):
        self.embeddings=embeddings
        self.solid_file=cfg.GAN_VOXEL_FOLDER
        self.phase=phase
        self.noise=noise

    def __len__(self):
        return len(self.embeddings)

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
        
        if self.noise:
            learned_embedding=torch.cat((learned_embedding,sample_z()),1)


        return model_id, label, learned_embedding, raw_caption , voxel
    

def collate_embedding(data):
    model_ids, labels, learned_embeddings, raw_captions , voxels = zip(*data)


    voxels = torch.stack(voxels, 0)
    return model_ids,  torch.Tensor(labels), learned_embeddings, raw_captions,voxels



def check_dataset(dataset, batch_size):
    flag = False
    if len(dataset) % batch_size != 0:
        flag = True
    
    return flag


    
    

        
