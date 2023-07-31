import pandas as pd
import os
import random 
from itertools import combinations
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
from config import cfg
import json
import spacy
import numpy as np

class Read_Load_BuildBatch():
    def __init__(self, data_dir,batch_size):
        self.data_dir=data_dir
        self.stanfordDataReader()
        self.build_word_dict()
        self.tokenize_captions()
        self.aggregate_samples(batch_size)



    def stanfordDataReader(self):
        self.train=[]
        self.test=[]
        self.val=[]
        self.train_model_to_idx=[]
        self.train_idx_to_model=[]
        self.test_model_to_idx=[]
        self.test_idx_to_model=[]
        self.val_model_to_idx=[]
        self.val_idx_to_model=[]

        self.bad_ids_captions=[]

        for phase in ['train','test','val']:
            with open(getattr(cfg,"EMBEDDING_{}_SPLIT".format(phase.upper())), 'rb') as pickle_file:
                models=pickle.load(pickle_file)
                model_to_idx={i:idx for idx,i in enumerate(models['caption_matches'].keys())}
                idx_to_model={idx:i for idx,i in enumerate(models['caption_matches'].keys())}

                setattr(self, "{}_model_to_idx".format(phase), model_to_idx)
                setattr(self, "{}_idx_to_model".format(phase), idx_to_model)

        with open(cfg.EMBEDDING_BAD_IDS, 'rb') as pickle_file:
            self.bad_ids=pickle.load(pickle_file)

        with open(cfg.EMBEDDING_SHAPENET, 'rb') as json_file:
            
            captions=json.load(json_file)
            for i in captions['captions']:
                if i['model'] in self.bad_ids:
                    self.bad_ids_captions.append(i['caption'])
                    continue

                if i['model'] in self.train_model_to_idx.keys():
                    self.train.append((i['model'],self.train_model_to_idx[i['model']],i['caption']))
                elif i['model'] in self.test_model_to_idx.keys():
                    self.test.append((i['model'],self.test_model_to_idx[i['model']],i['caption']))
                elif i['model'] in self.val_model_to_idx.keys():
                    self.val.append((i['model'],self.val_model_to_idx[i['model']],i['caption']))


    def build_word_dict(self):
        word_count = {}
        counter=2
        self.wordlens=[]
        for item in self.train:    
            self.wordlens.append(len(item[2]))     
            for word in item[2]:
                if word  not in word_count.keys():
                    word_count[word] = counter
                    counter+=1          

        for item in self.bad_ids_captions:
            self.wordlens.append(len(item))     

            for word in item:
                if word  not in word_count.keys():
                    word_count[word] = counter
                    counter+=1  

        word_count["<UNK>"]=1
        word_count["<PAD>"]=0
        self.dict_word2idx=word_count
        self.dict_idx2word={item: key for key,item in word_count.items()}


    def tokenize_captions(self):
        self.data_group_train={}
        self.data_group_test={}
        self.data_group_val={}

        for phase in ['train','test','val']:
            data_group = {}
            elements=getattr(self, "{}".format(phase))
            for item in elements:

                model_id = item[0]

                label = item[1]

                words = item[2]

                if model_id in data_group.keys():
                    data_group[model_id].append((model_id, label, words))
                else:
                    data_group[model_id] = [(model_id, label, words)]


            setattr(self, "data_group_{}".format(phase), data_group)

    def aggregate_samples(self,batch_size):
        
        for phase in ['train','test','val']:
            # get all combinations
            """
            data_comb=[]

            data_group=getattr(self, "data_group_{}".format(phase))
            for key in data_group.keys():
                if len(data_group[key]) >=4:
                    comb = list(combinations(data_group[key], 2))
                    random.shuffle(comb)
                    data_comb.extend(comb)


            # aggregate batch    
            data = []
            idx2label = {i: data_comb[i][0][0] for i in range(len(data_comb))}
            chosen_label = []
            
            while len(data) < 2 * len(data_comb):
                if len(chosen_label) == batch_size:
                    chosen_label = []
                idx = np.random.randint(len(data_comb))
                if idx2label[idx] in chosen_label:
                    continue
                else:
                    data.extend([data_comb[idx][i] for i in range(2)])
                    chosen_label.append(idx2label[idx])
                
            setattr(self, "data_agg_{}".format(phase), data)
            """

            data_comb={}
            available_data={}
            data_group=getattr(self, "data_group_{}".format(phase))
            for key in data_group.keys():
                if len(data_group[key]) >=2:
                    comb = list(combinations(data_group[key], 2))
                    random.shuffle(comb)
                    data_comb[key]=comb
                    available_data[key]=len(comb)

                # aggregate batch    
            data = []
            
            while sum(available_data.values())>=batch_size: 
                available_keys = [k for k, v in available_data.items() if v > 0]

                if len(available_keys)<batch_size:
                    break

                # dodaj zapisane do finalnego
                chosen_model_ids=random.sample(available_keys, k=batch_size)

                # odśwież slwonik
                available_data={key : (val-1 if key in chosen_model_ids else val) for key,val in available_data.items()}

                random.shuffle(chosen_model_ids)
                for j in chosen_model_ids:                
                    data.extend([data_comb[j][available_data[j]][i] for i in range(2)])


            setattr(self, "data_agg_{}".format(phase), data)
            
            