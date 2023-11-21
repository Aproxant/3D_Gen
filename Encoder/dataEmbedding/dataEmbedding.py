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
from collections import Counter


class Read_Load_BuildBatch():
    def __init__(self, batch_size):
        self.batch_size=batch_size
        self.label_enc={'03001627':0,'04379243':1}

        self.stanfordDataReader()
        self.build_word_dict()
        self.tokenize_captions()
        #self.aggregate_samples()
        #self.buildBatch('train')

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

        self.train_num_data=[]
        self.test_num_data=[]
        self.val_num_data=[]

        self.caption_tuples_train=[]
        self.caption_tuples_test=[]
        self.caption_tuples_val=[]

        self.caption_matches_train=[]
        self.caption_matches_test=[]
        self.caption_matches_val=[]

        with open(cfg.EMBEDDING_SHAPENET, 'rb') as pickle_file:
            shape=json.load(pickle_file)
            self.idx_to_word=shape['idx_to_word']
            self.word_to_idx=shape['word_to_idx']

        for phase in ['train','test','val']:
            with open(getattr(cfg,"EMBEDDING_{}_SPLIT".format(phase.upper())), 'rb') as pickle_file:
                models=pickle.load(pickle_file)
                model_to_idx={i:idx for idx,i in enumerate(models['caption_matches'].keys())}
                idx_to_model={idx:i for idx,i in enumerate(models['caption_matches'].keys())}

                setattr(self, "{}_model_to_idx".format(phase), model_to_idx)
                setattr(self, "{}_idx_to_model".format(phase), idx_to_model)
                                
                setattr(self, "{}_num_data".format(phase), len(idx_to_model))
                setattr(self, "caption_tuples_{}".format(phase), models['caption_tuples'])
                setattr(self, "caption_matches_{}".format(phase), models['caption_matches'])


        with open(cfg.EMBEDDING_BAD_IDS, 'rb') as pickle_file:
            self.bad_ids=pickle.load(pickle_file)

        with open(cfg.EMBEDDING_SHAPENET, 'rb') as json_file:
            
            captions=json.load(json_file)
            for i in captions['captions']:
                #if i['model'] in self.bad_ids:
                #    self.bad_ids_captions.append(i['caption'])
                #    continue
                if i['model'] in self.train_model_to_idx.keys():
                    self.train.append((i['model'],self.label_enc[i['category']],self.train_model_to_idx[i['model']],i['caption']))
                elif i['model'] in self.test_model_to_idx.keys():
                    self.test.append((i['model'],self.label_enc[i['category']],self.test_model_to_idx[i['model']],i['caption']))
                elif i['model'] in self.val_model_to_idx.keys():
                    self.val.append((i['model'],self.label_enc[i['category']],self.val_model_to_idx[i['model']],i['caption']))




    def build_word_dict(self):
        word_count = {}
        counter=2
        self.wordlens=[]
        for item in self.train:    
            self.wordlens.append(len(item[3]))   
            for word in item[3]:
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

                main_category=item[1]

                label = item[2]

                words = item[3]

                if model_id in data_group.keys():
                    data_group[model_id].append((model_id,main_category, label, words))
                else:
                    data_group[model_id] = [(model_id,main_category, label, words)]


            setattr(self, "data_group_{}".format(phase), data_group)

    def get_matching_tuples(self, caption_tuples,caption_matches,num_data,db_ind, model_id_list, category_list):

        while True:
            caption_tuple = caption_tuples[db_ind]
            cur_model_id = caption_tuple[2]

            try:
                assert cur_model_id not in model_id_list

                matching_caption_tuple = self.load_matching_caption_tuple(caption_tuples,caption_matches,db_ind)

            except AssertionError:  # Retry if only one caption for current model
                db_ind = np.random.randint(num_data)  # Choose new caption
                continue
            break
        return caption_tuple, matching_caption_tuple
    
    def load_matching_caption_tuple(self,caption_tuples,caption_matches, db_ind):

        caption_tuple = caption_tuples[db_ind]

        model_id = caption_tuple[2]
        match_idxs = caption_matches[model_id]

        assert len(match_idxs) > 1

        # Select a caption from the matching caption list
        selected_idx = db_ind
        while selected_idx == db_ind:
            selected_idx = random.choice(match_idxs)


        assert model_id == caption_tuples[selected_idx][2]

        return caption_tuples[selected_idx]
    
    def get_next_minibatch(self,phase):
        num_data=getattr(self, "{}_num_data".format(phase))
        half_batch_size = self.batch_size // 2
        if (self.cur + half_batch_size) >= num_data:
            return None



        db_inds = self.perm[self.cur:min(self.cur + half_batch_size, num_data)]
        self.cur += half_batch_size
        return db_inds
    
    def shuffle_db_inds(self,phase):
        num_data=getattr(self, "{}_num_data".format(phase))

        self.perm = np.random.permutation(np.arange(num_data))

        self.cur = 0
    
    def verify_batch(self, data_list):
        assert len(data_list) == self.batch_size
        counter = Counter(data_list)
        for _, v in counter.items():
            assert v == 2


    def buildBatch(self,phase):
        self.shuffle_db_inds(phase)
        num_data=getattr(self, "{}_num_data".format(phase))
        caption_tuples=getattr(self, "caption_tuples_{}".format(phase))
        caption_matches=getattr(self, "caption_matches_{}".format(phase))

        self.newGenBatch=[]
        while self.cur < num_data:
            db_inds = self.get_next_minibatch(phase)
            if db_inds is None:
                return
            data_list = []  # captions
            category_list = []  # categories
            model_list = []  # models
            model_id_list = []
            main_category_list=[]
            caption=[]
            for db_ind in db_inds:
                caption_tuple, matching_caption_tuple = self.get_matching_tuples(caption_tuples,caption_matches,num_data,db_ind,
                                                                                 model_id_list,
                                                                                 category_list)
                model_id_list.append(caption_tuple[2])
                data_list.append(caption_tuple[0])  # 0th element is the caption
                data_list.append(matching_caption_tuple[0])  # 0th element is the caption
                category_list.append(self.label_enc[caption_tuple[1]])
                category_list.append(self.label_enc[matching_caption_tuple[1]])
                model_list.append(caption_tuple[2])
                model_list.append(matching_caption_tuple[2])
                main_category_list.extend([db_ind,db_ind])
                indices=[]
                for word in caption_tuple[0]:
                    if str(word) in self.idx_to_word.keys():
                        indices.append(self.idx_to_word[str(word)])
                    else:
                        indices.append("<PAD>")

                caption.append(indices)
                indices=[]

                for word in matching_caption_tuple[0]:
                    if str(word) in self.idx_to_word.keys():
                        indices.append(self.idx_to_word[str(word)])
                    else:
                        indices.append("<PAD>")
                caption.append(indices)

            
            self.verify_batch(model_list)
            
            self.newGenBatch.extend(zip(model_list,category_list,main_category_list,caption))

    def returnNewEpoch(self,phase):
        self.buildBatch(phase)
        return self.newGenBatch
    def aggregate_samples(self,phase):
        
            # get all combinations
            
        data_comb=[]

        data_group=getattr(self, "data_group_{}".format(phase))
        for key in data_group.keys():
            if len(data_group[key]) >=2:
                comb = list(combinations(data_group[key], 2))
                random.shuffle(comb)
                data_comb.extend(comb)

        data = []
        idx2label = {i: data_comb[i][0][0] for i in range(len(data_comb))}
        chosen_label = []
        while len(data) < 2 * len(data_comb):
            if len(chosen_label) == self.batch_size//2:
                chosen_label = []
            idx = np.random.randint(len(data_comb))
            if idx2label[idx] in chosen_label:
                continue
            else:
                data.extend([data_comb[idx][i] for i in range(2)])
                chosen_label.append(idx2label[idx])

        """
            # aggregate batch    
        data = []
        idx2label = {i: data_comb[i][0][0] for i in range(len(data_comb))}
        unseenLabels={i: 1 for i in range(len(data_comb))}
        chosen_label = []
        while len(unseenLabels.keys())>0:

            while len(data) < 2 * len(data_comb):
                if len(chosen_label) == self.batch_size:
                    chosen_label = []
                idx = np.random.randint(len(data_comb))
                if idx2label[idx] in chosen_label:
                    continue
                else:
                    data.extend([data_comb[idx][i] for i in range(2)])
                    chosen_label.append(idx2label[idx])
        """
        self.newGenBatch=data

    def returnNewEpoch2(self,phase):
        self.aggregate_samples(phase)
        return self.newGenBatch
    


            
            