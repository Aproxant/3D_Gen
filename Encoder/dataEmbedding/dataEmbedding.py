import pandas as pd
import os
import random 
from itertools import combinations
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
from config import cfg
import numpy as np
import json

class Read_Load_BuildBatch():
    def __init__(self, data_dir,batch_size,datasetName='primitives',caption_file=None,):
        self.data_dir=data_dir
        self.labelEnc={}
        self.counter=0
        self.batch_size=batch_size

        if datasetName=='primitives':           
            self.primitives_DataReader(self.data_dir)
            #self.build_word_dict()
            self.tokenize_captions('primitives')
            self.split_data_primi(0.1,0.1)
            self.aggregate_samples('primitives',batch_size)
        elif datasetName=='stanford_data':
            self.stanford_DataReader()
            #self.build_word_dict()
            #self.tokenize_captions('stanford_data')
            self.aggregate_samples_new('stanford_data',batch_size)
            #self.aggregate_samples('stanford_data',batch_size)

    def generateLabelEncoding(self,label):
        if label not in self.labelEnc:
            self.labelEnc[label]=self.counter
            self.counter+=1


    def stanford_DataReader(self):

        with open(cfg.EMBEDDING_TRAIN_FILE, 'rb') as train_file:
            self.train_split=pickle.load(train_file)

        with open(cfg.EMBEDDING_TEST_FILE, 'rb') as test_file:
            self.test_split=pickle.load(test_file)

        with open(cfg.EMBEDDING_VAL_FILE, 'rb') as val_file:
            self.val_split=pickle.load(val_file)
        
        with open(cfg.EMEDDING_PROBLEMATIC_MODELS_FILE,'rb') as problematic_models:
            self.problematic=pickle.load(problematic_models)

        with open(cfg.EMBEDDING_CAPTIONS_FILE, 'rb') as caption_file:
            self.captions=json.load(caption_file)

        self.caption_tuples = self.train_split['caption_tuples']
        self.caption_matches=self.train_split['caption_matches']

        self.matches_keys = list(self.caption_matches.keys())
        self.n_captions_per_model = cfg.EMBEDDING_N_CAPTIONS_PER_MODEL


        self.n_unique_shape_categories = self.batch_size
        self.n_models_per_batch = self.n_unique_shape_categories


        self.max_sentence_length = len(self.caption_tuples[0][0])

        lengths = []
        for cur_tup in self.caption_matches.values():
            lengths.append(len(cur_tup))

        self.num_data=len(range(len(self.caption_matches)))
        
    def primitives_DataReader(self,path):
        path=os.path.join(path,'primitives.v2')
        self.annotations=[]
        for i in os.listdir(path):
            if os.path.isdir(os.path.join(path,i)):
                model_id=[]             
            for j in os.listdir(os.path.join(path,i)):
                if j.endswith('.nrrd'):   

                    model_id.append(j.split('.')[0])

            with open(os.path.join(path,i,'descriptions.txt'), "r") as f:
                #to do zmiany bo jest ograniczenie na 10 cations można 12 
                captions=f.read().splitlines()
                for m in range(len(model_id)):
                    cap=random.sample(captions,k=10)
                    for c in cap:
                        self.annotations.append((model_id[m].split('_')[1],i,c))

    def split_data_primi(self,test_size,val_size):
        modelId=self.data_group.keys()
        items_nr=len(modelId)
        test_it=random.choices(list(modelId), k=int(items_nr*test_size))
        train={}
        test=[]
        for ele in modelId:
            if ele not in test_it:
                train[ele]=self.data_group[ele]
            else:
                test.append(random.choice(self.data_group[ele]))
                

        rem = list(filter(lambda i: i not in test_it, modelId))

        val_it=random.choices(rem, k=int(len(train)*val_size))
        train2={}
        val=[]
        for ele in train:
            if ele not in val_it:
                train2[ele]=self.data_group[ele]
            else:
                val.append(random.choice(self.data_group[ele]))


        self.data_group=train2
        self.test=test
        self.val=val



    def split_data(self,modelId,test_size,val_size):

        items_nr=len(modelId)

        test_count=int(items_nr*test_size)

        test_val=random.sample(list(modelId), int(items_nr*(val_size+test_size)))


        #test_it=random.choices(modelId, k=int(items_nr*test_size))
        #train=[]
        #test=[]

        """
        for ele in modelId:
            if ele not in test_it:
                train.append(ele)
            else:
                test.append(ele)
        """
        train = list(filter(lambda i: i not in test_val, modelId))

        test=random.sample(test_val, test_count)
        
        val=list(filter(lambda i: i not in test, test_val))

        """
        train2=[]
        val=[]
        for ele in train:
            if ele not in val_it:
                train2.append(ele)
            else:
                val.append(ele)

        """
        return train,test,val



    def build_word_dict(self):
        split_data = self.train+self.val+self.test
        word_count = {}
        sw_nltk = stopwords.words('english')
        
        lemmatizer = WordNetLemmatizer()
        for item in split_data:
            words = [lemmatizer.lemmatize(word) for word in item[2].split() if word.lower() not in sw_nltk]
            for word in words:
                if word in word_count.keys():
                    word_count[word] += 1
                else:
                    word_count[word] = 1            

        word_count = sorted(word_count.items(), key=lambda x: x[1], reverse=True)
        self.dict_word2idx = {word_count[i][0]: str(i + 2) for i in range(len(word_count))}
        self.dict_idx2word = {str(i + 2): word_count[i][0] for i in range(len(word_count))}

        self.dict_word2idx["<PAD>"] = str(0)
        self.dict_idx2word[str(0)] = "<PAD>"
        self.dict_word2idx["<UNK>"] = str(1)
        self.dict_idx2word[str(1)] = "<UNK>"

    def tokenize_captions(self,dataset_name):
              
        sw_nltk = stopwords.words('english')
        
        lemmatizer = WordNetLemmatizer() 

        for phase in ['train','test','val']:
            data_group = {}
            elements=getattr(self, "{}".format(phase))
            for item in elements:

                model_id = item[0]

                label = item[1]

                words = item[2]

                indices=words
                if phase=='train':
                    words = [lemmatizer.lemmatize(word) for word in words.split() if word.lower() not in sw_nltk]
                    indices = " ".join(words)

                """
                indices = []
                for word in words.split():
                    if word in self.dict_word2idx.keys():
                        indices.append(int(self.dict_word2idx[word]))
                    else:
                        indices.append(int(self.dict_word2idx["<UNK>"]))
                """
                


                if dataset_name == 'stanford_data':
                    if model_id in data_group.keys():
                        data_group[model_id].append((model_id, label, indices,self.labelEnc[label]))
                    else:
                        data_group[model_id] = [(model_id, label, indices,self.labelEnc[label])]

                elif dataset_name == 'primitives':
                    if label in data_group.keys():
                        data_group[label].append((model_id, label, indices,self.labelEnc[label]))
                    else:
                        data_group[label] = [(model_id, label,indices,self.labelEnc[label])]

                      

            setattr(self, "data_group_{}".format(phase), data_group)

    def get_next_minibatch(self):
        if (self.cur + self.batch_size) >= self.num_data:
            self.shuffle_db_inds()

        db_inds = self.perm[self.cur:min(self.cur + self.batch_size, self.num_data)]
        self.cur += self.batch_size
        return db_inds
    
    def shuffle_db_inds(self):
        self.perm = np.random.permutation(np.arange(self.num_data))

        self.cur = 0

    def verify_batch(self, caption_tuples):
        """Simply verify that all caption tuples correspond to the same category and model ID.
        """
        category = caption_tuples[0][1]
        model_id = caption_tuples[0][2]
        for tup in caption_tuples:
            assert tup[1] == category
            assert tup[2] == model_id
        return category, model_id
    
    def is_bad_model_id(self, model_id):
        """Code reuse.
        """
        if self.problematic is not None:
            return model_id in self.problematic
        else:
            return False
        
    def aggregate_samples_new(self,dataset_name,batch_size):
        self.shuffle_db_inds()
        self.final=[]
        for phase in ['train','test','val']:
            while self.cur < self.num_data:
                db_inds = self.get_next_minibatch()

                shapes_list = []
                captions_list = []
                category_list = []
                model_id_list = []
                for db_ind in db_inds:  # Loop through each selected shape
                    selected_shapes = []
                    while True:
                        cur_key = self.matches_keys[db_ind]
                        caption_idxs = self.caption_matches[cur_key]
                        if len(caption_idxs) < self.n_captions_per_model:
                            db_ind = np.random.randint(self.num_data)
                            continue
                        selected_caption_idxs = random.sample(caption_idxs, k=self.n_captions_per_model)
                        selected_tuples = [self.caption_tuples[idx] for idx in selected_caption_idxs]
                        cur_category, cur_model_id = self.verify_batch(selected_tuples)
                        selected_model_ids = [cur_model_id]
                        for cur_model_id in selected_model_ids:
                            if self.is_bad_model_id(cur_model_id):
                                db_ind = np.random.randint(self.num_data)  # Choose new caption
                                continue
                            try:
                                cur_shape = None #load_voxel(cur_category, cur_model_id)
                            except FileNotFoundError:
                                print('ERROR: Cannot find file with the following model ID:', cur_key)
                                db_ind = np.random.randint(self.num_data)
                                continue
                            selected_shapes.append(cur_shape)
                        break
                    selected_captions = [tup[0] for tup in selected_tuples]
                    captions_list.extend(selected_captions)
                    for selected_shape in selected_shapes:
                        shapes_list.append(selected_shape)
                        cur_categories = [cur_category for _ in selected_captions]
                        cur_model_ids = [cur_model_id for _ in selected_captions]
                        category_list.extend(cur_categories)
                        model_id_list.extend(cur_model_ids)

                    
                label_list = [x for x in range(self.n_unique_shape_categories)
                          for _ in range(self.n_captions_per_model)]

                batch_captions = np.array(captions_list).astype(np.int32)
                batch_shapes = np.array(shapes_list).astype(np.float32)
                batch_label = np.array(label_list).astype(np.int32)



                batch_data = {
                    'raw_embedding_batch': batch_captions,
                    'voxel_tensor_batch': batch_shapes,
                    'caption_label_batch': batch_label,
                    'category_list': category_list,
                    'model_list': model_id_list,
                    }
                self.final.append(batch_data)
                print(self.cur)
            break

    

    def aggregate_samples(self,dataset_name,batch_size):
        
        for phase in ['train','test','val']:

            if dataset_name == 'stanford_data':
                # get all combinations
                data_comb={}
                available_data={}
                data_group=getattr(self, "data_group_{}".format(phase))
                for key in data_group.keys():
                    if len(data_group[key]) >=4:
                        comb = list(combinations(data_group[key], 2)) # generate combination of size two for loss
                        random.shuffle(comb)
                        data_comb[key]=comb
                        available_data[key]=len(comb)

                # aggregate batch    
                data = []
                idx2label = {i: data_comb[i][0][0] for i in range(len(data_comb))}
                chosen_label = []
                while len(data) < 2 * len(data_comb):
                    if len(chosen_label) == self.batch_size:
                        chosen_label = []
                    idx = np.random.randint(len(data_comb))
                    if idx2label[idx] in chosen_label:
                        continue
                    else:
                        data.extend([data_comb[idx][i] for i in range(CONF.TRAIN.N_CAPTION_PER_MODEL)])
                        chosen_label.append(idx2label[idx])
                
                while sum(available_data.values())>=batch_size: 
                    available_keys = [k for k, v in available_data.items() if v > 0]

                    if len(available_keys)<batch_size:
                        break

                    # dodaj zapisane do finalnego
                    chosen_model_ids=random.sample(available_keys, k=batch_size)

                    # odśwież slwonik
                    available_data={key : (val-1 if key in chosen_model_ids else val) for key,val in available_data.items()}

                    for j in chosen_model_ids:                
                        data.extend([data_comb[j][available_data[j]][i] for i in range(2)])


                setattr(self, "data_agg_{}".format(phase), data)
                
            elif dataset_name == 'primitives':
            
                #słownik {label:model:{[lista podpisów]}}
                new_group_data = {}
                for label in self.data_group.keys():
                    grouped_label = {}
                    for item in self.data_group[label]:
                        if item[0] in grouped_label.keys():
                            grouped_label[item[0]].append(item)
                        else:
                            grouped_label[item[0]] = [item]
                    new_group_data[label] = grouped_label
                group_data = new_group_data

            

                #łącze kombinacje w pary dwa opisy na model w różnych kombinacjach

                # get all combinations
                for label in group_data.keys():
                    for model_id in group_data[label].keys():
                        comb = list(combinations(group_data[label][model_id], 2))
                        random.shuffle(comb)
                        group_data[label][model_id] = comb

                all_model_ids = list(group_data.keys())
            

                final=[]


                # bo 10 kształtów na kategorię
                for label_id in range(10):
                    num_pairs = int((10// 2) * (10 - 1)) #WZÓR NA LICZBE KOMBINACJI (10 * 9)/2
                    for pair_id in range(num_pairs):
                        num_rand = len(all_model_ids) //batch_size # self.batch_size ZMIENIC NA PODZIELNE PRZEZ 2 I USTAWIC NA ZMIENNA
                        if num_rand < 1:
                            num_rand = 1

                        available_models_ids=all_model_ids.copy()
                        for _ in range(num_rand):
                            items_nr=len(available_models_ids)
                            chosen_model_ids=random.sample(range(0,items_nr), k=batch_size) #ZMIENIC NA PODZIELNE PRZEZ 2 I USTAWIC NA ZMIENNA
                            new_data=[]
                            batch_data=[]
                            for idx, ele in enumerate(available_models_ids):
                                if idx not in chosen_model_ids:
                                    new_data.append(ele)
                                else:
                                    batch_data.append(ele)
                 
                            available_models_ids=new_data
                            random.shuffle(batch_data)
                            for model_id in batch_data:
                                final.extend([group_data[model_id][str(label_id)][pair_id][i] for i in range(2)])
                            
                self.data_agg=final