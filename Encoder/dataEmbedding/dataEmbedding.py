import pandas as pd
import os
import random 
from itertools import combinations
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download("stopwords")
nltk.download('wordnet')


class Read_Load_BuildBatch():
    def __init__(self, data_dir,batch_size,datasetName='primitives',caption_file=None,):
        self.data_dir=data_dir
        self.labelEnc={}
        self.counter=0
        if datasetName=='primitives':           
            self.primitives_DataReader(self.data_dir)
            #self.build_word_dict()
            self.tokenize_captions('primitives')
            self.split_data_primi(0.1,0.1)
            self.aggregate_samples('primitives',batch_size)
        elif datasetName=='stanford_data':
            self.stanford_DataReader(caption_file,self.data_dir,0.1,0.1)
            self.build_word_dict()
            self.tokenize_captions('stanford_data')
            self.aggregate_samples('stanford_data',batch_size)

    def generateLabelEncoding(self,label):
        if label not in self.labelEnc:
            self.labelEnc[label]=self.counter
            self.counter+=1


    def stanford_DataReader(self,caption_file,solid_file,test_size,val_size):
        self.train=[]
        self.test=[]
        self.val=[]
      
        caption_df=pd.read_csv(caption_file)
        caption_df=caption_df.dropna()
        modelIds=caption_df.modelId.unique()
        train,test,valid=self.split_data(modelIds,test_size,val_size)

        for index,i in caption_df.iterrows():
            self.generateLabelEncoding(i['category'])
            if i['modelId'] in train:
                self.train.append((i['modelId'],i['category'],i['description']))
            elif i['modelId'] in test:
                self.test.append((i['modelId'],i['category'],i['description']))
            elif i['modelId'] in valid:
                self.val.append((i['modelId'],i['category'],i['description']))

      

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
        test_it=random.choices(modelId, k=int(items_nr*test_size))
        train=[]
        test=[]
        for ele in modelId:
            if ele not in test_it:
                train.append(ele)
            else:
                test.append(ele)

        rem = list(filter(lambda i: i not in test_it, modelId))

        val_it=random.choices(rem, k=int(len(train)*val_size))
        train2=[]
        val=[]
        for ele in train:
            if ele not in val_it:
                train2.append(ele)
            else:
                val.append(ele)


        return train2,test,val



    def build_word_dict(self):
        split_data = self.train
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

    def aggregate_samples(self,dataset_name,batch_size):
        
        for phase in ['train','test','val']:

            if dataset_name == 'stanford_data':
                # get all combinations
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
                chosen_label = []


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