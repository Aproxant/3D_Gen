from tqdm import tqdm
import torch
import pickle
import os
from torch.utils.data import DataLoader

from dataEmbedding.dataEmbeddingLoader import GenerateDataLoader,check_dataset,collate_embedding
from config import cfg
import random

def build_embeedings_CWGAN(text_encoder_file,model,data_dict,vocab_dict,save_path,phase,type):

    model.load_state_dict(torch.load(text_encoder_file,map_location=torch.device('cpu')))
    model=model.to(cfg.DEVICE)
    model.eval()

    data=[]
    for i in data_dict.keys():
        for elem in data_dict[i]:
            data.append((elem[0],elem[1],elem[2],elem[3]))

    random.shuffle(data)
    new_dataset=GenerateDataLoader(data,vocab_dict,phase)
    loader = DataLoader(new_dataset, batch_size=cfg.EMBEDDING_BATCH_SIZE,collate_fn=collate_embedding,shuffle=False)

    GanData=[]
    
    for (model_id,main_cat,labels,texts) in tqdm(loader):
        texts = texts.to(cfg.DEVICE)

        text_embedding = model(texts)
        for i,elem in enumerate(model_id):  
            if type=='info':
                GanData.append((elem,main_cat[i].detach(),labels[i].detach(),text_embedding[i].detach(),texts[i].detach()))
            else:
                GanData.append((elem,text_embedding[i].detach().cpu().numpy()))
            
    with open(os.path.join(save_path,'{}.p'.format(phase)), 'wb') as file:
        pickle.dump(GanData, file)


def singleRun(data,model,vocab_dict):
    new_dataset=GenerateDataLoader([data],vocab_dict)
    loader = DataLoader(new_dataset, batch_size=1,collate_fn=collate_embedding)
    for (_,_,_,texts) in tqdm(loader):
        texts = texts.to(cfg.DEVICE)
        text_embedding = model(texts)
    return text_embedding

