from dataEmbedding.dataEmbeddingLoader import GenerateDataLoader,check_dataset,collate_embedding
from tqdm import tqdm
from Models.EncoderModels import TextEncoder
from dataEmbedding.dataEmbedding import Read_Load_BuildBatch
from torch.utils.data import DataLoader
import torch
import nrrd
from config import cfg
import os

def build_embeedings_CWGAN(text_encoder_file,dataloader,device):

    model=TextEncoder()
    model.load_state_dict(torch.load(text_encoder_file))
    model.eval()

    for phase in ['train','test','val']:
        data=[]
        for (model_id,labels,_,texts , _, ) in tqdm(dataloader[phase]):

            texts = texts.to(device)

            text_embedding = model(texts)

            for i,elem in enumerate(model_id):         
                data.append((elem,labels[i],text_embedding[i]))

        nrrd.write(os.path.join(cfg.EMBEDDING_SAVE_PATH,phase+'.nrrd'),data)



if __name__ == '__main__':
    stanData=Read_Load_BuildBatch(cfg.EMBEDDING_VOXEL_FOLDER,cfg.EMBEDDING_BATCH_SIZE)
    train_dataset = GenerateDataLoader(stanData.data_agg_train,stanData.data_dir,stanData.dict_word2idx,'train')

    val_dataset = GenerateDataLoader(stanData.data_agg_val,stanData.data_dir,stanData.dict_word2idx,'val')

    test_dataset=GenerateDataLoader(stanData.data_agg_test,stanData.data_dir,stanData.dict_word2idx,'test')

    dataloader = {
            'train': DataLoader(
                train_dataset, 
                batch_size=cfg.EMBEDDING_BATCH_SIZE * 2,              
                drop_last=check_dataset(train_dataset, cfg.EMBEDDING_BATCH_SIZE * 2),
                collate_fn=collate_embedding,
                num_workers=4
            ),
            'val': DataLoader(
                val_dataset, 
                batch_size=cfg.EMBEDDING_BATCH_SIZE*2,
                collate_fn=collate_embedding,
                num_workers=4
            ),
            'test': DataLoader(
                test_dataset, 
                batch_size=cfg.EMBEDDING_BATCH_SIZE*2,
                collate_fn=collate_embedding
                #num_workers=2
            )
    }

    build_embeedings_CWGAN(cfg.EMBEDDING_MODELS_PATH,dataloader,cfg.DEVICE)