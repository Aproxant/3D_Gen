from tqdm import tqdm
import torch
import pickle
import os

def build_embeedings_CWGAN(text_encoder_file,model,loader,save_path,device):

    model.load_state_dict(torch.load(text_encoder_file))
    
    model.eval()

    for phase in ['train','test','val']:



        data=[]
        for (model_id,_,labels,texts) in tqdm(loader[phase]):

            texts = texts.to(device)

            text_embedding = model(texts)
            for i,elem in enumerate(model_id):         
                data.append((elem,labels[i].detach(),text_embedding[i].detach(),texts[i].detach()))

        with open(os.path.join(save_path,phase+'.p'), 'wb') as file:
            pickle.dump(data, file)


