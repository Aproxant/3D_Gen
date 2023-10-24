
from torch.optim.lr_scheduler import StepLR
from torch.nn.utils import clip_grad_value_
from tqdm import tqdm
import torch
import numpy as np
from config import cfg
from Solvers.Evaluation import compute_metrics
import os
#from tqdm.notebook import tqdm as tqdm_nb
from torch.utils.data import DataLoader
from dataEmbedding.dataEmbeddingLoader import GenerateDataLoader,check_dataset,collate_embedding


class Solver():
    def __init__(self, text_encoder,stanData, optimizer, criterion, batch_size,mode,device):
        self.text_encoder=text_encoder
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.criterion = criterion
        self.device=device
        self.stanData=stanData
        self.eval_acc = np.asarray([0.] * 5)
        self.eval_ckpts = [None] * 5
        self.mode=mode

    def dynamicBatchConstruction(self):
        if False:
            epochData=self.stanData.returnNewEpoch('train')
            epochDataVAL=self.stanData.returnNewEpoch('val')

            epochDataTEST=self.stanData.returnNewEpoch('test')
        else:
            epochData=self.stanData.returnNewEpoch2('train')
            epochDataVAL=self.stanData.returnNewEpoch2('val')

            epochDataTEST=self.stanData.returnNewEpoch2('test')
        train_dataset = GenerateDataLoader(epochData,self.stanData.dict_word2idx)

        val_dataset = GenerateDataLoader(epochDataVAL,self.stanData.dict_word2idx)

        test_dataset=GenerateDataLoader(epochDataTEST,self.stanData.dict_word2idx)

        self.dataloader = {
            'train': DataLoader(
            train_dataset, 
            batch_size=cfg.EMBEDDING_BATCH_SIZE,              
            drop_last=check_dataset(train_dataset, cfg.EMBEDDING_BATCH_SIZE),
            collate_fn=collate_embedding,
            num_workers=0
            ),
            'val': DataLoader(
            val_dataset, 
            batch_size=cfg.EMBEDDING_BATCH_SIZE,
            collate_fn=collate_embedding,
            num_workers=0
            ),
            'test': DataLoader(
            test_dataset, 
            batch_size=cfg.EMBEDDING_BATCH_SIZE,
            collate_fn=collate_embedding,
            #num_workers=2
            )
            }    
        
    def train(self, epoch,idx_word):
        scheduler = StepLR(self.optimizer, step_size=cfg.EMBEDDING_SCHEDULER_STEP, gamma=cfg.EMBEDDING_SCHEDULER_GAMMA)
        self.dynamicBatchConstruction()
        for epoch_id in range(epoch):
            print("Epoch [{}/{}] starting...\n".format(epoch_id+1, epoch))

            print("Training...")
            #pbar = tqdm_nb()
            #pbar.reset(total=len(self.dataloader['train']))

                
            train_log = {
                    'total_loss': [],
                    'metric_loss': [],
                    'text_norm_penalty': [],
                    'separator_loss': []

                    }
            val_log = {
                    'total_loss': [],
                    'metric_loss': [],
                    'text_norm_penalty': [],
                    'separator_loss': []
                    }
            
            
            self.text_encoder.train()
            if self.mode=='online':
                self.dynamicBatchConstruction()
            
            epochLoss=[]
            for i,(_,main_cat,labels,texts) in tqdm(enumerate(self.dataloader['train']),total=len(self.dataloader['train'])):
                #pbar.update()
                losses = self.forward(texts, labels,main_cat)

                train_log['total_loss'].append(losses['total_loss'].item())                   
                train_log['metric_loss'].append(losses['metric_loss'].item())
                train_log['separator_loss'].append(losses['separator_loss'].item())


                train_log['text_norm_penalty'].append(losses['text_norm_penalty'].item())

                epochLoss.append(losses['total_loss'].item())
                # back prop
                self.optimizer.zero_grad()

                losses['total_loss'].backward()

                clip_grad_value_(self.text_encoder.parameters(), cfg.EMBEDDING_GRADIENT_CLIPPING)

                self.optimizer.step()


                desc = 'Training: [%d/%d][%d/%d], Total loss: %.4f' \
                    % (epoch_id+1, epoch, i+1, len(self.dataloader['train']), losses['total_loss'].item())
                #pbar.set_description(desc)
                
            print('Total epoch loss: %.4f' % np.mean(epochLoss))
            #pbar.close()
            # validate
            if epoch_id % 5==0 and epoch_id!=0:
                val_log = self.validate(val_log)
            

                self._epoch_report(train_log, val_log, epoch_id, epoch)
            
            
                # evaluate
                metrics_t2t = self.evaluate(self.text_encoder, idx_word)
                # Check if we should terminate training
                cur_eval_acc = metrics_t2t.precision[4]  # Precision @ 5
                if all(self.eval_acc > cur_eval_acc):
                    #terminate training
                    print('Best checkpoint:', self.val_ckpts[np.argmax(self.eval_acc)])
                    return 
                else:  # Update val acc list
                    if max(self.eval_acc)<cur_eval_acc:
                        print("saving models...\n")

                        torch.save(self.text_encoder.state_dict(), os.path.join(cfg.EMBEDDING_TEXT_MODELS_PATH,"text_encoder.pth"))

                    self.eval_acc = np.roll(self.eval_acc, 1)
                    self.eval_acc[0] = cur_eval_acc
                    self.eval_ckpts = np.roll(self.eval_ckpts, 1)
                    self.eval_ckpts[0] = epoch_id + 1


            scheduler.step()

    def forward(self,texts, labels,main_cat):

        texts = texts.to(self.device)
        text_labels = labels.to(self.device)
        main_cat=main_cat.to(self.device)

        t = self.text_encoder(texts)

        
        losses = self.compute_loss(t, text_labels,main_cat)

        return losses


    def compute_loss(self, t, t_labels,main_cat):

        
        n_pair = self.criterion['metric_main'](t,t_labels)
        
        text_norm_penalty = self._norm_penalty(t)
        tripletBatch=self.construct_tripletLoss(t,main_cat) #to dla dzielenia przy uyciu triple

        metric_loss_triplet=self.criterion['metric_separator'](tripletBatch,main_cat)


        loss =  cfg.METRIC_WEIGHT*n_pair
        loss += cfg.TEXT_NORM_MULTIPLIER * text_norm_penalty
        loss+=cfg.TRIPLET_MULTIPLIER*metric_loss_triplet

        losses = {
                'total_loss': loss,
                'metric_loss': n_pair*cfg.METRIC_WEIGHT,
                'separator_loss': cfg.TRIPLET_MULTIPLIER*metric_loss_triplet,
                'text_norm_penalty': text_norm_penalty*cfg.TEXT_NORM_MULTIPLIER
                }
                

        return losses
    
    def construct_tripletLoss(self,t,main_cat):
        unique_elements = torch.unique(main_cat)

        indices = [torch.where(main_cat == element)[0] for element in unique_elements]
        min_len=min(len(indices[0]), len(indices[1]))

        cat_indices=[]
        min_len=(min_len//2)*2
        for i in range(0,min_len,2):
            one_a=indices[0][i]
            one_b=indices[0][i+1]

            two_a=indices[1][i]
            two_b=indices[1][i+1]
            cat_indices.extend([t[one_a],t[one_b],t[two_a],t[two_b]])

        return torch.stack(cat_indices)


    def _norm_penalty(self,embedding):
        norm = torch.norm(embedding, p=2, dim=1)
        penalty = torch.max(torch.zeros(norm.size()).to(self.device), norm - cfg.MAX_NORM).mean()

        return penalty

    def validate(self, val_log):
        print("Validating...\n")

        self.text_encoder.eval()

        #pbar = tqdm_nb()
        #pbar.reset(total=len(self.dataloader['val']))

        for i,(_,main_cat,labels,texts) in tqdm(enumerate(self.dataloader['val']),total=len(self.dataloader['val'])):
            #pbar.update()

            with torch.no_grad():
                losses = self.forward(texts, labels,main_cat)

            # record

            val_log['total_loss'].append(losses['total_loss'].item())
            val_log['metric_loss'].append(losses['metric_loss'].item())
            val_log['separator_loss'].append(losses['separator_loss'].item())
            val_log['text_norm_penalty'].append(losses['text_norm_penalty'].item())

            desc = 'Validating: [%d/%d], Total loss: %.4f' \
                    % (i+1, len(self.dataloader['val']), losses['total_loss'].item())
            #pbar.set_description(desc)

        #pbar.close()

        return val_log
    
    def evaluate(self,text_encoder,idx_word):

        text_encoder.eval()

        print("Evaluating...")
        embedding = self.build_embeedings_for_eval(idx_word,'test')

        metrics = compute_metrics(embedding)
        return metrics

    def build_embeedings_for_eval(self,idx_word,phase):
        data = {}
        #pbar = tqdm_nb()
        #pbar.reset(total=len(self.dataloader[phase]))
        self.text_encoder=self.text_encoder.to('cpu')
        for j,(model_id,_,_,texts) in tqdm(enumerate(self.dataloader[phase])):
            #pbar.update()

            texts = texts.to('cpu')
           
            text_embedding = self.text_encoder(texts)
            for i,elem in enumerate(model_id):    

                caption=" ".join([idx_word[item.item()] for item in texts[i] if item.item()!=0])   
                if elem in data.keys():
                    data[elem]['text_embedding'].append((caption,text_embedding[i]))
                else:
                    data[elem] = {
                        'text_embedding': [(caption,text_embedding[i])]
                    }
            desc = 'Evaluating: [%d/%d]' \
                    % (j+1, len(self.dataloader[phase]))
            #pbar.set_description(desc)
        self.text_encoder=self.text_encoder.to(cfg.DEVICE)
        #pbar.close()
        return data


    def _epoch_report(self,train_log, val_log, epoch_id, epoch):
        # show report
        
        print("epoch [{}/{}] done...".format(epoch_id+1, epoch))
        print("------------------------summary------------------------")
        print("[train] total_loss: %f \n train_min_loss: %f \n train_max_loss: %f" % (
                np.mean(train_log['total_loss']),np.min(train_log['total_loss']),np.max(train_log['total_loss'])
        ))
        print("[val]   total_loss: %f \n val_min_loss: %f \n train_max_loss: %f" % (
            np.mean(val_log['total_loss']),np.min(val_log['total_loss']),np.max(train_log['total_loss'])
        ))
        print("[train] metric_loss: %f" % (
            np.mean(train_log['metric_loss'])
        ))
        print("[train] separator_loss: %f" % (
            np.mean(train_log['separator_loss'])
        ))
        print("[val]  metric_loss: %f" % (
            np.mean(val_log['metric_loss'])
        ))
        print("[val] separator_loss: %f" % (
            np.mean(val_log['separator_loss'])
        ))