
from torch.optim.lr_scheduler import StepLR
from torch.nn.utils import clip_grad_value_
from tqdm import tqdm
import torch
import numpy as np
from config import cfg
from tqdm.notebook import tqdm as tqdm_nb



class Solver():
    def __init__(self, text_encoder,shape_encoder,dataloader, optimizer, criterion, batch_size,device):
        self.dataloader = dataloader
        self.text_encoder=text_encoder
        self.shape_encoder=shape_encoder
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.criterion = criterion
        self.device=device
    def train(self, epoch):
        scheduler = StepLR(self.optimizer, step_size=cfg.EMBEDDING_SCHEDULER_STEP, gamma=cfg.EMBEDDING_SCHEDULER_GAMMA)
        pbar = tqdm_nb()
        piter = 0
        for epoch_id in range(epoch):
            pbar.reset(total=len(self.dataloader['train']))

            train_log = {
                'total_loss': [],
                'walker_loss_tst': [],
                'walker_loss_sts': [],
                'visit_loss_ts': [],
                'visit_loss_st': [],
                'metric_loss_st': [],
                'metric_loss_tt': [],
                'shape_norm_penalty': [],
                'text_norm_penalty': [],
                }
            val_log = {
                'total_loss': [],
                'walker_loss_tst': [],
                'walker_loss_sts': [],
                'visit_loss_ts': [],
                'visit_loss_st': [],
                'metric_loss_st': [],
                'metric_loss_tt': [],
                'shape_norm_penalty': [],
                'text_norm_penalty': []
                }
            #print("epoch [{}/{}] starting...\n".format(epoch_id+1, epoch))
            
            self.shape_encoder.train()

            self.text_encoder.train()
            iter=1
            piter=0
            for (_,_,labels,texts , _, shapes) in self.dataloader['train']:
                pbar.update()
                piter += 1

                losses = self.forward(shapes, texts, labels)
                train_log['total_loss'].append(losses['total_loss'].item())
                train_log['walker_loss_tst'].append(losses['walker_loss_tst'].item())
                train_log['walker_loss_sts'].append(losses['walker_loss_sts'].item())
                train_log['visit_loss_ts'].append(losses['visit_loss_ts'].item())
                train_log['visit_loss_st'].append(losses['visit_loss_st'].item())
                train_log['metric_loss_st'].append(losses['metric_loss_st'].item())
                train_log['metric_loss_tt'].append(losses['metric_loss_tt'].item())
                train_log['shape_norm_penalty'].append(losses['shape_norm_penalty'].item())
                train_log['text_norm_penalty'].append(losses['text_norm_penalty'].item())

                # back prop
                self.optimizer.zero_grad()

                losses['total_loss'].backward()

                clip_grad_value_(list(self.shape_encoder.parameters()) + list(self.text_encoder.parameters()), cfg.GRADIENT_CLIPPING)

                self.optimizer.step()
                """
                if iter % 100==0:
                    print(sum(train_log['total_loss'][-iter:])/iter)
                """
                iter+=1

                desc = '[%d/%d][%d/%d] Total Loss: %.4f' \
                    % (epoch_id+1, epoch, piter, len(self.dataloader['train']), losses['total_loss'].item())
                pbar.set_description(desc)


        
            
            # validate
            val_log = self.validate(val_log)
            
            self._epoch_report(train_log, val_log, epoch_id, epoch)

            """
            # evaluate
            metrics_t2s, metrics_s2t = self.evaluate(shape_encoder, text_encoder, dataloader)
            total_score_t2s = metrics_t2s.recall_rate[0] + metrics_t2s.recall_rate[4] + metrics_t2s.ndcg[4]
            total_score_s2t = metrics_s2t.recall_rate[0] + metrics_s2t.recall_rate[4] + metrics_s2t.ndcg[4]
            total_score = total_score_t2s + total_score_s2t
            """

            scheduler.step()
        
        
        pbar.refresh()

    def forward(self, shapes, texts, labels):

        batch_size = shapes.size(0)
        texts = texts.to(self.device)
        text_labels = labels.to(self.device)

        shapes = shapes.to(self.device).index_select(0, torch.LongTensor([i * 2 for i in range(batch_size // 2)]).to(self.device))
        shape_labels = labels.to(self.device).index_select(0, torch.LongTensor([i * 2 for i in range(batch_size // 2)]).to(self.device))
        
        s = self.shape_encoder(shapes)
        t = self.text_encoder(texts)

        losses = self.compute_loss(s, t, shape_labels, text_labels)

        return losses

    def compute_loss(self,s, t, s_labels, t_labels):

        batch_size = t.size(0)

        equality_matrix = t_labels.reshape(-1,1).eq(t_labels).float()
        p_target = (equality_matrix / equality_matrix.sum(1))

        walker_loss_tst = self.criterion['walker'](t, s, p_target)
        visit_loss_ts = self.criterion['visit'](t, s)


        equality_matrix_s = s_labels.reshape(-1,1).eq(s_labels).float()
        p_target_s = (equality_matrix_s / equality_matrix_s.sum(1))

        walker_loss_sts = self.criterion['walker'](s, t, p_target_s)
        visit_loss_st = self.criterion['visit'](s, t)


        metric_loss_tt = self.criterion['metric'](t)

        s_mask = torch.BoolTensor([[1], [0]]).repeat(batch_size // 2, cfg.EMBEDDING_DIM).to(self.device)
        t_mask = torch.BoolTensor([[0], [1]]).repeat(batch_size // 2, cfg.EMBEDDING_DIM).to(self.device)
        selected_s = s
        selected_t = t.index_select(0, torch.LongTensor([i * 2 for i in range(batch_size // 2)]).to(self.device))
        masked_s = torch.zeros(batch_size, cfg.EMBEDDING_DIM).to(self.device).masked_scatter_(s_mask, selected_s)
        masked_t = torch.zeros(batch_size, cfg.EMBEDDING_DIM).to(self.device).masked_scatter_(t_mask, selected_t)
        embedding = masked_s + masked_t

        metric_loss_st = self.criterion['metric'](embedding)
                    
        flipped_t = t.index_select(0, torch.LongTensor([i * 2 + 1 for i in range(batch_size // 2)]).to(self.device))
        flipped_masked_t = torch.zeros(batch_size, cfg.EMBEDDING_DIM).to(self.device).masked_scatter_(t_mask, flipped_t)
        embedding = masked_s + flipped_masked_t
        metric_loss_st += self.criterion['metric'](embedding)



        shape_norm_penalty = self._norm_penalty(s)
        text_norm_penalty = self._norm_penalty(t)


        loss = cfg.WALKER_WEIGHT*(walker_loss_tst + walker_loss_sts) + cfg.VISIT_WEIGHT*(visit_loss_ts + visit_loss_st)
        loss +=  cfg.METRIC_WEIGHT*(metric_loss_st + metric_loss_tt)
        loss += (cfg.SHAPE_NORM_MULTIPLIER* shape_norm_penalty + cfg.TEXT_NORM_MULTIPLIER * text_norm_penalty)

        losses = {
            'total_loss': loss,
            'walker_loss_tst': walker_loss_tst*cfg.WALKER_WEIGHT,
            'walker_loss_sts': walker_loss_sts*cfg.WALKER_WEIGHT,
            'visit_loss_ts': visit_loss_ts*cfg.VISIT_WEIGHT,
            'visit_loss_st': visit_loss_st*cfg.VISIT_WEIGHT,
            'metric_loss_st': metric_loss_st*cfg.METRIC_WEIGHT,
            'metric_loss_tt': metric_loss_tt*cfg.METRIC_WEIGHT,
            'shape_norm_penalty': shape_norm_penalty*cfg.SHAPE_NORM_MULTIPLIER,
            'text_norm_penalty': text_norm_penalty*cfg.TEXT_NORM_MULTIPLIER
            }

        return losses

    def _norm_penalty(self,embedding):
        norm = torch.norm(embedding, p=2, dim=1)
        penalty = torch.max(torch.zeros(norm.size()).to(self.device), norm - cfg.MAX_NORM).mean()

        return penalty

    def validate(self, val_log):
        print("validating...\n")
        self.shape_encoder.eval()
        self.text_encoder.eval()
        for (_,_,labels,texts , _, shapes) in tqdm(self.dataloader['val']):
            
            with torch.no_grad():
                losses = self.forward(shapes, texts, labels)

            #print(losses['total_loss'].item())
            # record
            val_log['total_loss'].append(losses['total_loss'].item())
            val_log['walker_loss_tst'].append(losses['walker_loss_tst'].item())
            val_log['walker_loss_sts'].append(losses['walker_loss_sts'].item())
            val_log['visit_loss_ts'].append(losses['visit_loss_ts'].item())
            val_log['visit_loss_st'].append(losses['visit_loss_st'].item())
            val_log['metric_loss_st'].append(losses['metric_loss_st'].item())
            val_log['metric_loss_tt'].append(losses['metric_loss_tt'].item())
            val_log['shape_norm_penalty'].append(losses['shape_norm_penalty'].item())
            val_log['text_norm_penalty'].append(losses['text_norm_penalty'].item())



        return val_log
    """
    def evaluate(self,shape_encoder, text_encoder, dataloader):
        shape_encoder.eval()
        text_encoder.eval()
        embedding = self.build_embeedings_for_eval('test')

        print("evaluating...")
        metrics_t2s = compute_metrics("shapenet", embedding, mode='t2s', metric='cosine')
        metrics_s2t = compute_metrics("shapenet", embedding, mode='s2t', metric='cosine')

        return metrics_t2s, metrics_s2t
    """

    def build_embeedings_for_eval(self,phase):
        data = {}
        for (model_id,_,labels,texts , _, shapes) in tqdm(self.dataloader[phase]):

            shapes = shapes.to(self.device)
            texts = texts.to(self.device)

            shape_embedding = self.shape_encoder(shapes)
            text_embedding = self.text_encoder(texts)

            for i,elem in enumerate(model_id):         
                if elem in data.keys():
                    data[elem]['text_embedding'].append(text_embedding[i]) 
                else:
                    data[elem] = {
                        'shape_embedding': shape_embedding[i],
                        'text_embedding': [text_embedding[i]]
                    }
        return data

    def build_embeedings_CWGAN(self,phase):
        data = []

        for (model_id,labels,_,texts , _, shapes) in tqdm(self.dataloader[phase]):

            texts = texts.to(self.device)

            text_embedding = self.text_encoder(texts)

            for i,elem in enumerate(model_id):         
                data.append((elem,labels[i],text_embedding[i]))

        return data


    def _epoch_report(self,train_log, val_log, epoch_id, epoch):
        # show report
        print("epoch [{}/{}] done...".format(epoch_id+1, epoch))
        print("------------------------summary------------------------")
        print("[train] total_loss: %f" % (
            np.mean(train_log['total_loss'])
        ))
        print("[val]   total_loss: %f" % (
            np.mean(val_log['total_loss'])
        ))
        print("[train] walker_loss_tst: %f, walker_loss_sts: %f" % (
            np.mean(train_log['walker_loss_tst']),
            np.mean(train_log['walker_loss_sts'])
        ))
        print("[val]   walker_loss_tst: %f, walker_loss_sts: %f" % (
            np.mean(val_log['walker_loss_tst']),
            np.mean(val_log['walker_loss_sts'])
        ))
        print("[train] visit_loss_ts: %f, visit_loss_st: %f" % (
            np.mean(train_log['visit_loss_ts']),
            np.mean(train_log['visit_loss_st'])
        ))
        print("[val]   visit_loss_ts: %f, visit_loss_st: %f" % (
            np.mean(val_log['visit_loss_ts']),
            np.mean(val_log['visit_loss_st'])
        ))
        print("[train] metric_loss_st: %f, metric_loss_tt: %f" % (
            np.mean(train_log['metric_loss_st']),
            np.mean(train_log['metric_loss_tt'])
        ))
        print("[val]   metric_loss_st: %f, metric_loss_tt: %f" % (
            np.mean(val_log['metric_loss_st']),
            np.mean(val_log['metric_loss_tt'])
        ))
        print("[train] shape_norm_penalty: %f, text_norm_penalty: %f" % (
            np.mean(train_log['shape_norm_penalty']),
            np.mean(train_log['text_norm_penalty'])
        ))
        print("[val]   shape_norm_penalty: %f, text_norm_penalty: %f\n" % (
            np.mean(val_log['shape_norm_penalty']),
            np.mean(val_log['text_norm_penalty'])
        ))
