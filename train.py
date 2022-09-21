import time
import torch
from torch.optim import Adam
import torch.nn.functional as F
import pandas as pd
from metrics import Meter
from dataloader import get_dataloader
import numpy as np
from models.transformer import generate_square_subsequent_mask
from pos_enc import positional_encoding


class Trainer:
    def __init__(self, config, experiment, data, net):
        self.net = net.to(config.device)
        self.config = config
        self.experiment = experiment
        self.criterion = config.loss_fn
        self.optimizer = Adam(self.net.parameters(), lr=config.lr, betas=(0.9, 0.99), weight_decay=3e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')
        self.best_loss = float('inf')
        self.phases = ['train', 'val', 'test']
        self.dataloaders = {
            phase: get_dataloader(config, data, phase) for phase in self.phases
        }
        self.attention_masks = generate_square_subsequent_mask(self.config)
    
    def _train_epoch(self, phase):

        self.net.train()
        meter = Meter()
        meter.init_metrics(phase)

        for i, (data, target, tsindex, context) in enumerate(self.dataloaders[phase]):
            #data = data.to(config.device)
            #target = target.to(config.device)

            if self.config.experiment_name != "simple_transformer":
                data = positional_encoding(self.config, data, tsindex)
                if self.config.context:
                    data = torch.cat((data, context), dim=1)

            if self.config.experiment_name == "simple_transformer":
                pred = self.net(tsindex, data, context, self.attention_masks)
                pred = pred.reshape(pred.shape[0], -1)   
            else: 
                pred = self.net(data)
 
            #print("data shape for loss: ", data.shape)
            #print("pred shape for loss: ", pred.shape)   
            loss = self.criterion(pred, data)

            if phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            meter.update(pred.detach().numpy(), data, phase, loss.item())

        metrics = meter.get_metrics()
        metrics = {k:v / i for k, v in metrics.items()}
        df_logs = pd.DataFrame([metrics])

        return loss, df_logs            
    
    def run(self):
        for epoch in range(self.config.n_epochs):
            print('Epoch: %d | time: %s' %(epoch, time.strftime('%H:%M:%S')))
            
            train_loss, train_logs = self._train_epoch(phase='train')
            print(train_logs)
            self.experiment.log_metrics(dic=train_logs, step=epoch)

            with torch.no_grad():
                val_loss, val_logs = self._train_epoch(phase='val')
                print(val_logs)
                self.experiment.log_metrics(dic=val_logs, step=epoch)
                self.scheduler.step(val_loss)
            
            if (val_loss + 0.001) < self.best_loss:
                self.best_loss = val_loss
                print('-- new checkpoint --')
                self.best_loss = val_loss
                #self.experiment.log_metrics(self.best_loss, step=epoch)
                #save best model 
                #torch.save(self.net.state_dict(), os.path.join(self.config.model_save_path, f"best_model_epoc{epoch}.pth"))


    def eval(self):
        self.net.eval()
        targets = np.array([])
        embeddings = np.array([])

        with torch.no_grad():
            for i, (data, target, tsindex, context) in enumerate(self.dataloaders['test']):

                if self.config.experiment_name != "simple_transformer":
                    data = positional_encoding(self.config, data, tsindex)
                    if self.config.context:
                        data = torch.cat((data, context), dim=1)

                if self.config.experiment_name == "simple_transformer":
                    pred = self.net(tsindex, data, context, self.attention_masks)
                    pred = pred.reshape(pred.shape[0], -1)   
                else: 
                    pred = self.net(data)

                embeddings = np.append(embeddings, pred.detach().numpy() )
                targets = np.append(targets, target.detach().numpy())  #always +bs
        
            embeddings = embeddings.reshape(targets.shape[0], -1)
            df_logs = pd.DataFrame([])

        return targets, embeddings, df_logs

            
            

            


