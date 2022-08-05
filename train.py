import time
import torch
import os
import torch.nn as nn
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd
from metrics import Meter
from dataloader import get_dataloader
import numpy as np
from transformer import generate_square_subsequent_mask


class Trainer:
    def __init__(self, config, experiment, data, net):
        self.net = net.to(config.device)
        self.config = config
        self.experiment = experiment
        self.criterion = nn.MSELoss()
        self.optimizer = Adam(self.net.parameters(), lr=config.lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.n_epochs, eta_min=5e-6)
        self.best_loss = float('inf')
        self.phases = ['train', 'val', 'test']
        self.dataloaders = {
            phase: get_dataloader(config, data, phase) for phase in self.phases
        }
        self.attention_masks = generate_square_subsequent_mask(6)

    
    def _train_epoch(self, phase):

        self.net.train() if phase == 'train' else self.net.eval()
        meter = Meter()
        meter.init_metrics(phase)

        for i, (data, target, index) in enumerate(self.dataloaders[phase]):
            #data = data.to(config.device)
            #target = target.to(config.device)
            data = nn.functional.normalize(data, p=2, dim=2)

            if self.config.MOD_TRANSFORMER == True:
                data = data.squeeze(1)
                index = index.squeeze(1)
                pred, psu_class, transf_emb, transf_reconst = self.net(index, data, self.attention_masks)
                pred = pred.squeeze(1)
            else: 
                pred = self.net(data)
                data = data.squeeze(1)
            
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
                self.scheduler.step()
            
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                print('-- new checkpoint --')
                self.best_loss = val_loss
                #save best model 
                torch.save(self.net.state_dict(), os.path.join(self.config.model_save_path, f"best_model_epoc{epoch}.pth"))
            
            #self.experiment.log_metrics(pd.DataFrame({'train_loss': [train_loss.detach().numpy()], 'val_loss': [val_loss.detach().numpy()]}), epoch=epoch)

    
    def eval(self):
        self.net.eval()
        embeddings = np.array([])
        targets = np.array([])
        with torch.no_grad():
            for i, (data, target, index) in enumerate(self.dataloaders['test']):

                if self.config.MOD_TRANSFORMER == True:
                    data = nn.functional.normalize(data, p=2, dim=1).squeeze(1)
                    index = index.squeeze(1)
                    pred, psu_class, transf_emb, transf_reconst = self.net(index, data, self.attention_masks)
                else:
                    data = nn.functional.normalize(data, p=2, dim=1)
                    pred = self.net(data)
                
                embeddings = np.append(embeddings, pred.detach().numpy() )
                targets = np.append(targets, target.detach().numpy())  #always +bs

        embeddings = embeddings.reshape(-1, self.config.emb_size)
        return embeddings, targets


