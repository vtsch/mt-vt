import time
import torch
import torch.nn as nn
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import (CosineAnnealingLR,
                                      CosineAnnealingWarmRestarts,
                                      StepLR,
                                      ExponentialLR)
import pandas as pd
from metrics import Meter
from dataloader import get_dataloader, dataloader
import numpy as np


class Trainer:
    def __init__(self, config, experiment, train_data, test_data, net):
        self.net = net.to(config.device)
        self.config = config
        self.experiment = experiment
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.net.parameters(), lr=config.lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.n_epochs, eta_min=5e-6)
        self.best_loss = float('inf')
        self.phases = ['train', 'val']
        self.dataloaders = {
            phase: get_dataloader(train_data, phase, config.batch_size) for phase in self.phases
        }
        self.test_dataloader = dataloader(test_data, config.batch_size)
        self.train_df_logs = pd.DataFrame()
        self.val_df_logs = pd.DataFrame()

    
    def _train_epoch(self, config, phase):

        self.net.train() if phase == 'train' else self.net.eval()
        meter = Meter()
        meter.init_metrics(phase)

        for i, (data, target) in enumerate(self.dataloaders[phase]):
            #data = data.to(config.device)
            #target = target.to(config.device)
            output = self.net(data)
            loss = self.criterion(output, target)

            if phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            target = target.unsqueeze(1).repeat(1, config.emb_size)
            meter.update(output.detach().numpy(), target, phase, loss.item())

        metrics = meter.get_metrics()
        metrics = {k:v / i for k, v in metrics.items()}
        df_logs = pd.DataFrame([metrics])

        return loss, df_logs    
    
    def run(self, config):
        for epoch in range(self.config.n_epochs):
            print('Epoch: %d | time: %s' %(epoch, time.strftime('%H:%M:%S')))
            train_loss, train_logs = self._train_epoch(config, phase='train')
            print(train_logs)
            self.experiment.log_metrics(dic=train_logs, step=epoch)
            with torch.no_grad():
                val_loss, val_logs = self._train_epoch(config, phase='val')
                print(val_logs)
                self.experiment.log_metrics(dic=val_logs, step=epoch)
                self.scheduler.step()
            
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                print('New checkpoint')
                self.best_loss = val_loss
                #save best model here to config.model_save_path
                torch.save(self.net.state_dict(), config.model_save_path)
            
            self.experiment.log_metrics(pd.DataFrame({'train_loss': [train_loss.detach().numpy()], 'val_loss': [val_loss.detach().numpy()]}), epoch=epoch)

    
    def eval(self, config):
        self.net.eval()
        embeddings = np.array([])
        targets = np.array([])
        with torch.no_grad():
            for i, (data, target) in enumerate(self.test_dataloader):
                #data = data.to(config.device)
                output = self.net(data)
                embeddings = np.append(embeddings, output.detach().numpy() )
                targets = np.append(targets, target.detach().numpy())  #always +bs

        embeddings = embeddings.reshape(-1, config.emb_size)
        return embeddings, targets


