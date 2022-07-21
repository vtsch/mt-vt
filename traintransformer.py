from distutils.command.config import config
import time
from comet_ml import Experiment
from sklearn import metrics
import torch
import torch.nn as nn
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import (CosineAnnealingLR,
                                      CosineAnnealingWarmRestarts,
                                      StepLR,
                                      ExponentialLR)
from dataloader import dataloader, get_dataloader
import pandas as pd
from metrics import Meter
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score


class TransformerTrainer:
    def __init__(self, config, experiment, train_data, test_data, net):
        self.net = net.to(config.device)
        self.config = config
        self.experiment = experiment
        self.criterion = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(net.parameters(), lr=config.lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.n_epochs, eta_min=5e-6)
        self.best_loss = float('inf')
        self.phases = ['train', 'val']
        self.test_dataloader = dataloader(test_data, config.batch_size)
        self.train_dataloader = {
            phase: get_dataloader(train_data, phase, config.batch_size) for phase in self.phases
        }
        self.train_df_logs = pd.DataFrame()
        self.val_df_logs = pd.DataFrame()


    def _train_epoch(self, phase):
        """
        input: model, takes input: 3D tensor of size (batch, num_features, 1)
        """
        learning_rate_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda i: min(i / (self.config.lr / self.config.batch_size), 1.0))

        self.net.train() if phase == 'train' else self.net.eval()
        meter = Meter(self.config.n_clusters)
        meter.init_metrics()

        for i, (data, target) in enumerate(self.train_dataloader[phase]):

            predictions = self.net(data)
            loss = self.criterion(predictions, target.float().view(-1, 1))
            
            if phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                # clip up
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1)
                self.optimizer.step()
                learning_rate_scheduler.step()
            
            meter.update(predictions, target, loss.item())
                
        metrics = meter.get_metrics()
        metrics = {k:v / i for k, v in metrics.items()}
        df_logs = pd.DataFrame([metrics])

        if phase == 'train':
            self.train_df_logs = pd.concat([self.train_df_logs, df_logs], axis=0)
        else:
            self.val_df_logs = pd.concat([self.val_df_logs, df_logs], axis=0)
        
        # show logs
        print('{} phase | {}: {}, {}: {}, {}: {}'
              .format(phase, *(x for kv in metrics.items() for x in kv))
             )

        return loss, df_logs  
  

    def run(self, model):    
        for epoch in range(self.config.n_epochs):
            print('Epoch: %d | time: %s' %(epoch, time.strftime('%H:%M:%S')))
            train_loss, train_logs = self._train_epoch(phase='train')
            self.experiment.log_metrics(dic=train_logs, epoch=epoch)
            with torch.no_grad():
                val_loss, val_logs = self._train_epoch(phase='val')
                self.experiment.log_metrics(dic=val_logs, epoch=epoch)
                self.scheduler.step()
            
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                print('New checkpoint')
                self.best_loss = val_loss
                #save best model here
            
            self.experiment.log_metrics(pd.DataFrame({'train_loss': [train_loss.detach().numpy()], 'val_loss': [val_loss.detach().numpy()]}), epoch=epoch)

                
    def eval(self, emb_size):
        self.net.eval()
        predictions = np.array([])
        targets = np.array([])
        with torch.no_grad():
            for data, target in self.test_dataloader:
                #data = data.to(config.device)
                prediction = self.net(data)
                predictions = np.append(predictions, prediction.detach().numpy() )
                targets = np.append(targets, target.detach().numpy())  #always +bs

        #embeddings = embeddings.reshape(-1, emb_size)

        return predictions, targets


