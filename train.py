import time
import torch
import torch.nn as nn
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import (CosineAnnealingLR,
                                      CosineAnnealingWarmRestarts,
                                      StepLR,
                                      ExponentialLR)
import pandas as pd
from utils import Meter
from dataloader import get_dataloader, get_test_dataloader
import numpy as np

#https://www.kaggle.com/code/polomarco/ecg-classification-cnn-lstm-attention-mechanism 

class Trainer:
    def __init__(self, config, train_data, test_data, net, lr, batch_size, num_epochs):
        #self.net = net.to(config.device)
        self.net = net.to('cpu')
        self.config = config
        self.num_epochs = num_epochs
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.net.parameters(), lr=lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epochs, eta_min=5e-6)
        self.best_loss = float('inf')
        self.phases = ['train', 'val']
        self.dataloaders = {
            phase: get_dataloader(train_data, phase, batch_size) for phase in self.phases
        }
        self.test_dataloader = get_test_dataloader(test_data, batch_size)
        self.train_df_logs = pd.DataFrame()
        self.val_df_logs = pd.DataFrame()

    
    def _train_epoch(self, phase):
        print(f"{phase} mode | time: {time.strftime('%H:%M:%S')}")
        
        self.net.train() if phase == 'train' else self.net.eval()
        meter = Meter()
        meter.init_metrics()
        
        for i, (data, target) in enumerate(self.dataloaders[phase]):
            #data = data.to(config.device)
            #target = target.to(config.device)
            output = self.net(data)
            loss = self.criterion(output, target)

            if phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            meter.update(output, target, loss.item())
        
        metrics = meter.get_metrics()
        metrics = {k:v / i for k, v in metrics.items()}
        df_logs = pd.DataFrame([metrics])
        #confusion_matrix = meter.get_confusion_matrix()
        
        if phase == 'train':
            self.train_df_logs = pd.concat([self.train_df_logs, df_logs], axis=0)
        else:
            self.val_df_logs = pd.concat([self.val_df_logs, df_logs], axis=0)
        
        # show logs
        print('{}: {}, {}: {}, {}: {}'
              .format(*(x for kv in metrics.items() for x in kv))
             )       
        return loss
    
    def run(self):
        history = dict(train_loss=[], val_loss=[])

        for epoch in range(self.num_epochs):
            train_loss = self._train_epoch(phase='train')
            history['train_loss'].append(train_loss.detach().numpy())
            with torch.no_grad():
                val_loss = self._train_epoch(phase='val')
                history['val_loss'].append(val_loss.detach().numpy())
                self.scheduler.step()
            
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                print('New checkpoint')
                self.best_loss = val_loss
                #torch.save(self.net.state_dict(), f"best_model_epoc{epoch}.pth")
            
            #clear_output()
            print('Epoch: %d, train loss: %f, val loss: %f' %(epoch, train_loss, val_loss))

        return history
    
    def eval(self, emb_size):
        #self.net.eval()
        embeddings = np.array([])
        targets = np.array([])
        with torch.no_grad():
            for i, (data, target) in enumerate(self.test_dataloader):
                #data = data.to(config.device)
                output = self.net(data)
                embeddings = np.append(embeddings, output.detach().numpy() )
                targets = np.append(targets, target.detach().numpy())  #always +bs

        embeddings = embeddings.reshape(-1, emb_size)
        return embeddings, targets


