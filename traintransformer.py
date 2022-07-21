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
from dataloader import dataloader
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score


class TransformerTrainer:
    def __init__(self, config, experiment, train_data, test_data, net):
        #self.net = net.to(config.device)
        self.net = net.to('cpu')
        self.config = config
        self.experiment = experiment
        self.optimizer = torch.optim.Adam(net.parameters(), lr=config.lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.n_epochs, eta_min=5e-6)
        self.best_loss = float('inf')
        self.phases = ['train', 'val']
        self.test_dataloader = dataloader(test_data, config.batch_size)
        self.train_dataloader = dataloader(train_data, config.batch_size)

    def run(self, model):
        """
        input: model, takes input: 3D tensor of size (batch, num_features, 1)
        """

        optimizer = Adam(model.parameters(), lr=self.config.lr)

        criterion = torch.nn.BCELoss()
        train_auc = []
        test_auc = []
        learning_rate_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda i: min(i / (self.config.lr / self.config.batch_size), 1.0))

        for epoch in range(self.config.n_epochs):
            model.train()
            epoch_loss = 0
            temp_train_auc = 0

           
            print('Epoch: %d | time: %s' %(epoch, time.strftime('%H:%M:%S')))

            for data, target in self.train_dataloader: 
                predictions = model(data)
                loss = criterion(predictions, target.float().view(-1, 1))
                epoch_loss += loss.item()
                temp_train_auc += roc_auc_score(
                    target.numpy(), predictions.detach().numpy())
                
                optimizer.zero_grad()
                loss.backward()
        
                # clip up
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

                optimizer.step()
                learning_rate_scheduler.step()
            
            train_auc.append(temp_train_auc/len(self.train_dataloader))
            if epoch % 2 == 0: print('train auc:', train_auc[-1], 'loss:', epoch_loss/len(self.train_dataloader))
            
            with torch.no_grad():
                model.eval()
                temp_test_auc = 0
                for data, target in self.test_dataloader:
                    predictions = model(data)
                    temp_test_auc += roc_auc_score(
                        target.numpy(), predictions.numpy())

            test_auc.append(temp_test_auc/len(self.test_dataloader))
            if epoch % 2 == 0: print('test auc:', test_auc[-1], 'loss:', epoch_loss/len(self.test_dataloader))
                
            
    def eval(self, emb_size):
        #self.net.eval()
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


