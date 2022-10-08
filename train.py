from torch import nn
import time
import os
import torch
from bunch import Bunch
from torch.optim import Adam
import pandas as pd
from typing import Tuple
from metrics import Meter
from dataloader import get_dataloader
import numpy as np
from models.transformer import generate_square_subsequent_mask
from pos_enc import positional_encoding
from sklearn.metrics import balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier

class Trainer:
    def __init__(self, config: Bunch, experiment, data: pd.DataFrame, net: nn.Module):
        '''
        Initialize trainer
        Args:
            config: config file
            experiment: comet_ml experiment
            data: dict of dataloaders
            net: model
        '''
        self.net = net.to(config.device)
        self.config = config
        self.experiment = experiment
        self.criterion = torch.nn.MSELoss(reduction='sum')
        self.optimizer = Adam(self.net.parameters(), lr=config.lr, betas=(0.9, 0.99), weight_decay=3e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')
        self.best_loss = float('inf')
        self.phases = ['train', 'val', 'test']
        self.dataloaders = {
            phase: get_dataloader(config, data, phase) for phase in self.phases
        }
        self.attention_masks = generate_square_subsequent_mask(self.config)
        self.clf = KNeighborsClassifier(n_neighbors=1)
    
    def _add_posenc_and_context(self, data, tsindex, context):
        if self.config.experiment_name != "simple_transformer":
            data = positional_encoding(self.config, data, tsindex)
            if self.config.context:
                data = torch.cat((data, context), dim=1)

        if self.config.experiment_name == "simple_transformer":
            pred = self.net(tsindex, data, context, self.attention_masks)
            if self.config.context:
                data = torch.cat((data, context), dim=1)
        else: 
            pred = self.net(data)

        return pred, data

    
    def _train_epoch(self, phase: str) -> Tuple[float, dict]:
        '''
        Train one epoch
        Args:
            phase: train, val or test
        Returns:
            loss: loss of the epoch
            logs: logs of the epoch
        '''
        self.net.train()
        meter = Meter()
        meter.init_metrics(phase)
        labels = np.array([])
        embeddings = np.array([])

        for i, (data, label, tsindex, context) in enumerate(self.dataloaders[phase]):

            pred, data = self._add_posenc_and_context(data, tsindex, context)
 
            #print("pred shape for loss: ", pred.shape) 
            #print("data shape for loss: ", data.shape)
            loss = self.criterion(pred, data)

            if phase == 'train':
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            if phase == 'val':
                embeddings = np.append(embeddings, pred.detach().numpy() )
                labels = np.append(labels, label.detach().numpy() )

            meter.update(pred, data, phase, loss.item())

        metrics = meter.get_metrics()
        metrics = {k:v / i for k, v in metrics.items()}  # i = nr of batches
        df_logs = pd.DataFrame([metrics])

        if phase == 'val':
            embeddings = embeddings.reshape(labels.shape[0], -1)
            self.clf.fit(embeddings, labels)

        return loss, df_logs            
    
    def run(self) -> None:
        '''
        Run training
        '''
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
                #save best model 
                torch.save(self.net.state_dict(), os.path.join(self.config.model_save_path, f"best_model_epoc{epoch}.pth"))


    def eval(self) -> Tuple[np.ndarray, np.ndarray, dict]:
        '''
        Evaluate model
        Returns:
            labels: true labels
            embeddings: embeddings of the test set
            df_logs: logs
        '''
        self.net.eval()
        labels = np.array([])
        embeddings = np.array([])

        with torch.no_grad():

            for i, (data, label, tsindex, context) in enumerate(self.dataloaders['test']):
                pred, _ = self._add_posenc_and_context(data, tsindex, context)
                embeddings = np.append(embeddings, pred.detach().numpy() )
                labels = np.append(labels, label.detach().numpy())  #always +bs
            
            embeddings = embeddings.reshape(labels.shape[0], -1)
            
            # calculate representation accuracy with a 1NN classifier and log score
            nn_predictions = self.clf.predict(embeddings)
            representation_score = balanced_accuracy_score(labels, nn_predictions)
            print(f"Representation Accuracy: {representation_score}")
            self.experiment.log_metric("rep_accuracy", representation_score)
            np.savetxt(os.path.join(self.config.model_save_path, "rep_accuracy.txt"), np.array([representation_score]))

        return labels, embeddings

