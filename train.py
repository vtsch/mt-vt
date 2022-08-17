import time
import torch
import os
import torch.nn as nn
from torch.optim import Adam
import pandas as pd
from metrics import Meter
from dataloader import get_dataloader, data_generator
import numpy as np
from transformer import generate_square_subsequent_mask
from models.TC import TC

class Trainer:
    def __init__(self, config, experiment, data, net):
        self.net = net.to(config.device)
        self.config = config
        self.experiment = experiment
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(self.net.parameters(), lr=config.lr, betas=(config.beta1, config.beta2), weight_decay=3e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')
        self.best_loss = float('inf')
        self.phases = ['train', 'val', 'test']
        self.dataloaders = {
            phase: get_dataloader(config, data, phase) for phase in self.phases
        }
        self.attention_masks = generate_square_subsequent_mask(6)
        self.temporal_contr_model = TC(config).to(config.device)
        self.temp_cont_optimizer = torch.optim.Adam(self.temporal_contr_model.parameters(), lr=config.lr, betas=(config.beta1, config.beta2), weight_decay=3e-4)
        self.tstcc_train_dl, self.tstcc_valid_dl, self.tstcc_test_dl = data_generator(data, config)
    
    def _train_epoch(self, phase):

        self.net.train()
        meter = Meter()
        meter.init_metrics(phase)

        for i, (data, target, index) in enumerate(self.dataloaders[phase]):
            #data = data.to(config.device)
            #target = target.to(config.device)
            #if config.NOPOSENC:
            #index = torch.zeros(index.shape) 

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

    def _train_tstcc_epoch(self):
        meter = Meter()
        meter.init_metrics('train')

        self.net.train()
        self.temporal_contr_model.train()

        for batch_idx, (data, labels, aug1, aug2) in enumerate(self.tstcc_train_dl):
            # send to device
            data, labels = data.float().to(self.config.device), labels.long().to(self.config.device)
            aug1, aug2 = aug1.float().to(self.config.device), aug2.float().to(self.config.device)

            # optimizer
            self.optimizer.zero_grad()
            self.temp_cont_optimizer.zero_grad()

            output = self.net(data)
                
            predictions, features = output
            loss = self.criterion(predictions, labels)
            loss.backward()
            self.optimizer.step()
            self.temp_cont_optimizer.step()

            pred = predictions.max(1, keepdim=True)[1]
            meter.update(pred.detach().numpy(), labels, 'train', loss.item())

        metrics = meter.get_metrics()
        metrics = {k:v / batch_idx for k, v in metrics.items()}
        df_logs = pd.DataFrame([metrics])

        return loss, df_logs
        
    
    def run(self):
        for epoch in range(self.config.n_epochs):
            print('Epoch: %d | time: %s' %(epoch, time.strftime('%H:%M:%S')))
            
            if self.config.MOD_TSTCC:
                train_loss, train_logs = self._train_tstcc_epoch()
                print(train_logs)
                self.experiment.log_metrics(dic=train_logs, step=epoch)
                val_loss, valid_acc, outs, trgs, features, val_logs = self.eval()
                print(val_logs)
                self.experiment.log_metrics(dic=val_logs, step=epoch)
                self.scheduler.step(val_loss)
            else:
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
        self.temporal_contr_model.eval()

        total_loss = []
        total_acc = []
        outs = np.array([])
        targets = np.array([])
        embeddings = np.array([])

        with torch.no_grad():
            if self.config.MOD_TSTCC:
                meter = Meter()
                meter.init_metrics('val')
                for batch_idx, (data, labels, _, _) in enumerate(self.tstcc_test_dl):
                    data = data.float().to(self.config.device)
                    labels = labels.long().to(self.config.device)
                    output = self.net(data)

                    # compute loss
                    predictions, features = output
                    loss = self.criterion(predictions, labels)
                    total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())
                    total_loss.append(loss.item())
                    embeddings = np.append(embeddings, features.detach().numpy())

                    pred = predictions.max(1, keepdim=True)[1]  # get the index of the max log-probability
                    meter.update(pred.detach().numpy(), labels, 'val', loss.item())
                    outs = np.append(outs, pred.cpu().numpy())
                    targets = np.append(targets, labels.data.cpu().numpy())
                    embeddings = np.append(embeddings, features.detach().numpy())

                metrics = meter.get_metrics()
                metrics = {k:v / batch_idx for k, v in metrics.items()}
                df_logs = pd.DataFrame([metrics])

                total_loss = torch.tensor(total_loss).mean()  # average loss
                total_acc = torch.tensor(total_acc).mean()  # average acc
                embeddings = embeddings.reshape(targets.shape[0], -1, features.shape[2])

            else:
                for i, (data, target, index) in enumerate(self.dataloaders['test']):

                    if self.config.MOD_TRANSFORMER == True:
                        data = data.squeeze(1)
                        index = index.squeeze(1)
                        pred, psu_class, transf_emb, transf_reconst = self.net(index, data, self.attention_masks)
                    else:
                        pred = self.net(data)
                    
                    embeddings = np.append(embeddings, pred.detach().numpy() )
                    targets = np.append(targets, target.detach().numpy())  #always +bs
            
                embeddings = embeddings.reshape(targets.shape[0], -1)

        return total_loss, total_acc, outs, targets, embeddings, df_logs

            
            

            


