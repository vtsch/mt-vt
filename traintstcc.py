import os
import sys

sys.path.append("..")
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from dataloader import data_generator_tstcc
from models.transformer import generate_square_subsequent_mask
from models.tstcc_TC import TC
from models.tstcc_loss import NTXentLoss


class TSTCCTrainer:
    def __init__(self, config, experiment, data, net):
        self.net = net.to(config.device)
        self.config = config
        self.experiment = experiment
        self.criterion = config.loss_fn
        self.optimizer = Adam(self.net.parameters(), lr=config.lr, betas=(0.9, 0.99), weight_decay=3e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min')
        self.best_loss = float('inf')
        self.attention_masks = generate_square_subsequent_mask(self.config)
        self.temporal_contr_model = TC(config).to(config.device)
        self.temp_cont_optimizer = Adam(self.temporal_contr_model.parameters(), lr=config.lr, betas=(0.9, 0.99), weight_decay=3e-4)
        self.tstcc_train_dl, self.tstcc_valid_dl, self.tstcc_test_dl = data_generator_tstcc(data, config)
    
    def run(self):
        # Start training
        for epoch in range(1, self.config.n_epochs + 1):
            # Train and validate
            train_loss, train_acc = self.model_train()
            self.experiment.log_metrics(dic={'train_loss': train_loss, 'train_acc': train_acc}, step=epoch)
            valid_loss, valid_acc, _, _, _ = self.model_evaluate()
            self.experiment.log_metrics(dic={'valid_loss': valid_loss, 'valid_acc': valid_acc}, step=epoch)
            if self.config.tstcc_training_mode != 'self_supervised':  # use scheduler in all other modes.
                self.scheduler.step(valid_loss)

            print(f'\nEpoch : {epoch}\n'
                        f'Train Loss     : {train_loss:.4f}\t | \tTrain Accuracy     : {train_acc:2.4f}\n'
                        f'Valid Loss     : {valid_loss:.4f}\t | \tValid Accuracy     : {valid_acc:2.4f}')

        os.makedirs(self.config.model_save_path, exist_ok=True)
        chkpoint = {'model_state_dict': self.net.state_dict(), 'temporal_contr_model_state_dict': self.temporal_contr_model.state_dict()}
        torch.save(chkpoint, os.path.join(self.config.model_save_path, f'ckp_last.pt'))

        if self.config.tstcc_training_mode != "self_supervised":  # no need to run the evaluation for self-supervised mode.
            # evaluate on the test set
            test_loss, test_acc, _, _, _ = self.model_evaluate()
            self.experiment.log_metrics(dic={'test_loss': test_loss, 'test_acc': test_acc}, step=epoch)
            print(f'Test loss      :{test_loss:0.4f}\t | Test Accuracy      : {test_acc:0.4f}')


    def model_train(self):
        total_loss = []
        total_acc = []

        self.net.train()
        self.temporal_contr_model.train()

        for batch_idx, (data, labels, aug1, aug2) in enumerate(self.tstcc_train_dl):
            # send to device
            data, labels = data.float().to(self.config.device), labels.long().to(self.config.device)
            aug1, aug2 = aug1.float().to(self.config.device), aug2.float().to(self.config.device)

            # optimizer
            self.optimizer.zero_grad()
            self.temp_cont_optimizer.zero_grad()

            if self.config.tstcc_training_mode == "self_supervised":
                predictions1, features1 = self.net(aug1)
                predictions2, features2 = self.net(aug2)

                # normalize projection feature vectors
                features1 = F.normalize(features1, dim=1)
                features2 = F.normalize(features2, dim=1)

                temp_cont_loss1, temp_cont_lstm_feat1 = self.temporal_contr_model(features1, features2)
                temp_cont_loss2, temp_cont_lstm_feat2 = self.temporal_contr_model(features2, features1)

                # normalize projection feature vectors
                zis = temp_cont_lstm_feat1 
                zjs = temp_cont_lstm_feat2 

            else:
                output = self.net(data)

            # compute loss
            if self.config.tstcc_training_mode == "self_supervised":
                lambda1 = 1
                lambda2 = 0.7
                nt_xent_criterion = NTXentLoss(self.config.device, self.config.batch_size,use_cosine_similarity=True)
                loss = (temp_cont_loss1 + temp_cont_loss2) * lambda1 +  nt_xent_criterion(zis, zjs) * lambda2
                
            else: # supervised training or fine tuining
                predictions, features = output
                loss = self.criterion(predictions, labels)
                total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())

            total_loss.append(loss.item())
            loss.backward()
            self.optimizer.step()
            self.temp_cont_optimizer.step()

        total_loss = torch.tensor(total_loss).mean()

        if self.config.tstcc_training_mode == "self_supervised":
            total_acc = 0
        else:
            total_acc = torch.tensor(total_acc).mean()
        return total_loss, total_acc


    def model_evaluate(self):
        self.net.eval()
        self.temporal_contr_model.eval()

        total_loss = []
        total_acc = []

        eval_criterion = nn.CrossEntropyLoss()
        outs = np.array([])
        trgs = np.array([])
        embeddings = np.array([])

        with torch.no_grad():
            for batch_idx, (data, labels, _, _) in enumerate(self.tstcc_test_dl):
                data, labels = data.float().to(self.config.device), labels.long().to(self.config.device)

                if self.config.tstcc_training_mode == "self_supervised":
                    pass
                else:
                    output = self.net(data)

                # compute loss
                if self.config.tstcc_training_mode != "self_supervised":
                    predictions, features = output
                    loss = eval_criterion(predictions, labels)
                    total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())
                    total_loss.append(loss.item())

                if self.config.tstcc_training_mode != "self_supervised":
                    pred = predictions.max(1, keepdim=True)[1]  # get the index of the max log-probability
                    outs = np.append(outs, pred.cpu().numpy())
                    trgs = np.append(trgs, labels.data.cpu().numpy())
                    embeddings = np.append(embeddings, features.detach().numpy())

        if self.config.tstcc_training_mode != "self_supervised":
            total_loss = torch.tensor(total_loss).mean()  # average loss
            embeddings = embeddings.reshape(trgs.shape[0], -1, features.shape[2])
        else:
            total_loss = 0

        if self.config.tstcc_training_mode == "self_supervised":
            total_acc = 0
            return total_loss, total_acc, [], [], []
        else:
            total_acc = torch.tensor(total_acc).mean()  # average acc
            return total_loss, total_acc, outs, trgs, embeddings
