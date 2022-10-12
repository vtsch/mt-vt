from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score
from pos_enc import positional_encoding
from models.tstcc_loss import NTXentLoss
from models.tstcc_TC import TC
from models.transformer import generate_square_subsequent_mask
from dataloader import data_generator_tstcc
from torch.optim import Adam
import torch.nn.functional as F
import torch.nn as nn
import torch
from typing import Tuple
from bunch import Bunch
import numpy as np
import os
import sys
import pandas as pd
sys.path.append("..")


class TSTCCTrainer:
    def __init__(self, config: Bunch, experiment, data: pd.DataFrame, net: nn.Module) -> None:
        '''
        Initialize the trainer
        Args:
            config: configuration parameters
            experiment: comet_ml experiment
            data: data to train on
            net: model to train
        '''
        self.net = net.to(config.device)
        self.config = config
        self.experiment = experiment
        self.df_psa_u, self.df_psa_orig = data
        self.data = self.df_psa_u if self.config.upsample else self.df_psa_orig
        self.criterion = torch.nn.CrossEntropyLoss(
        ) if self.config.tstcc_training_mode == 'supervised' else torch.nn.MSELoss()
        self.optimizer = Adam(self.net.parameters(
        ), lr=config.lr, betas=(0.9, 0.99), weight_decay=3e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'min')
        self.best_loss = float('inf')
        self.attention_masks = generate_square_subsequent_mask(self.config)
        self.temporal_contr_model = TC(config).to(config.device)
        self.temp_cont_optimizer = Adam(self.temporal_contr_model.parameters(
        ), lr=config.lr, betas=(0.9, 0.99), weight_decay=3e-4)
        self.tstcc_train_dl, self.tstcc_valid_dl, self.tstcc_test_dl = data_generator_tstcc(
            self.data, config)
        _, _, self.tstcc_test_dl_orig = data_generator_tstcc(
            self.df_psa_orig, config)
        self.clf = KNeighborsClassifier(n_neighbors=1)

    def run(self) -> None:
        '''
        Run the TS-TCC training
        '''
        # Start training
        for epoch in range(1, self.config.n_epochs + 1):
            # Train and validate
            train_loss, train_acc = self.model_train()
            self.experiment.log_metrics(
                dic={'train_loss': train_loss, 'train_acc': train_acc}, step=epoch)
            valid_loss, valid_acc, _, _, _ = self.model_evaluate(phase='val')
            self.experiment.log_metrics(
                dic={'valid_loss': valid_loss, 'valid_acc': valid_acc}, step=epoch)
            # use scheduler in all other modes.
            if self.config.tstcc_training_mode != 'self_supervised':
                self.scheduler.step(valid_loss)

            print(f'\nEpoch : {epoch}\n'
                  f'Train Loss     : {train_loss:.4f}\t | \tTrain Accuracy     : {train_acc:2.4f}\n'
                  f'Valid Loss     : {valid_loss:.4f}\t | \tValid Accuracy     : {valid_acc:2.4f}')

        os.makedirs(self.config.model_save_path, exist_ok=True)
        chkpoint = {'model_state_dict': self.net.state_dict(
        ), 'temporal_contr_model_state_dict': self.temporal_contr_model.state_dict()}
        torch.save(chkpoint, os.path.join(
            self.config.model_save_path, f'ckp_last.pt'))

        # no need to run the evaluation for self-supervised mode.
        if self.config.tstcc_training_mode != "self_supervised":
            # evaluate on the test set
            test_loss, test_acc, _, _, _ = self.model_evaluate(phase='test')
            self.experiment.log_metrics(
                dic={'test_loss': test_loss, 'test_acc': test_acc}, step=epoch)
            print(
                f'Test loss      :{test_loss:0.4f}\t | Test Accuracy      : {test_acc:0.4f}')

    def model_train(self) -> Tuple[torch.Tensor, torch.Tensor]:
        '''
        Train the model
        Returns:
            train_loss: training loss
            train_acc: training accuracy
        '''
        total_loss = []
        total_acc = []

        self.net.train()
        self.temporal_contr_model.train()

        for batch_idx, (psa_data, label, aug1, aug2, tsindex, context) in enumerate(self.tstcc_train_dl):
            # optimizer
            self.optimizer.zero_grad()
            self.temp_cont_optimizer.zero_grad()

            # apply positional encoding
            data_pos_enc = positional_encoding(self.config, psa_data, tsindex)

            if self.config.tstcc_training_mode == "self_supervised":
                # encode augmented data
                logits1, features1 = self.net(aug1)
                logits2, features2 = self.net(aug2)

                # normalize projection feature vectors
                features1 = F.normalize(features1, dim=1)
                features2 = F.normalize(features2, dim=1)

                # temporal contrasting model and add context
                temp_cont_loss1, temp_cont_lstm_feat1 = self.temporal_contr_model(
                    features1, features2, context)
                temp_cont_loss2, temp_cont_lstm_feat2 = self.temporal_contr_model(
                    features2, features1, context)

                # normalize projection feature vectors
                zis = temp_cont_lstm_feat1
                zjs = temp_cont_lstm_feat2
            else:
                if self.config.context:
                    # (batch_size, ts_length + context_dim)
                    data_pos_enc = torch.cat((data_pos_enc, context), dim=1)
                logits, features = self.net(data_pos_enc)

            # compute loss
            if self.config.tstcc_training_mode == "self_supervised":
                lambda1 = 1
                lambda2 = 0.7
                nt_xent_criterion = NTXentLoss(
                    self.config.device, self.config.batch_size, use_cosine_similarity=True)
                loss = (temp_cont_loss1 + temp_cont_loss2) * \
                    lambda1 + nt_xent_criterion(zis, zjs) * lambda2
            else:
                if self.config.tstcc_training_mode == "supervised":
                    loss = self.criterion(logits, label.long())
                    total_acc.append(
                        label.eq(logits.detach().argmax(dim=1)).float().mean())
                else:
                    loss = self.criterion(logits, psa_data)
                    total_acc.append(
                        label.eq(logits.detach().argmax(dim=1)).float().mean())

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

    def model_evaluate(self, phase: str) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:
        '''
        Evaluate the model (validation or test)
        Returns:
            total_loss: validation loss
            total_acc: validation accuracy
            preds: predicted labels
            trgs: true labels
            embeddings: learned embeddings of the data
        '''
        self.net.eval()
        self.temporal_contr_model.eval()

        total_loss = []
        total_acc = []

        preds = np.array([])
        true_labels = np.array([])
        embeddings = np.array([])

        tstcc_dl = self.tstcc_valid_dl if phase == 'val' else self.tstcc_test_dl_orig

        with torch.no_grad():
            for batch_idx, (psa_data, label, _, _, tsindex, context) in enumerate(tstcc_dl):
                psa_data, label, context = psa_data.float().to(self.config.device), label.long(
                ).to(self.config.device), context.float().to(self.config.device)

                # apply positional encoding
                data_pos_enc = positional_encoding(
                    self.config, psa_data, tsindex)

                if self.config.tstcc_training_mode == "self_supervised":
                    pass

                else:
                    if self.config.context:
                        # (batch_size, ts_length + context_dim)
                        data_pos_enc = torch.cat(
                            (data_pos_enc, context), dim=1)
                    logits, features = self.net(data_pos_enc)

                    # compute loss
                    if self.config.tstcc_training_mode == "supervised":
                        loss = self.criterion(logits, label.long())
                        total_acc.append(
                            label.eq(logits.detach().argmax(dim=1)).float().mean())
                    else:
                        loss = self.criterion(logits, psa_data)
                        total_acc.append(
                            label.eq(logits.detach().argmax(dim=1)).float().mean())
                    total_loss.append(loss.item())

                    # return embeddings, predictions and targets
                    if self.config.tstcc_training_mode == "supervised":
                        # get the index of the max log-probability
                        pred = logits.max(1, keepdim=True)[1]
                    else:
                        pred = features
                    preds = np.append(preds, pred.cpu().numpy())
                    true_labels = np.append(
                        true_labels, label.data.cpu().numpy())
                    embeddings = np.append(embeddings, logits.detach().numpy())

        if self.config.tstcc_training_mode != "self_supervised":
            total_loss = torch.tensor(total_loss).mean()  # average loss
            embeddings = embeddings.reshape(
                true_labels.shape[0], -1)  # reshape embeddings
            if phase == 'val':
                self.clf.fit(embeddings, true_labels)
            elif phase == 'test':
                # calculate representation accuracy with a 1NN classifier and log score
                nn_predictions = self.clf.predict(embeddings)
                representation_score = balanced_accuracy_score(
                    true_labels, nn_predictions)
                print(f"Representation Accuracy: {representation_score}")
                self.experiment.log_metric(
                    "rep_accuracy", representation_score)
                np.savetxt(os.path.join(self.config.model_save_path,
                           "rep_accuracy.txt"), np.array([representation_score]))
            else:
                print("training in supervised mode")
        else:
            total_loss = 0

        if self.config.tstcc_training_mode == "self_supervised":
            total_acc = 0
            return total_loss, total_acc, [], [], []
        else:
            total_acc = torch.tensor(total_acc).mean()  # average acc
            return total_loss, total_acc, preds, true_labels, embeddings
