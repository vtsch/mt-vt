import os
import sys

sys.path.append("..")
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.loss import NTXentLoss



def TSTCCTrainer(model, temporal_contr_model, train_dl, valid_dl, test_dl, config):
    # Start training
    model_optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, betas=(config.beta1, config.beta2), weight_decay=3e-4)
    temporal_contr_optimizer = torch.optim.Adam(temporal_contr_model.parameters(), lr=config.lr, betas=(config.beta1, config.beta2), weight_decay=3e-4)

    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min')

    for epoch in range(1, config.n_epochs + 1):
        # Train and validate
        train_loss, train_acc = model_train(model, temporal_contr_model, model_optimizer, temporal_contr_optimizer, criterion, train_dl, config)
        valid_loss, valid_acc, _, trgs, features = model_evaluate(model, temporal_contr_model, valid_dl, config)
        scheduler.step(valid_loss)

        if epoch % 5 == 0:
            print(f'\nEpoch : {epoch}\n'
                        f'Train Loss     : {train_loss:.4f}\t | \tTrain Accuracy     : {train_acc:2.4f}\n'
                        f'Valid Loss     : {valid_loss:.4f}\t | \tValid Accuracy     : {valid_acc:2.4f}')

    #os.makedirs(os.path.join(config.model_save_dir, "saved_models"), exist_ok=True)
    #chkpoint = {'model_state_dict': model.state_dict(), 'temporal_contr_model_state_dict': temporal_contr_model.state_dict()}
    #torch.save(chkpoint, os.path.join(config.model_save_dir, "saved_models", f'ckp_last.pt'))

    # evaluate on the test set
    test_loss, test_acc, outs, trgs, features = model_evaluate(model, temporal_contr_model, test_dl, config)
    print(f'Test loss      :{test_loss:0.4f}\t | Test Accuracy      : {test_acc:0.4f}')

    print("\n################## Training is Done! #########################")

    return features, trgs


def model_train(model, temporal_contr_model, model_optimizer, temp_cont_optimizer, criterion, train_loader, config):
    total_loss = []
    total_acc = []
    model.train()
    temporal_contr_model.train()

    for batch_idx, (data, labels, aug1, aug2) in enumerate(train_loader):
        # send to device
        data, labels = data.float().to(config.device), labels.long().to(config.device)
        aug1, aug2 = aug1.float().to(config.device), aug2.float().to(config.device)

        # optimizer
        model_optimizer.zero_grad()
        temp_cont_optimizer.zero_grad()

        output = model(data)
            
        predictions, features = output
        loss = criterion(predictions, labels)
        total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())

        total_loss.append(loss.item())
        loss.backward()
        model_optimizer.step()
        temp_cont_optimizer.step()

    total_loss = torch.tensor(total_loss).mean()
    total_acc = torch.tensor(total_acc).mean()
    return total_loss, total_acc


def model_evaluate(model, temporal_contr_model, test_dl, config):
    model.eval()
    temporal_contr_model.eval()

    total_loss = []
    total_acc = []

    criterion = nn.CrossEntropyLoss()
    outs = np.array([])
    trgs = np.array([])
    all_features = np.array([])

    with torch.no_grad():
        for data, labels, _, _ in test_dl:
            data = data.float().to(config.device)
            labels = labels.long().to(config.device)
            output = model(data)

            # compute loss
            predictions, features = output
            loss = criterion(predictions, labels)
            total_acc.append(labels.eq(predictions.detach().argmax(dim=1)).float().mean())
            total_loss.append(loss.item())

            pred = predictions.max(1, keepdim=True)[1]  # get the index of the max log-probability
            outs = np.append(outs, pred.cpu().numpy())
            trgs = np.append(trgs, labels.data.cpu().numpy())
            all_features = np.append(all_features, features.detach().numpy())

        total_loss = torch.tensor(total_loss).mean()  # average loss
        total_acc = torch.tensor(total_acc).mean()  # average acc

    all_features = all_features.reshape(trgs.shape[0], -1, features.shape[2])
    return total_loss, total_acc, outs, trgs, all_features
