import argparse 
import numpy as np

import torch 
from torch.optim import Adam
from torch.nn import BCELoss
from sklearn.metrics import accuracy_score
from model import Net   
from data import data_loader



def _train_epoch(model, dataloader, optimizer, criterion):
    correct = 0
    total = 0
    model.train()

    for idx, (inputs, targets) in enumerate(dataloader):
        # compute output
        output = model(inputs.float())

        loss = criterion(output, targets.float())

        # compute batch accuracy 
        batch_preds = torch.max(output, 1)[1]
        batch_labels = torch.max(targets, 1)[1]
        batch_acc = accuracy_score(
            batch_labels.detach().numpy(), batch_preds.detach().numpy()
        )

        # accumulate number of correctly classified and total processed 
        correct += (batch_preds == batch_labels).astype(torch.int32).sum()
        total += len(inputs)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return correct/total


def validate(model, dataloader, criterion):
    correct = 0
    total = 0
    total_loss = 0
    model.eval()

    for idx, (inputs, targets) in enumerate(dataloader):
        # compute output
        output = model(inputs.float())
        loss = criterion(output, targets.float())

        batch_preds = torch.max(output, 1)[1]
        batch_labels = torch.max(targets, 1)[1]

        # accumulate number of correctly classified and total processed 
        correct += (batch_preds == batch_labels).astype(torch.int32).sum()
        total += len(inputs)
        total_loss += loss.item()
    
    return correct/total


def train(args):
    # get dataloaders for training and testing 
    train_loader, test_loader = data_loader(args.batch_size)

    # get model 
    model = Net()

    # setup optimizer and criterion 
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    criterion = BCELoss()

    for epoch in range(args.epochs):
        # train for one epoch
        epoch_acc = _train_epoch(model, train_loader, optimizer, criterion)
        val_acc = _train_epoch(model, test_loader)

        print(f'Epoch[{epoch}/{args.epochs}] Train Acc: {epoch_acc:.4f}    Test Acc: {val_acc:.4f}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch-size', type=int, default=1024)
    parser.add_argument('-e', '--epochs', type=int, default=50)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    args = parser.parse_args()

    # TODO: create checkpoint directory 

    # TODO: log parameters  

    train(args)
