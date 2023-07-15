#%%
import argparse
import sys

import torch
from torch.utils.tensorboard import SummaryWriter

from pathlib import Path
from tqdm import tqdm
from torchmetrics import Accuracy, Precision
from datetime import datetime

from models import convolutional
from loaders import frame_only_loader as loader


def predict_batch(model, loss_fn, inputs_batch, targets_batch, device, t, mode, sw):
    inputs_batch = inputs_batch.to(device)
    targets_batch = targets_batch.to(device)
    preds_batch = model(inputs_batch)
    # threshold = torch.tensor([0.4]).to(device)
    # preds_batch = (preds_batch > threshold).float()

    loss = loss_fn(preds_batch, targets_batch)
    sw.add_scalar(f"{mode}/loss", loss, t.n)

    precision = Precision(task='multilabel', average='macro', num_labels=5).to(device)
    precision_value = precision(preds_batch, targets_batch)
    sw.add_scalar(f"{mode}/precision", precision_value, t.n)

    accuracy = Accuracy(task='multilabel', average='macro', num_labels=5).to(device)
    accuracy_value = accuracy(preds_batch, targets_batch)
    sw.add_scalar(f"{mode}/accuracy", accuracy_value, t.n)

    t.set_description(f"Loss: {torch.round(loss, decimals=5)} Accuracy: {accuracy_value} Precision: {precision_value}")
    sw.flush()
    return loss


def train(model, optimizer, loss_fn, loaders, n_epochs, device, sw):
    for epoch in range(n_epochs):
        print(f"Epoch {epoch}: Starting training cycle")
        # Training cycle
        model.train()
        for inputs_batch, targets_batch in (t := tqdm(loaders["train"])):
            # print(f"\tBatch {batch_num}")
            loss = predict_batch(model, loss_fn, inputs_batch, targets_batch, device, t, "train", sw)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation cycle
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Starting validation cycle")
            model.eval()
            for inputs_batch, targets_batch in (t := tqdm(loaders["valid"])):
                # print(f"\tBatch {batch_num}")
                loss = predict_batch(model, loss_fn, inputs_batch, targets_batch, device, t, "valid", sw)

    print(f"Finished training.")
    # Test cycle
    model.eval()
    with torch.no_grad():
        print(f"Epoch {epoch}: Starting testing cycle")
        for inputs_batch, targets_batch in (t := tqdm(loaders["test"])):
            # print(f"\tBatch {batch_num}")
            loss = predict_batch(model, loss_fn, inputs_batch, targets_batch, device, t, "test", sw)

if __name__ == "__main__":
    # Simulate arguments
    sys.argv = [sys.argv[0], 'f:/Work2/drum-onset-detection/data/ADTOF-master/dataset', '10', '--gpu']

    parser = argparse.ArgumentParser(prog="trainer.py",
                                     description="Train a model.")
    parser.add_argument('data_folder', help='Path to folder containing data')
    parser.add_argument('n_epochs', help='Number of epochs to train for')
    parser.add_argument('--gpu', action='store_true', help='Toggle for using GPU')
    args = parser.parse_args()

    data_folder = Path(args.data_folder)
    n_epochs = int(args.n_epochs)

    device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'

    model = convolutional.TFDConvNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    # label_weights = torch.tensor([1, 2, 3, 4, 5]).to(device)
    # loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=label_weights)
    label_weights = torch.tensor([0.059, 0.15, 0.214, 0.281, 0.296]).to(device)
    pos_weights = torch.tensor([76.04548784316461, 98.72052530284162, 61.23326264342795, 333.7409675443968, 199.35948788294468]).to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss(weight=label_weights, pos_weight=pos_weights)
    #loss_fn = torch.nn.BCEWithLogitsLoss()
    loaders = loader.create_dataloaders(data_folder, 0.05, 0.9, 512, 128, True)

    # TensorBoard
    now = datetime.now()
    sw = SummaryWriter(f'runs/{now.day}-{now.month}-{now.year}_{now.hour}-{now.minute}-{now.second}')

    train(model, optimizer, loss_fn, loaders, n_epochs, device, sw)