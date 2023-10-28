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


def predict_batch(model, loss_fn, inputs_batch, targets_batch, device, epoch, t, mode, sw, log):
    inputs_batch = inputs_batch.to(device)
    targets_batch = targets_batch.to(device)
    preds_batch = model(inputs_batch)
    # threshold = torch.tensor([0.4]).to(device)
    # preds_batch = (preds_batch > threshold).float()

    loss = loss_fn(preds_batch, targets_batch)

    precision = Precision(task='multilabel', average='macro', num_labels=5).to(device)
    precision_value = precision(preds_batch, targets_batch)

    accuracy = Accuracy(task='multilabel', average='macro', num_labels=5).to(device)
    accuracy_value = accuracy(preds_batch, targets_batch)

    t.set_description(f"Loss: {torch.round(loss, decimals=5)} Accuracy: {accuracy_value} Precision: {precision_value}")

    if log:
        sw.add_scalar(f"{mode}/loss", loss, epoch * len(t) + t.n)
        sw.add_scalar(f"{mode}/accuracy", accuracy_value, epoch * len(t) + t.n)
        sw.add_scalar(f"{mode}/precision", precision_value, epoch * len(t) + t.n)
        sw.flush()
    return loss, accuracy_value, precision_value


def save_checkpoint(model, optimizer, epoch, train_cycle_complete, output_dir, run_name):
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            "train_cycle_complete": train_cycle_complete,
            "run_name": run_name
            }, output_dir / f"{run_name}.tar")


def train(model, optimizer, loss_fn, loaders, n_epochs, device, sw, resume_info, output_dir, run_name):
    for epoch in range(n_epochs):
        # Skip completed epochs
        if epoch < resume_info["epoch"]:
            continue

        # Training cycle
        if not resume_info["train_cycle_complete"]:
            print(f"Epoch {epoch}: Starting training cycle")
            model.train()
            for inputs_batch, targets_batch in (t := tqdm(loaders["train"])):
                # print(f"\tBatch {batch_num}")
                loss, _, _ = predict_batch(model, loss_fn, inputs_batch, targets_batch, device, epoch, t, "train", sw, True)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        # Save checkpoint here
        resume_info["train_cycle_complete"] = True

        # Validation cycle
        print(f"Epoch {epoch}: Starting validation cycle")
        valid_loss = []
        valid_accuracy = []
        valid_precision = []
        model.eval()
        with torch.no_grad():
            for inputs_batch, targets_batch in (t := tqdm(loaders["valid"])):
                # print(f"\tBatch {batch_num}")
                loss, accuracy_value, precision_value = predict_batch(model, loss_fn, inputs_batch, targets_batch, device, epoch, t, "valid", sw, False)
                valid_loss.append(loss)
                valid_accuracy.append(accuracy_value)
                valid_precision.append(precision_value)
            sw.add_scalar(f"valid/loss", torch.stack(valid_loss).mean(), epoch)
            sw.add_scalar(f"valid/accuracy", torch.stack(valid_accuracy).mean(), epoch)
            sw.add_scalar(f"valid/precision", torch.stack(valid_precision).mean(), epoch)
            # Save checkpoint here
            resume_info["train_cycle_complete"] = False

    print(f"Finished training.")
    # Test cycle
    test_loss = []
    test_accuracy = []
    test_precision = []
    model.eval()
    with torch.no_grad():
        print(f"Epoch {epoch}: Starting testing cycle")
        for inputs_batch, targets_batch in (t := tqdm(loaders["test"])):
            # print(f"\tBatch {batch_num}")
            loss, accuracy_value, precision_value = predict_batch(model, loss_fn, inputs_batch, targets_batch, device, epoch, t, "test", sw, False)
            test_loss.append(loss)
            test_accuracy.append(accuracy_value)
            test_precision.append(precision_value)
            sw.add_scalar(f"valid/loss", torch.stack(test_loss).mean(), 0)
            sw.add_scalar(f"valid/accuracy", torch.stack(test_accuracy).mean(), 0)
            sw.add_scalar(f"valid/precision", torch.stack(test_precision).mean(), 0)

if __name__ == "__main__":
    # Simulate arguments
    sys.argv = [sys.argv[0], 'f:/Work2/drum-onset-detection/data/ADTOF-master/dataset', '10', '--gpu', '--checkpoint', 'f:/Work2/drum-onset-detection/models/MiniMobileNet/28-10-23-1422.tar']

    parser = argparse.ArgumentParser(prog="trainer.py",
                                     description="Train a model.")
    parser.add_argument('data_folder', help='Path to folder containing data')
    parser.add_argument('n_epochs', help='Number of epochs to train for')
    parser.add_argument('--gpu', action='store_true', help='Toggle for using GPU')
    parser.add_argument('--checkpoint', help='Checkpoint to resume training from', default=None)
    args = parser.parse_args()

    data_folder = Path(args.data_folder)
    n_epochs = int(args.n_epochs)
    device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(args.checkpoint) if args.checkpoint else None
    # TODO: argparse this
    output_dir = Path("f:/Work2/drum-onset-detection/models")

    model = convolutional.miniMobileNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # label_weights = torch.tensor([1, 2, 3, 4, 5]).to(device)
    # loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=label_weights)
    #label_weights = torch.tensor([0.059, 0.15, 0.214, 0.281, 0.296]).to(device)
    #pos_weights = torch.tensor([76.04548784316461, 98.72052530284162, 61.23326264342795, 333.7409675443968, 199.35948788294468]).to(device)
    #loss_fn = torch.nn.BCEWithLogitsLoss(weight=label_weights, pos_weight=pos_weights)
    # TODO: remove or investigate pos weights, doesn't work
    loaders, pos_weights = loader.create_dataloaders(data_folder, 0.05, 0.9, 512, 2048, True)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # Resume from checkpoint
    if checkpoint:
        # Load run name
        run_name = checkpoint['run_name']
        print(f"Resuming previous job: {run_name}")

        # Read resume_info
        resume_info = {"epoch": checkpoint["epoch"], "train_cycle_complete": checkpoint["train_cycle_complete"]}

        # Load previous model and optimizer states
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Tensorboard
        sw = SummaryWriter(run_name)
    else:
        # Create new run name
        now = datetime.now()
        run_name = f'runs/{now.day}-{now.month}-{now.year}_{now.hour}-{now.minute}-{now.second}'
        print(f"Starting new job: {run_name}")

        # Default resume_info
        resume_info = {"run_name": run_name, "epoch": 0, "train_cycle_complete": False}

    # Initialise Tensorboard
    sw = SummaryWriter(run_name)

    # Start the training run
    train(model, optimizer, loss_fn, loaders, n_epochs, device, sw, resume_info, output_dir, run_name)