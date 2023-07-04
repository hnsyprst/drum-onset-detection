import argparse

import torch

from pathlib import Path

from models import recurrent
import loader


def predict_batch(model, loss_fn, inputs_batch, targets_batch):
    preds_batch = model(inputs_batch)
    loss = loss_fn(preds_batch, targets_batch)
    return loss


def train(model, optimizer, loss_fn, loaders, n_epochs):
    for epoch in range(n_epochs):
        print(f"Epoch {epoch}: Starting training cycle")
        # Training cycle
        model.train()
        for inputs_batch, targets_batch in loaders["train"]:
            # print(f"\tBatch {batch_num}")
            loss = predict_batch(model, loss_fn, inputs_batch, targets_batch)
            print(f"\Loss {loss}")
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation cycle
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Starting validation cycle")
            model.eval()
            for inputs_batch, targets_batch in loaders["valid"]:
                # print(f"\tBatch {batch_num}")
                loss = predict_batch(model, loss_fn, inputs_batch, targets_batch)
                print(f"\Loss {loss}")

    print(f"Finished training.")
    # Test cycle
    model.eval()
    with torch.no_grad():
        print(f"Epoch {epoch}: Starting testing cycle")
        for inputs_batch, targets_batch in loaders["test"]:
            # print(f"\tBatch {batch_num}")
            loss = predict_batch(model, loss_fn, inputs_batch, targets_batch)
            print(f"\Loss {loss}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="trainer.py",
                                     description="Train a model.")
    parser.add_argument('data_folder', help='Path to folder containing data')
    parser.add_argument('n_epochs', help='Number of epochs to train for')
    args = parser.parse_args()

    data_folder = Path(args.data_folder)
    n_epochs = int(args.n_epochs)

    model = recurrent.RecurrentModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()
    loaders = loader.create_dataloaders(data_folder, 0.05, 0.9, 512, 8, True)

    train(model, optimizer, loss_fn, loaders, n_epochs)