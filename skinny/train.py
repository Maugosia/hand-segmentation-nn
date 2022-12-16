from .parameters import Parameters

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch

from tqdm.notebook import tqdm


def train(parameters: Parameters, train_dataset: Dataset, val_dataset: Dataset) -> nn.Module:
    """
    Main training procedure.
    """
    train_loader = DataLoader(
        train_dataset, parameters.batch_size, shuffle=True)
    val_dataset = DataLoader(val_dataset, parameters.batch_size, shuffle=True)

    model = parameters.model
    optimizer = parameters.optimizer

    writer = SummaryWriter()

    for epoch in range(parameters.epochs):
        model.train()
        train_loss = []

        bar = tqdm(train_loader, position=0, leave=False,
                   desc=f"epoch {epoch + 1}")

        for batch in bar:
            images, masks = batch
            logits, loss = calculate_logits_and_loss(
                images, model, parameters.criterion)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss.append(loss)

        avg_train_loss = torch.stack(train_loss).mean()
        writer.add_scalar("Loss/train", avg_train_loss, epoch + 1)
        print(f"Epoch: {epoch + 1} train loss: {avg_train_loss.item()}")

        model.eval()

        with torch.no_grad():
            val_loss = []

            for batch in val_loader:
                images, masks = batch
                logits, loss = calculate_logits_and_loss(
                    images, model, parameters.criterion)
                val_loss.append(loss)

            avg_val_loss = torch.stack(val_loss).mean()
            writer.add_scalar('Loss/validation', avg_val_loss, epoch + 1)
            print(f"Epoch: {epoch + 1} validation loss: {avg_val_loss.item()}")

    return model, writer


def calculate_logits_and_loss(x: torch.Tensor, y: torch.Tensor, model: nn.Module, criterion):
    """
    Runs tensor through the model and calculates loss function output.
    """
    x = x.to(device=device, dtype=torch.float32,
             memory_format=torch.channels_last)
    y = y.to(device=device, dtype=torch.float32)

    logits = model(x)
    loss = criterion(x, y)

    return logits, loss