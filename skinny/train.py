from .parameters import Parameters
from .utils import calculate_logits_and_loss

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch

from tqdm.notebook import tqdm

from torchmetrics import JaccardIndex


def train(parameters: Parameters, train_dataset: Dataset, val_dataset: Dataset):
    """
    Main training procedure.
    """
    train_loader = DataLoader(
        train_dataset, parameters.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, parameters.batch_size, shuffle=True)

    model = parameters.model
    optimizer = parameters.optimizer

    writer = SummaryWriter()

    iou = JaccardIndex(num_classes=2, task="binary").to(parameters.device)

    for epoch in range(parameters.epochs):
        model.train()
        train_loss = []

        with tqdm(train_loader, unit="batch", desc=f"Epoch {epoch + 1}/{parameters.epochs}") as bar:
            for batch in bar:
                images, masks = batch
                logits, loss = calculate_logits_and_loss(
                    images, masks, model, parameters.criterion, parameters.device)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                train_loss.append(loss)
                bar.set_postfix(loss=loss.item())

        avg_train_loss = torch.stack(train_loss).mean()
        writer.add_scalar("Loss/train", avg_train_loss, epoch + 1)

        model.eval()

        with torch.no_grad():
            val_loss = []

            with tqdm(val_loader, colour="blue", unit="batch", desc=f"Validation epoch {epoch + 1}/{parameters.epochs}") as bar:
                for batch in bar:
                    images, masks = batch
                    logits, loss = calculate_logits_and_loss(
                        images, masks, model, parameters.criterion, parameters.device)

                    val_loss.append(loss)
                    iou_score = iou(logits, masks)
                    bar.set_postfix(loss=loss.item(), iou=iou_score)

            avg_val_loss = torch.stack(val_loss).mean()
            writer.add_scalar('Loss/validation', avg_val_loss, epoch + 1)

    return model, writer
