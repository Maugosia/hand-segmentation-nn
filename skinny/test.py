from .parameters import Parameters
from .utils import calculate_logits_and_loss
from .models import SkinnyBasic, SkinnyInception

from typing import Union

from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch

from torchmetrics import JaccardIndex

from tqdm.notebook import tqdm


def test(trained_model: Union[SkinnyBasic, SkinnyInception], parameters: Parameters, test_dataset: Dataset):
    """
    Main testing procedure.
    """
    test_loader = DataLoader(test_dataset, parameters.batch_size, shuffle=True)
    test_loss = []
    number_of_batches = 0
    iou_sum = 0

    bar = tqdm(test_loader, position=0, leave=False, desc="test")

    iou = JaccardIndex(num_classes=2, task="binary").to(parameters.device)

    trained_model.eval()

    with torch.no_grad():
        for batch in bar:
            number_of_batches += 1

            images, masks = batch

            logits, loss, masks = calculate_logits_and_loss(
                images, masks, trained_model, parameters.criterion, parameters.device)

            iou_sum += iou(logits[0][0], masks[0][0]).item()
            test_loss.append(loss)

        avg_test_loss = torch.stack(test_loss).mean()
        avg_iou_score = iou_sum / number_of_batches
        print('test_loss', avg_test_loss.item())
        print(f"IOU score: {avg_iou_score}")

    return avg_iou_score
