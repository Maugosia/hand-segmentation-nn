from library.skinny_model_nid import Skinny
from library.skin_dataset import SkinDataset
import torch
import torchvision.transforms as transforms
#  from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
# from torch.utils.tensorboard import SummaryWriter
from torchmetrics import JaccardIndex
from tensorboard import program
from torch.utils.tensorboard import SummaryWriter

tracking_address = "out_data/logs"

EPOCHS = 40
LR = 0.001
SAVING = True
SAVING_PATH = "models/first/model"
N_SAVING = 5
BATCH_SIZE = 4


def dice_bce_loss(x, y):
    x = x.flatten()
    y = y.flatten()

    intersection = (x * y).sum()
    dice_loss = 1 - (2. * intersection) / (x.sum() + y.sum())
    BCE = F.binary_cross_entropy(x, y)
    loss = (BCE + dice_loss) / 2

    return loss


def append_new_line(file_name, text_to_append):
    """Append given text as a new line at the end of file"""
    # Open the file in append & read mode ('a+')
    with open(file_name, "a+") as file_object:
        # Move read cursor to the start of file.
        file_object.seek(0)
        # If file is not empty then append '\n'
        data = file_object.read(100)
        if len(data) > 0:
            file_object.write("\n")
        # Append text at the end of file
        file_object.write(text_to_append)


def prepare_data(batch_s):
    mean = torch.tensor([63.4062, 57.8478, 55.2177], dtype=torch.float64)
    std = torch.tensor([52.1825, 46.3412, 44.8947], dtype=torch.float64)

    transform_label = transforms.Compose([transforms.ToTensor()])
    transform_image = transforms.Compose([transforms.Normalize(mean, std)])

    data_1 = SkinDataset("data_type1.csv", 512, transform_image, transform_label)
    data_2a = SkinDataset("data_type2a.csv", 512, transform_image, transform_label)
    data_2b = SkinDataset("data_type2b.csv", 512, transform_image, transform_label)

    train_percent = 0.8
    train_len_1 = int(np.floor(train_percent * len(data_1)))
    train_len_2a = int(np.floor(train_percent * len(data_2a)))
    train_len_2b = int(np.floor(train_percent * len(data_2b)))

    train_1, test_1 = torch.utils.data.random_split(data_1, [train_len_1, len(data_1) - train_len_1],
                                                    generator=torch.Generator().manual_seed(42))
    train_2a, test_2a = torch.utils.data.random_split(data_2a, [train_len_2a, len(data_2a) - train_len_2a],
                                                      generator=torch.Generator().manual_seed(42))
    train_2b, test_2b = torch.utils.data.random_split(data_2b, [train_len_2b, len(data_2b) - train_len_2b],
                                                      generator=torch.Generator().manual_seed(42))

    data_train_val = torch.utils.data.ConcatDataset([train_1, train_2a, train_2b])
    data_test = torch.utils.data.ConcatDataset([test_1, test_2a, test_2b])

    val_percent = 0.15
    val_len = int(np.floor(val_percent * len(data_train_val)))
    data_train, data_val = torch.utils.data.random_split(data_train_val,
                                                         [len(data_train_val) - val_len, val_len],
                                                         generator=torch.Generator().manual_seed(42))

    print("len data_train", len(data_train))
    print("len data_val", len(data_val))
    print("len data_test_1", len(test_1))
    print("len data_test_2a", len(test_2a))
    print("len data_test_2b", len(test_2b))

    train_loader = DataLoader(data_train, shuffle=True, batch_size=batch_s)
    test_loader = DataLoader(data_test, shuffle=True, batch_size=batch_s)
    val_loader = DataLoader(data_val, shuffle=True, batch_size=batch_s)

    test1_loader = DataLoader(test_1, shuffle=True, batch_size=batch_s)
    test2a_loader = DataLoader(test_2a, shuffle=True, batch_size=batch_s)
    test2b_loader = DataLoader(test_2b, shuffle=True, batch_size=batch_s)
    return train_loader, val_loader, test_loader, test1_loader, test2a_loader, test2b_loader


def train_network(model, train_loader, val_loader, device, criterion, optimizer):
    writer = SummaryWriter()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = []

        # bar = tqdm(train_loader, position=0, leave=False, desc='epoch %d' % epoch)
        for batch in train_loader:
            images, masks = batch[0], batch[1]

            images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            masks = masks.to(device=device, dtype=torch.float32)

            logits = model(images)
            loss = criterion(logits, masks)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss.append(loss)

        avg_train_loss = torch.stack(train_loss).mean()
        writer.add_scalar('Loss/train', avg_train_loss, epoch)
        append_new_line('out_data/train.txt', str(avg_train_loss.item()))
        print(epoch, "/", EPOCHS, '   train_loss', avg_train_loss.item())

        if SAVING:
            if epoch % N_SAVING == 0:
                torch.save(model, SAVING_PATH + str(epoch))

        model.eval()
        with torch.no_grad():
            val_loss = []
            for batch in val_loader:
                images, masks = batch[0], batch[1]

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                masks = masks.to(device=device, dtype=torch.float32)

                logits = model(images)
                loss = criterion(logits, masks)

                val_loss.append(loss)
            avg_val_loss = torch.stack(val_loss).mean()
            writer.add_scalar('Loss/validation', avg_val_loss, epoch)
            append_new_line('out_data/val.txt', str(avg_val_loss.item()))
            print(epoch, "/", EPOCHS, '   val_loss', avg_val_loss.item())
    writer.flush()
    writer.close()


def test_network(model, test_loader, criterion, device, type_testing):
    iou = JaccardIndex(num_classes=2, task="binary").to(device)
    # bar = tqdm(test_loader, position=0, leave=False, desc='test')
    test_loss = []
    correct = 0
    total = 0
    number_of_batches = 0
    iou_sum = 0

    with torch.no_grad():
        for batch in test_loader:
            number_of_batches += 1

            images, masks = batch[0], batch[1]

            images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            masks = masks.to(device=device, dtype=torch.float32)

            logits = model(images)
            loss = criterion(logits, masks)

            iou_sum += iou(logits, masks)

            test_loss.append(loss)
        avg_test_loss = torch.stack(test_loss).mean()
        print(type_testing + '_loss', avg_test_loss.item())
        print(f"IOU score: {iou_sum / number_of_batches}")


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader, test1_loader, test2a_loader, test2b_loader = prepare_data(BATCH_SIZE)
    model = Skinny(3, 2)
    model.to(device=device)
    criterion = dice_bce_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, foreach=True)

    train_network(model, train_loader, val_loader, device, criterion, optimizer)

    test_network(model, test_loader, criterion, device, "test")
    test_network(model, test1_loader, criterion, device, "test1")
    test_network(model, test2a_loader, criterion, device, "test2a")
    test_network(model, test2b_loader, criterion, device, "test2b")

    test_network(model, train_loader, criterion, device, "train")
    test_network(model, val_loader, criterion, device, "validation")


if __name__ == "__main__":
    main()
