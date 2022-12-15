import os
from csv import writer

import torch


def append_csv_description(image_dir, label_dir, csv_file):
    with open(csv_file, 'a', newline='\n') as f_object:
        writer_object = writer(f_object)

        for file in os.listdir(image_dir):
            if file.endswith(".jpg"):
                base_name = file[:-4]
                csv_row = [os.path.join(image_dir, file),
                           os.path.join(label_dir, base_name + ".bmp")]
                writer_object.writerow(csv_row)

        f_object.close()


def get_all_dirs(dir_name):
    sub_dirs = [f.path for f in os.scandir(dir_name) if f.is_dir()]
    return sub_dirs


def create_segmentation_description_file(description_file, image_dir,
                                         label_dir,
                                         clear=False):
    if clear:
        if os.path.exists(description_file):
            print("clearing old content ...")
            os.remove(description_file)

    f = open(description_file, 'a', newline='')
    w = writer(f)
    if clear:
        w.writerow(["image_name", "label_name"])
    f.close()
    append_csv_description(image_dir, label_dir, description_file)
    print("\nfile {} generated".format(description_file))


def dice_bce_loss(x: torch.Tensor, y: torch.Tensor):
    """
    Calculate loss based on average of Dice loss
    and binary cross-entropy loss.
    """
    x = x.flatten()
    y = y.flatten()

    intersection = (x * y).sum()
    dice_loss = 1 - (2. * intersection) / (x.sum() + y.sum())
    BCE = F.binary_cross_entropy(x, y)
    loss = (BCE + dice_loss) / 2

    return loss
