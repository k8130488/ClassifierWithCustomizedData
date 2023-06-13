import torchvision.transforms as trns
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import os
from utils.DataSets import MyDataSet
import torch


def loadData00(path, batch_size, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], shuffle=False):
    dataset = MyDataSet(root=path, transform=trns.Compose([
            trns.ToTensor(),
            trns.Normalize(mean=mean, std=std),
        ]))
    # dataset.dataset.to(device)
    # dataset.to(device)
    loader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=4)
    return loader


def loadDataSet(path, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], resiz=None):
    if resiz is None:
        return MyDataSet(root=path, transform=trns.Compose([
                trns.ToTensor(),
                trns.Normalize(mean=mean, std=std),
            ]))
    elif type(resiz) is not tuple:
        print("Please give resize width and height in a tuple type like (w, h) or (h, w).")
    else:
        return MyDataSet(root=path, transform=trns.Compose([
            trns.Resize(resiz),
            trns.ToTensor(),
            trns.Normalize(mean=mean, std=std),
        ]))


if __name__ == '__main__':
    batch_size = 16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    train_set = loadDataSet("../dataset/dogs_vs_cats/train", resiz=(320, 320))
    valid_set = loadDataSet("../dataset/dogs_vs_cats/test", resiz=(320, 320))
    train_loader = DataLoader(
        dataset=train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    valid_loader = DataLoader(
        dataset=valid_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    a = train_loader.dataset.labelmap
    # train_loader01 = loadData00("dataset/SAR/test", batch_size)
    # Get images and labels in a mini-batch of train_loader
    for imgs, lbls in train_loader:
        print('Size of image:', imgs.size())  # batch_size * 3 * 224 * 224
        print('Type of image:', imgs.dtype)  # float32
        print('Size of label:', lbls.size())  # batch_size
        print('Type of label:', lbls.dtype)  # int64(long)
        break
else:
    pass
