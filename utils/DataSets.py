import torchvision.transforms as trns
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import os


class MyDataSet(Dataset):
    def __init__(self, root, transform):
        # --------------------------------------------
        # Initialize paths, transforms, and so on
        # --------------------------------------------
        self.transform = transform

        # Load image path and annotations
        folders = os.listdir(root)
        self.imgs = []
        self.lbls = []
        self.labelmap = {}
        self.lbl_count = {}
        for i, folder in enumerate(folders):
            files_name = os.listdir(f"{root}/{folder}")
            for file_name in files_name:
                self.imgs.append(f"{root}/{folder}/{file_name}")
                self.lbls.append(i)
                if i not in self.lbl_count:
                    self.lbl_count[i] = 0
                self.lbl_count[i] += 1
            self.labelmap[i] = folder

    def __getitem__(self, index):
        # --------------------------------------------
        # 1. Read from file (using numpy.fromfile, PIL.Image.open)
        # 2. Preprocess the data (torchvision.Transform)
        # 3. Return the data (e.g. image and label)
        # --------------------------------------------
        imgpath = self.imgs[index]
        img = Image.open(imgpath).convert('RGB')
        lbl = int(self.lbls[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, lbl

    def __len__(self):
        # --------------------------------------------
        # Indicate the total size of the dataset
        # --------------------------------------------
        return len(self.imgs)

    def num_classes(self):
        return len(self.labelmap)

    def lbls_num(self):
        return [self.lbl_count[x] for x in self.lbl_count]
