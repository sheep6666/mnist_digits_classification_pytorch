import os
import cv2
import PIL
import pandas as pd

import cfg
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class MnistDataset(Dataset):
    def __init__(self, label_path, img_dir, transform=None):
        self.label_path = label_path
        self.img_dir = img_dir
        self.transform = transform

        self.label_df = pd.read_csv(label_path).sample(10000).reset_index(drop=True)
        
    def __len__(self):
        return len(self.label_df)

    def __getitem__(self, idx):
        img_name = self.label_df.loc[idx, "name"]
        label    = self.label_df.loc[idx, "label"]
        
        img_path = os.path.join(self.img_dir, img_name)
        image = PIL.Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image, label
    
def build_transform():
    img_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    return img_transforms


def build_loader():
    
    img_transforms = build_transform()
        
    train_dataset = MnistDataset(
        label_path=cfg.train_label_path, #r"/DATA_1/Projects/exercise/mnist_digits_classification_pytorch/mnist/train_labels.csv",
        img_dir=cfg.train_img_dir, #r"/DATA_1/Projects/exercise/mnist_digits_classification_pytorch/mnist/train",
        transform=img_transforms
    )
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    
    test_dataset = MnistDataset(
        label_path=cfg.test_label_path,#r"/DATA_1/Projects/exercise/mnist_digits_classification_pytorch/mnist/test_labels.csv",
        img_dir=cfg.test_img_dir, #r"/DATA_1/Projects/exercise/mnist_digits_classification_pytorch/mnist/test",
        transform=img_transforms
    )
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    return train_dataset, train_loader, test_dataset, test_loader


