import torch
from PIL import Image
import numpy as np
import argparse
import os
import csv
import sys 
sys.path.append("..") 
import clip_modified
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ClarityDataset(Dataset):

    def __init__(self,args,preprocess):
        # init path of label and csv
        self.img_folder_path = args.img_folder_path 
        self.csv_file_path = args.csv_file_path
        # csv format Filename High Low1 Low2 Low3 Low4 Split
        with open(self.csv_file_path, 'r') as file:
            self.csv_file = list(csv.reader(file))[1:]
        # define transforms
        self.transforms = transforms.Compose([
                transforms.Resize([224,224]),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])
        self.preprocess = preprocess

    def __len__(self):
        data_size = len(self.csv_file)
        return data_size
    
    def __getitem__(self,idx):
        # get image
        img_path = os.path.join(self.img_folder_path,self.csv_file[idx][0])
        img = Image.open(img_path).convert('RGB')
        img = self.preprocess(img)
        # get text
        txt = self.csv_file[idx][1]
        txt = clip_modified.tokenize(txt)
        return img,txt.squeeze()


parser = argparse.ArgumentParser()
parser.add_argument("--img_folder_path", type=str,
                    default="/root/autodl-nas/Clarity-Data/Clarity-PNGs")
parser.add_argument("--csv_file_path", type=str, 
                    default='/root/autodl-nas/Clarity-Data/captions.csv',
                    help="GPU id to work on, \'cpu\'.")
parser.add_argument("--gpu_id", type=str, default='cuda:0',
                    help="GPU id to work on, \'cpu\'.")
args = parser.parse_args()

model, preprocess = clip_modified.load("RN50", device = args.gpu_id, jit = False)
dataset = ClarityDataset(args,preprocess)

print(len(dataset))
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

for i, (img,txt) in enumerate(dataloader):
    #[batch_size, 3, 398, 224] [batch_size, 100, 1] [batch_size, 100, 4]
    #print("shapes", img.shape)
    #print("shapes", txt.shape)
    if i == 1: break
