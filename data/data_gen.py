import torch
from PIL import Image
import numpy as np
import argparse
import os
import csv
import sys 
import random
sys.path.append("..") 
import clip_modified
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def imgaug(n_px):
    
    return Compose([
        Resize(n_px, interpolation=Image.BICUBIC),
        CenterCrop(n_px),
        #RandomVerticalFlip(p=3)
        lambda image: image.convert("RGB"),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.4), ratio=(0.3, 3.3), value=0),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
      
class ClarityDataset(Dataset):

    def __init__(self,args,preprocess,flag):
        # init path of label and csv
        self.img_folder_path = args.img_folder_path 
        self.csv_file_path = args.csv_file_path
        # csv format Filename High Low1 Low2 Low3 Low4 Split
        if flag=='train':
            with open(self.csv_file_path, 'r') as file:
                self.csv_file = list(csv.reader(file))[1:int(args.train_val_ratio*10200)]
        if flag=='val':
            with open(self.csv_file_path, 'r') as file:
                self.csv_file = list(csv.reader(file))[int(args.train_val_ratio*10200):]
        
        self.preprocess = preprocess
        self.num_captions = args.num_captions
         
    def __len__(self):
        data_size = len(self.csv_file)
        return data_size
    
    def __getitem__(self,idx):
        # get image
        img_path = os.path.join(self.img_folder_path,self.csv_file[idx][0])
        img = Image.open(img_path).convert('RGB')
        img = self.preprocess(img)
        # get text
        txt = ''
        for i in range(self.num_captions):
            txt += self.csv_file[idx][i+1]

        txt = clip_modified.tokenize(txt)
        return img,txt.squeeze()

class ClaritySepDataset(Dataset):

    def __init__(self,args,preprocess,flag):
        # init path of label and csv
        self.img_folder_path = args.img_folder_path 
        self.csv_file_path = args.csv_file_path
        # csv format Filename High Low1 Low2 Low3 Low4 Split
        if flag=='train':
            with open(self.csv_file_path, 'r') as file:
                self.csv_file = list(csv.reader(file))[1:int(args.train_val_ratio*46000)]
        if flag=='val':
            with open(self.csv_file_path, 'r') as file:
                self.csv_file = list(csv.reader(file))[int(args.train_val_ratio*46000):]
        
        self.preprocess = preprocess
        self.num_captions = args.num_captions
         
    def __len__(self):
        data_size = len(self.csv_file)
        return data_size
    
    def __getitem__(self,idx):
        # get image
        img_path = os.path.join(self.img_folder_path,self.csv_file[idx][0])
        img = Image.open(img_path).convert('RGB')
        img = self.preprocess(img)
        #img = self.transforms(img)
        
        # get text
        txt = self.csv_file[idx][1]
        txt = clip_modified.tokenize(txt)
        return img,txt.squeeze()

class RicoCapDataset(Dataset):

    def __init__(self,args,preprocess,flag):
        # init path of label and csv
        self.img_folder_path = args.img_folder_path 
        self.csv_file_path = args.csv_file_path
        # csv format Filename High Low1 Low2 Low3 Low4 Split
        if flag=='train':
            with open(self.csv_file_path, 'r') as file:
                self.csv_file = list(csv.reader(file))[1:int(args.train_val_ratio*112000)]
        if flag=='val':
            with open(self.csv_file_path, 'r') as file:
                self.csv_file = list(csv.reader(file))[int(args.train_val_ratio*112000):]
                
        # define transforms
        self.preprocess = preprocess
        self.num_captions = args.num_captions

    def __len__(self):
        data_size = len(self.csv_file)
        return data_size
    
    def __getitem__(self,idx):
        # get image
        img_path = os.path.join(self.img_folder_path,self.csv_file[idx][0]) + '.jpg'
        img = Image.open(img_path).convert('RGB')
        # resize to match caption
        img = img.resize((1440, 2560))
        # preprocess
        img = self.preprocess(img)
        # get text
        txt = self.csv_file[idx][1]
        #print(txt)
        txt = clip_modified.tokenize(txt)
        return img,txt.squeeze()

class RicoClsDataset(Dataset):

    def __init__(self,args,preprocess,sample_num):
        # init path of label and csv
        self.img_folder_path = args.img_folder_path 
        self.csv_file_path = args.csv_file_path
        # init num of labels
        self.num_class = args.num_class
        # set number of testing samples
        self.sample_num = sample_num
        # csv format Filename High Low1 Low2 Low3 Low4 Split
        with open(self.csv_file_path, 'r') as file:
            self.csv_file = list(csv.reader(file))[1:self.sample_num]
        # define transforms
        self.transforms = transforms.Compose([
                transforms.Resize([224,224]),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])
        self.preprocess = preprocess
        self.num_captions = args.num_captions
        self.label_dict = {'Checkbox':0, 'List Item':1, 'Number Stepper':2, 
            'Image':3, 'Toolbar':4, 'Multi-Tab':5, 'Card':6, 'Text':7,  
            'Drawer':8, 'Button Bar':9, 'Map View':10, 'Date Picker':11, 'Text Button':12, 
            'Pager Indicator':13, 'Input':14, 'Slider':15, 'Video':16, 'Modal':17,  
            'Radio Button':18, 'Bottom Navigation':19, 'Icon':20, 'On/Off Switch':21}

    def __len__(self):
        data_size = len(self.csv_file)
        return data_size
    
    def __getitem__(self,idx):
        # get image
        img_path = os.path.join(self.img_folder_path,self.csv_file[idx][0])+'.jpg'
        #print(img_path)
        img = Image.open(img_path).convert('RGB')
        # resize to match caption
        img = img.resize((1440, 2560))
        # preprocess
        img = self.preprocess(img)
        # get multilabels
        labels = self.csv_file[idx][1]
        labels = labels[1:-1].split(', ')
        label_oh = np.zeros(self.num_class)
        for temp in labels:
            if temp[1:-1] in self.label_dict:
                label_oh[self.label_dict[temp[1:-1]]]=1

        return img.cuda(),label_oh
    
class RicoGPTDataset(Dataset):

    def __init__(self,args,preprocess,flag):
        # init path of label and csv
        self.img_folder_path = args.img_folder_path 
        self.csv_file_path = args.csv_file_path
        # csv format: [index,bbox/leaves/global,captions]
        if flag=='train':
            with open(self.csv_file_path, 'r') as file:
                #0:int(args.train_val_ratio*208767)
                self.csv_file = list(csv.reader(file))
                random.shuffle(self.csv_file)
                self.csv_file = self.csv_file[:int(args.train_val_ratio*208767)]
        if flag=='val':
            with open(self.csv_file_path, 'r') as file:
                #int(args.train_val_ratio*208767):
                self.csv_file = list(csv.reader(file))
                random.shuffle(self.csv_file)
                self.csv_file = self.csv_file[int(args.train_val_ratio*208767):]
                
        # get transforms
        self.preprocess = preprocess
        #self.num_captions = args.num_captions

    def __len__(self):
        data_size = len(self.csv_file)
        return data_size
    
    def __getitem__(self,idx):
        # get image
        img_path = os.path.join(self.img_folder_path,self.csv_file[idx][0]) + '.jpg'
        img = Image.open(img_path).convert('RGB')
        # resize to match caption
        img = img.resize((1440, 2560))
        # get bbox/global
        if 'global' in self.csv_file[idx][1]:
            img = self.preprocess(img)
        elif 'leaves' in self.csv_file[idx][1]:
            img = self.preprocess(img)
        else:
            bbox_str = self.csv_file[idx][1]
            bbox=[int(bb) for bb in bbox_str[1:-1].split(', ')]
            if bbox[2]>bbox[0] and bbox[3]>bbox[1]:
                img = img.crop((bbox[0],bbox[1],bbox[2],bbox[3]))
            img = self.preprocess(img)
            #print(f'resized image size"{img.shape}')
        # get text
        txt = self.csv_file[idx][2]
        txt_tokens = clip_modified.tokenize(txt)
        
        return img,txt_tokens.squeeze()

class MixDataset(Dataset):
    
    def __init__(self,args,preprocess,flag):
        
        # init path of label and csv
        self.img_folder_path = args.img_folder_path 
        self.csv_file_path = args.csv_file_path
        # csv format: [index,bbox/leaves/global,captions]
        if flag=='train':
            with open(self.csv_file_path, 'r') as file:
                #0:int(args.train_val_ratio*208767)
                self.csv_file = list(csv.reader(file))
                random.shuffle(self.csv_file)
                self.csv_file = self.csv_file[:int(args.train_val_ratio*366850)]
        if flag=='val':
            with open(self.csv_file_path, 'r') as file:
                #int(args.train_val_ratio*208767):
                self.csv_file = list(csv.reader(file))
                random.shuffle(self.csv_file)
                self.csv_file = self.csv_file[int(args.train_val_ratio*366850):]
                
        # get transforms
        self.preprocess = preprocess
        #self.num_captions = args.num_captions

    def __len__(self):
        data_size = len(self.csv_file)
        return data_size
    
    def __getitem__(self,idx):
        # clarity part
        if 'png' in self.csv_file[idx][0]:
            self.img_folder_path = "/root/autodl-nas/Clarity-Data/Clarity-PNGs"
            # get image
            img_path = os.path.join(self.img_folder_path,self.csv_file[idx][0])
            img = Image.open(img_path).convert('RGB')
            img = self.preprocess(img)
            # get text
            txt = self.csv_file[idx][1]
            txt = clip_modified.tokenize(txt)
            return img,txt.squeeze()

        else:
            self.img_folder_path = '/root/autodl-tmp/RICO/combined/'
            img_path = os.path.join(self.img_folder_path,self.csv_file[idx][0]) + '.jpg'
            img = Image.open(img_path).convert('RGB')
            # resize to match caption
            img = img.resize((1440, 2560))
            # preprocess
            if 'global' in self.csv_file[idx][1]: 
                img = self.preprocess(img)
                # get text
                txt = self.csv_file[idx][2]
                txt_tokens = clip_modified.tokenize(txt)

            elif 'leaves' in self.csv_file[idx][1]:
                img = self.preprocess(img)
                # get text
                txt = self.csv_file[idx][2]
                txt_tokens = clip_modified.tokenize(txt)

            elif '[' in self.csv_file[idx][1]:
                bbox_str = self.csv_file[idx][1]
                bbox=[int(bb) for bb in bbox_str[1:-1].split(', ')]
                if bbox[2]>bbox[0] and bbox[3]>bbox[1]:
                    img = img.crop((bbox[0],bbox[1],bbox[2],bbox[3]))
                img = self.preprocess(img)
                # get text
                txt = self.csv_file[idx][2]
                txt_tokens = clip_modified.tokenize(txt)

            else:
                img = self.preprocess(img)
                # get text
                txt = self.csv_file[idx][1]
                txt_tokens = clip_modified.tokenize(txt)   

        return img,txt_tokens.squeeze()

####### TESING code ######
if False:
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_folder_path", type=str,
                        default="/root/autodl-nas/Clarity-Data/Clarity-PNGs")
    parser.add_argument("--csv_file_path", type=str, 
                        default='/root/autodl-nas/Clarity-Data/captions.csv',
                        help="GPU id to work on, \'cpu\'.")
    parser.add_argument("--gpu_id", type=str, default='cuda:0',
                        help="GPU id to work on, \'cpu\'.")
    parser.add_argument("--num_captions", type=int,
                        default=3, help="captions number per one image")
    parser.add_argument("--num_class", type=int,
                        default=22, help="total labels number")
    parser.add_argument("--train_val_ratio", type=int,
                        default=0.7, help="train_val_ratio")
    args = parser.parse_args()

    print('testing...')
    args.img_folder_path = '/root/autodl-tmp/RICO/combined/'
    #args.csv_file_path = '/root/autodl-tmp/UI_ITC/data/screen_summaries.csv'
    args.csv_file_path = '/root/autodl-tmp/RicoGPT-Caption-final.csv'

    model, preprocess, preprocess_aug = clip_modified.load("RN50", device = args.gpu_id, jit = False)
    print('model loaded...')

    dataset= RicoGPTDataset(args,preprocess,'train')

    #args.csv_file_path = '/root/autodl-tmp/UI_ITC/data/screen_summaries.csv'
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    print(len(dataloader))
    print('entering loop...')

    for i, (img,txt) in enumerate(dataloader):
        #[batch_size, 3, 398, 224] [batch_size, 100, 1] [batch_size, 100, 4]
        print("shapes", img.shape)
        print("shapes", txt.shape)
        #print(txt)
        if i == 1: break
