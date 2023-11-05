import clip_modified
import torch
from PIL import Image
import numpy as np
import argparse
import os
import time
from tqdm import tqdm
from utils.evaluation_tools import *
from data.data_gen import *
from utils.metrics import *
from utils.model import getCLIP, getCAM
from utils.preprocess import getImageTranform
from clip_modified import build_model
from torch.utils.data import SubsetRandomSampler, DataLoader
import sklearn
from sklearn.metrics import label_ranking_average_precision_score as lrap
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score

import warnings

parser = argparse.ArgumentParser()
# data
parser.add_argument("--img_folder_path", type=str,
                    default="/root/autodl-nas/Clarity-Data/Clarity-PNGs")
parser.add_argument("--csv_file_path", type=str, 
                    default='/root/autodl-nas/Clarity-Data/captions.csv')
parser.add_argument("--num_captions", type=int,
                    default=3, help="captions number per one image")
parser.add_argument("--num_class", type=int,
                    default=22, help="total labels number")
parser.add_argument("--train_val_ratio", type=float,
                    default=0.85, help="train set and validation set dataset ratio")
# hyper parameters
parser.add_argument("--gpu_id", type=int, default=0, help="GPU to work on")
parser.add_argument("--batch_size", type=int, default=2, help="batch size")
parser.add_argument("--num_workers", type=int, default=0, help="workers")
parser.add_argument("--epochs", type=int, default=20, help="the number of epochs")
parser.add_argument("--lr", type=int, default=0.001, help="learning rate")
# model and cls
parser.add_argument("--clip_model_name", type=str, default='RN50', help="Model name of CLIP")
parser.add_argument("--sentence_prefix", type=str, default='word', help="input text type: \"sentence\", \"word\"")
parser.add_argument("--sim2log", type=str, default='softmax', help="method to compute logits from similarity")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

# get model and load weights 
itc_model, preprocess, preprocess_aug = clip_modified.load(args.clip_model_name, device = args.gpu_id, jit = False)
#model, target_layer, reshape_transform = getCLIP(model_name=args.clip_model_name, gpu_id=args.gpu_id)
itc_model.to(device)
itc_model.load_state_dict(torch.load('/root/autodl-tmp/UI_ITC/checkpoints/ckp_ricogpt_rn50_0415_epoch40.pt'))
print('model loaded')

# get optimizer
exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
include = lambda n, p: not exclude(n, p)

named_parameters = list(itc_model.named_parameters())
# filter out the frozen parameters
gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]
optimizer = torch.optim.AdamW(
    [{"params": gain_or_bias_params, "weight_decay": 0.005},
    {"params": rest_params, "weight_decay": 0.005}],
    lr=args.lr)

print("finish setting optimizer!")

# convert args about data path
args.img_folder_path = '/root/autodl-tmp/RICO/combined/'
args.csv_file_path = '/root/autodl-tmp/RICO/cls.csv'

# setup dataset
args.num_class = 22
rico_cls_dataset = RicoClsDataset(args, preprocess = preprocess, sample_num = 4000)

# get corresponding label dict
label_dict={'Checkbox':0, 'Advertisement':1, 'List Item':2, 'Number Stepper':3, 
            'Image':4, 'Toolbar':5, 'Multi-Tab':6, 'Card':7, 'Text':8,  
            'Drawer':9, 'Button Bar':10, 'Map View':11, 'Date Picker':12, 'Text Button':13, 
            'Web View':14, 'Pager Indicator':15, 'Input':16, 'Slider':17, 'Video':18, 'Modal':19,  
            'Radio Button':20, 'Bottom Navigation':21, 'Icon':22, 'On/Off Switch':23, 'Background Image':24
}

label_dict_22 ={'Checkbox':0, 'List Item':1, 'Number Stepper':2, 
            'Image':3, 'Toolbar':4, 'Multi-Tab':5, 'Card':6, 'Text':7,  
            'Drawer':8, 'Button Bar':9, 'Map View':10, 'Date Picker':11, 'Text Button':12, 
            'Pager Indicator':13, 'Input':14, 'Slider':15, 'Video':16, 'Modal':17,  
            'Radio Button':18, 'Bottom Navigation':19, 'Icon':20, 'On/Off Switch':21
}
label_list_22 = ['Checkbox', 'List Item', 'Number Stepper', 
            'Image', 'Toolbar', 'Multi-Tab', 'Card', 'Text',  
            'Drawer', 'Button Bar', 'Map View', 'Date Picker', 'Text Button', 
            'Pager Indicator', 'Input', 'Slider', 'Video', 'Modal',  
            'Radio Button', 'Bottom Navigation', 'Icon', 'On/Off Switch']

# split train and valid by setting indice
'''
indices = list(range(len(rico_cls_dataset)))
train_indices = indices[:int(len(rico_cls_dataset)*args.train_val_ratio)]
valid_indices = indices[int(len(rico_cls_dataset)*args.train_val_ratio):]

train_dataloader = DataLoader(rico_cls_dataset, batch_size=args.batch_size,
                  sampler=SubsetRandomSampler(train_indices))
val_dataloader = DataLoader(rico_cls_dataset, batch_size=args.batch_size,
                  sampler=SubsetRandomSampler(valid_indices))
print(f"finish loading data: train:{len(train_dataloader)*args.batch_size} val:{len(val_dataloader)*args.batch_size}")
'''

val_dataloader = DataLoader(rico_cls_dataset,batch_size=args.batch_size,num_workers=args.num_workers,shuffle=True)
print(f'testing images number: {len(val_dataloader)*args.batch_size}')

text_tokens = torch.cat([clip_modified.tokenize(f'there is a {label} in the scene') for label in label_list_22]).to(device)
#print(text_embeddings.shape)
#print(f'shape: {text_embeddings}')

def cls_evaluate(args, dataloader, text_tokens, itc_model, device):
    ap_list,f1_list=[],[]
    txt_embeddings = itc_model.encode_text(text_tokens)
    txt_embeddings /= txt_embeddings.norm(dim=-1, keepdim=True)
    
    for i, (imgs, labels) in tqdm(enumerate(dataloader)):
        if i==10:break
        # here, due to nuknow reason, imgs can not be loaded in cuda here
        # to resolve this, the img directly .cuda() in data_gen
        img_embeddings =itc_model.encode_image(imgs)
        img_embeddings /= img_embeddings.norm(dim=-1, keepdim=True)
        logits = 100. * img_embeddings @ txt_embeddings.T
        #logits = torch.exp(logits)
        logits /= logits.norm(dim=-1, keepdim=True)

        print(f'pred logits: {logits}')
        #print(f'pred shape: {logits.shape}')
        print(labels)

        # do sum check to ignore samples without a label
        ap_temp = average_precision_score(labels.cpu().numpy(),logits.detach().cpu().numpy())
        #f1 = f1_score(labels.cpu().numpy(),logits.detach().cpu().numpy())
        
        ap_list.append(ap_temp)
        #f1_list.append(f1)

    print(f'map:{np.nanmean(ap_list)}')
 

print('start evaluating')
warnings.filterwarnings("ignore", message="No positive class found in y_true, recall is set to one for all thresholds.")
cls_evaluate(args,val_dataloader,text_tokens,itc_model, device)