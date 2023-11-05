import clip_modified
import torch
from PIL import Image
import numpy as np
import argparse
import os

from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from data import ClarityDataset
from utils.model import getCLIP, getCAM
from utils.preprocess import getImageTranform
from utils.engine import *
from utils.criterion import *

parser = argparse.ArgumentParser()

# path and dir
parser.add_argument("--img_folder_path", type=str,
                    default="/root/autodl-nas/Clarity-Data/Clarity-PNGs")
parser.add_argument("--csv_file_path", type=str, 
                    default='/root/autodl-nas/Clarity-Data/captions.csv')
parser.add_argument("--model_save_path", type=str, 
                    default='/root/autodl-tmp/')
# device
parser.add_argument("--gpu_id", type=str, default='cuda',
                    help="GPU id to work on, \'cpu\'.")
# model
parser.add_argument("--clip_model_name", type=str,
                    default='RN50', help="Model name of CLIP")
parser.add_argument("--cam_model_name", type=str,
                    default='GradCAM', help="Model name of GradCAM")
# hyperparameters
parser.add_argument("--train_val_ratio", type=float,
                    default=0.7, help="train set and validation set dataset ratio")
parser.add_argument("--resize", type=int,
                    default=1, help="Resize image or not")
parser.add_argument("--batch_size", type=int,
                    default=2, help="batch size of data")
parser.add_argument("--lr", type=float,
                    default=5e-5, help="learning rate")
parser.add_argument("--weight_decay", type=float,
                    default=0.3, help="weight_decay")
parser.add_argument("--eps", type=float,
                    default=1e-6, help="eps")
parser.add_argument("--accumulation_steps", type=int, default=4,
                    help="accumuate gradients")
parser.add_argument("--epochs", type=int, default=10,
                    help="total number of training epochs")
parser.add_argument("--eval_epochs", type=int, default=2,
                    help="interval of evalutation in training phase")
# attack
parser.add_argument("--attack_type", type=str, default=None,
                    help="attack type: \"snow\", \"fog\"")
# text input
parser.add_argument("--sentence", type=str, default='',
                    help="input text")
args = parser.parse_args()

# get CLIP model
#model, _, _ = getCLIP(model_name=args.clip_model_name, gpu_id=args.gpu_id)
model, preprocess = clip_modified.load(args.clip_model_name, device = args.gpu_id, jit = False)
model = model.cuda()
clip_modified.model.convert_weights(model)

# get image-text dataset
dataset = ClarityDataset(args,preprocess)

# split train and valid by setting indice
indices = list(range(len(dataset)))
train_indices = indices[int(len(dataset)*args.train_val_ratio):]
valid_indices = indices[:int(len(dataset)*args.train_val_ratio)]

train_dataloader = DataLoader(dataset, batch_size=args.batch_size,
                  sampler=SubsetRandomSampler(train_indices))
val_dataloader = DataLoader(dataset, batch_size=args.batch_size,
                  sampler=SubsetRandomSampler(valid_indices))
print(f"finish loading data: train:{len(train_dataloader)*args.batch_size} val:{len(val_dataloader)*args.batch_size}")

# get optimizer
exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
include = lambda n, p: not exclude(n, p)

named_parameters = list(model.named_parameters())
# filter out the frozen parameters
gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]
optimizer = torch.optim.Adam(
    [{"params": gain_or_bias_params, "weight_decay": 0.0},
    {"params": rest_params, "weight_decay": args.weight_decay}],
    lr=args.lr, eps=args.eps)

print("finish setting optimizer!")

# setup learning rate scheduler
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=5, 
    num_training_steps=int(len(train_dataloader) * args.epochs/args.accumulation_steps), 
    num_cycles=0.5)

# setup criterion
criterion=torch.nn.CrossEntropyLoss().to('cuda')

# train and eval
for epoch in range(args.epochs):
    print(f'start training at epoch : {epoch+1}')
    train_epoch(args, train_dataloader, model, criterion, optimizer, lr_scheduler)
    print(f'start evaluating at epoch : {epoch+1}')
    evaluate(args, val_dataloader, model, criterion, optimizer)

'''
    if epoch % args.eval_epochs==0:
        evaluate()
        model_save_path = os.path.join(args.model_save_path,f'epoch{epoch}.pt')
        torch.save(model.state_dict(), model_save_path)
        print(f'save at {model_save_path}')
'''
