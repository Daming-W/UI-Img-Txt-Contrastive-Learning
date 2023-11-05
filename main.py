import clip_modified
import torch
from PIL import Image
import numpy as np
import argparse
import os
from torchsummary import summary
from utils.logger import Logger
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, ConcatDataset
from utils.engine import *
from utils.criterion import *
from data import ClarityDataset,RicoCapDataset,ClaritySepDataset,RicoGPTDataset,MixDataset
from utils.model import getCLIP, getCAM
from utils.preprocess import getImageTranform

parser = argparse.ArgumentParser()
# path and dir
parser.add_argument("--dataset_name", type=str,
                    default="screen2words", help="clarity,claritysep,screen2words,ricogpt,mix")
parser.add_argument("--img_folder_path", type=str,
                    default="/root/autodl-tmp/RICO/combined/")
parser.add_argument("--csv_file_path", type=str, 
                    default='/root/autodl-tmp/UI_ITC/data/screen_summaries.csv')
parser.add_argument("--model_save_path", type=str, 
                    default='/root/autodl-tmp/')
# device
parser.add_argument("--gpu_id", type=str, default='cuda',
                    help="GPU id to work on, \'cpu\'.")
# model # ViT-B/32
parser.add_argument("--clip_model_name", type=str,
                    default='ViT-B/16',help="Model name of CLIP") 
parser.add_argument("--cam_model_name", type=str,
                    default='GradCAM', help="Model name of GradCAM")
# data
parser.add_argument("--train_val_ratio", type=float,
                    default=0.75, help="train set and val set dataset ratio")
parser.add_argument("--resize", type=int,
                    default=1, help="Resize image or not")
parser.add_argument("--batch_size", type=int,
                    default=32, help="batch size of data")
parser.add_argument("--num_captions", type=int,
                    default=5, help="captions number per one image")
parser.add_argument("--img_aug", type=bool,
                    default=True, help="do image augmentation or not")             
parser.add_argument("--num_workers", type=int,
                    default=18, help="number of workers")   
# hyperparameters
parser.add_argument("--lr", type=float,
                    default=5e-7, help="learning rate")
parser.add_argument("--do_scheduler", type=bool,
                    default=True, help='do scheduler step or not')       
parser.add_argument("--weight_decay", type=float,
                    default=0.001, help="weight_decay")
parser.add_argument("--eps", type=float,
                    default=1e-8, help="eps")
parser.add_argument("--warmup_ratio", type=float, default=0.1,
                    help="accumuate gradients")
parser.add_argument("--epochs", type=int, default=40,
                    help="total number of training epochs")
parser.add_argument("--eval_epochs", type=int, default=2,
                    help="interval of evalutation in training phase")
parser.add_argument("--resume", type=bool, default=False,
                    help="flag to resume training")
# attack
parser.add_argument("--attack_type", type=str, default=None,
                    help="attack type: \"snow\", \"fog\"")
# text input
parser.add_argument("--sentence", type=str, default='',
                    help="input text")
args = parser.parse_args()

# get CLIP model
#model, _, _ = getCLIP(model_name=args.clip_model_name, gpu_id=args.gpu_id)
model, preprocess, preprocess_aug= clip_modified.load(args.clip_model_name, device = args.gpu_id, jit = False)
model = model.cuda()
print(f'finish loading model (with visual backbone: {args.clip_model_name})')

# load checkpoints for resuming
if args.resume == True:
    checkpoints_path = '/root/autodl-tmp/UI_ITC/checkpoints/ckp_ricogpt_rn50_0415_epoch20.pt'
    model.load_state_dict(torch.load(checkpoints_path))
    resumed_epoch = str(checkpoints_path.split('/')[-1].split('_')[-1][:-3].split('h')[-1])
    print(f'resume training on {resumed_epoch}')

# get image-text dataset
# clarity
if args.dataset_name == 'clarity':
    args.img_folder_path = "/root/autodl-nas/Clarity-Data/Clarity-PNGs"
    args.csv_file_path = '/root/autodl-nas/Clarity-Data/captions.csv'
    
    if args.img_aug==True:
        train_dataset = ClarityDataset(args,preprocess_aug,'train')
        val_dataset = ClarityDataset(args,preprocess,'val')
    else:    
        train_dataset = ClarityDataset(args,preprocess,'train')   
        val_dataset = ClarityDataset(args,preprocess,'val')
# clarity sep low high
elif args.dataset_name == 'claritysep':
    args.img_folder_path = "/root/autodl-nas/Clarity-Data/Clarity-PNGs"
    args.csv_file_path = '/root/autodl-nas/z-Data/captions_sep.csv'
    
    if args.img_aug==True:
        train_dataset = ClaritySepDataset(args,preprocess_aug,'train')
        val_dataset = ClaritySepDataset(args,preprocess,'val')
    else:    
        train_dataset = ClaritySepDataset(args,preprocess,'train')   
        val_dataset = ClaritySepDataset(args,preprocess,'val')
# screen2words
elif args.dataset_name == 'screen2words':
    args.img_folder_path = '/root/autodl-tmp/RICO/combined/'
    args.csv_file_path = '/root/autodl-tmp/UI_ITC/data/screen_summaries.csv'
    
    if args.img_aug==True:
        train_dataset = RicoCapDataset(args,preprocess_aug,'train')
        val_dataset = RicoCapDataset(args,preprocess,'val')
    else:    
        train_dataset = RicoCapDataset(args,preprocess,'train')
        val_dataset = RicoCapDataset(args,preprocess,'val')
# ricogpt
elif args.dataset_name == 'ricogpt':
    args.img_folder_path = '/root/autodl-tmp/RICO/combined/'
    args.csv_file_path = '/root/autodl-tmp/UI_ITC/data/RicoGPT-Caption-final.csv'
    
    if args.img_aug==True:
        train_dataset = RicoGPTDataset(args,preprocess_aug,'train')
        val_dataset = RicoGPTDataset(args,preprocess,'val')
    else:    
        train_dataset = RicoGPTDataset(args,preprocess,'train')
        val_dataset = RicoGPTDataset(args,preprocess,'val')
# clarity-sep + screen2words + ricogpt
elif args.dataset_name == 'mix':
    args.csv_file_path = '/root/autodl-tmp/UI_ITC/data/mix.csv'

    train_dataset = MixDataset(args,preprocess_aug,'train')
    val_dataset = MixDataset(args,preprocess,'val')
# non existing
else:
    print('no dataset found!!!')

# convert dataset to dataloader
train_dataloader = DataLoader(train_dataset,batch_size=args.batch_size,num_workers=args.num_workers,shuffle=True)
val_dataloader = DataLoader(val_dataset,batch_size=args.batch_size,num_workers=args.num_workers,shuffle=True)

print(f'dataset selected : {args.dataset_name} and do img aug:{args.img_aug}')
if args.dataset_name == 'clarity': print(f'using clarity with {args.num_captions} captions')
print(f"finish loading data: train:{len(train_dataloader)*args.batch_size} val:{len(val_dataloader)*args.batch_size}")

# get optimizer
exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or 'logit_scale' in n
include = lambda n, p: not exclude(n, p)
named_parameters = list(model.named_parameters())
# filter out the frozen parameters
gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]
optimizer = torch.optim.AdamW(
    [{"params": gain_or_bias_params, "weight_decay": args.weight_decay},
    {"params": rest_params, "weight_decay": args.weight_decay}],
    lr=args.lr, eps=args.eps)
'''
optimizer = torch.optim.SGD( 
    [{"params": gain_or_bias_params, "weight_decay": args.weight_decay},
    {"params": rest_params, "weight_decay": args.weight_decay}],
    lr=args.lr,momentum=0.2)
'''
print("finish setting an optimizer")

# setup learning rate scheduler
lr_scheduler_cos = get_cosine_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=int(len(train_dataloader)) * args.epochs * args.warmup_ratio, 
    num_training_steps=int(len(train_dataloader) * args.epochs), 
    num_cycles=0.5)

lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
    optimizer, 
    0.9)

lr_scheduler_cosann = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=int(len(train_dataloader) * args.epochs),
    eta_min=5e-8)

print(f"finish setting a scheduler with warmup-ratio : {args.warmup_ratio}")

# setup criterion
criterion=torch.nn.CrossEntropyLoss().to('cuda')

# setup logger
logger_name = 'test'
if args.resume == True:
    logger = Logger('/root/autodl-tmp/UI_ITC/logs/'+logger_name+'.log',True)
else:
    logger = Logger('/root/autodl-tmp/UI_ITC/logs/'+logger_name+'.log',False)
    logger.append(args)
print('finish setting logger')

# train and eval
print('#start training#\n')
for epoch in range(args.epochs):
    if args.resume == True:
        print(f'Current EPOCH : {epoch+1+int(resumed_epoch)}')
        logger.append(f'epoch : {epoch+1+int(resumed_epoch)}')
    else:
        print(f'Current EPOCH : {epoch+1}')
        logger.append(f'epoch : {epoch+1}')
    
    train_epoch(args, train_dataloader, model, criterion, optimizer, lr_scheduler_cosann, logger)
    evaluate(args, val_dataloader, model, criterion, optimizer, logger)

    # save ckp
    if (epoch+1)%5==0:
        if args.resume == True:
            torch.save(model.state_dict(),f'/root/autodl-tmp/UI_ITC/checkpoints/ckp_{logger_name}_epoch{epoch+1+int(resumed_epoch)}.pt')
            print(f"save dict for epoch : {epoch+1+int(resumed_epoch)} ")
        else:
            torch.save(model.state_dict(),f'/root/autodl-tmp/UI_ITC/checkpoints/ckp_{logger_name}_epoch{epoch+1}.pt')
            print(f"save dict for epoch : {epoch+1} ")