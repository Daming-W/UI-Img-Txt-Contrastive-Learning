import clip_modified
import torch
from PIL import Image
import numpy as np
import argparse
import os
from torchsummary import summary
from utils.logger import Logger
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from utils.engine import *
from utils.criterion import *
from data import ClarityDataset,RicoCapDataset,ClaritySepDataset,RicoGPTDataset
from utils.model import getCLIP, getCAM,get_cnn_lstm
from utils.preprocess import getImageTranform


parser = argparse.ArgumentParser()
# path and dir
parser.add_argument("--dataset_name", type=str,
                    default="ricogpt")
parser.add_argument("--img_folder_path", type=str,
                    default="/root/autodl-tmp/RICO/combined/")
parser.add_argument("--csv_file_path", type=str, 
                    default='/root/autodl-tmp/UI_ITC/data/screen_summaries.csv')
parser.add_argument("--model_save_path", type=str, 
                    default='/root/autodl-tmp/')
# device
parser.add_argument("--gpu_id", type=str, default='cuda',
                    help="GPU id to work on, \'cpu\'.")
# model embed_size, hidden_size, vocab_size, num_layers
parser.add_argument("--embed_size", type=int,
                    default=300,help="通常这个超参数的值在200到500之间。更大的值可以让模型捕捉更微妙的单词关系") 
parser.add_argument("--hidden_size", type=int,
                    default=512, help="决定LSTM可以学习的隐藏状态中的特征数量")
parser.add_argument("--vocab_size", type=int,
                    default=1,help="训练数据集中唯一单词的数量") 
parser.add_argument("--lstm_num_layers", type=int,
                    default=3, help="决定网络中的LSTM层数")
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
                    default=16, help="number of workers")   
# hyperparameters
parser.add_argument("--lr", type=float,
                    default=1e-7, help="learning rate")
parser.add_argument("--do_scheduler", type=bool,
                    default=False, help='do scheduler step or not')       
parser.add_argument("--weight_decay", type=float,
                    default=0.001, help="weight_decay")
parser.add_argument("--eps", type=float,
                    default=1e-8, help="eps")
parser.add_argument("--warmup_ratio", type=float, default=0.1,
                    help="accumuate gradients")
parser.add_argument("--epochs", type=int, default=20,
                    help="total number of training epochs")
parser.add_argument("--eval_epochs", type=int, default=2,
                    help="interval of evalutation in training phase")
parser.add_argument("--resume", type=bool, default=False,
                    help="flag to resume training")
args = parser.parse_args()

# get data-preprocess from clip
_,preprocess,preprocess_aug = clip_modified.load("RN50", args.gpu_id, False)

# get image-text dataset
if args.dataset_name == 'clarity':
    args.img_folder_path = "/root/autodl-nas/Clarity-Data/Clarity-PNGs"
    args.csv_file_path = '/root/autodl-nas/Clarity-Data/captions.csv'
    
    if args.img_aug==True:
        train_dataset = ClarityDataset(args,preprocess_aug,'train')
        val_dataset = ClarityDataset(args,preprocess,'val')
    else:    
        train_dataset = ClarityDataset(args,preprocess,'train')   
        val_dataset = ClarityDataset(args,preprocess,'val')

elif args.dataset_name == 'claritysep':
    args.img_folder_path = "/root/autodl-nas/Clarity-Data/Clarity-PNGs"
    args.csv_file_path = '/root/autodl-nas/Clarity-Data/captions_sep.csv'
    
    if args.img_aug==True:
        train_dataset = ClaritySepDataset(args,preprocess_aug,'train')
        val_dataset = ClaritySepDataset(args,preprocess,'val')
    else:    
        train_dataset = ClaritySepDataset(args,preprocess,'train')   
        val_dataset = ClaritySepDataset(args,preprocess,'val')
        
elif args.dataset_name == 'rico':
    args.img_folder_path = '/root/autodl-tmp/RICO/combined/'
    args.csv_file_path = '/root/autodl-tmp/UI_ITC/data/screen_summaries.csv'
    
    if args.img_aug==True:
        train_dataset = RicoCapDataset(args,preprocess_aug,'train')
        val_dataset = RicoCapDataset(args,preprocess,'val')
    else:    
        train_dataset = RicoCapDataset(args,preprocess,'train')
        val_dataset = RicoCapDataset(args,preprocess,'val')

if args.dataset_name == 'ricogpt':
    args.img_folder_path = '/root/autodl-tmp/RICO/combined/'
    args.csv_file_path = '/root/autodl-tmp/UI_ITC/data/RicoGPT-Caption-final.csv'
    
    if args.img_aug==True:
        train_dataset = RicoGPTDataset(args,preprocess_aug,'train')
        val_dataset = RicoGPTDataset(args,preprocess,'val')
    else:    
        train_dataset = RicoGPTDataset(args,preprocess,'train')
        val_dataset = RicoGPTDataset(args,preprocess,'val')

else:
    print('no dataset found!!!')

# set up torch dataloaders
train_dataloader = DataLoader(train_dataset,batch_size=args.batch_size,num_workers=args.num_workers,shuffle=True)
val_dataloader = DataLoader(val_dataset,batch_size=args.batch_size,num_workers=args.num_workers,shuffle=True)

print(f'dataset selected : {args.dataset_name} and do img aug:{args.img_aug}')
if args.dataset_name == 'clarity': print(f'using clarity with {args.num_captions} captions')
print(f"finish loading data: train:{len(train_dataloader)*args.batch_size} val:{len(val_dataloader)*args.batch_size}")

# get image captioning model -- CNN-LSTM
model = get_cnn_lstm(args) 
model = model.cuda()
print("finish making a model")

# load checkpoints for resuming
if args.resume == True:
    checkpoints_path = '/root/autodl-tmp/UI_ITC/checkpoints_uicap/ckp_ricogpt_rn50_0415_epoch20.pt'
    model.load_state_dict(torch.load(checkpoints_path))
    resumed_epoch = str(checkpoints_path.split('/')[-1].split('_')[-1][:-3].split('h')[-1])
    print(f'resume training on {resumed_epoch}')

# setup criterion
criterion=torch.nn.CrossEntropyLoss().to('cuda')

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
    eta_min=1e-8)

print(f"finish setting a scheduler with warmup-ratio : {args.warmup_ratio}")

# setup logger
logger_name = 'ui_caption_ricogpt'
if args.resume == True:
    logger = Logger('/root/autodl-tmp/UI_ITC/logs_uicap/'+logger_name+'.log',True)
else:
    logger = Logger('/root/autodl-tmp/UI_ITC/logs_uicap/'+logger_name+'.log',False)
    logger.append(args)
print('finish setting logger')

# train and eval
print('start training\n')
for epoch in range(args.epochs):
    if args.resume == True:
        print(f'Current EPOCH : {epoch+1+int(resumed_epoch)}')
        logger.append(f'epoch : {epoch+1+int(resumed_epoch)}')
    else:
        print(f'Current EPOCH : {epoch+1}')
        logger.append(f'epoch : {epoch+1}')
    
    train_uicap(args, train_dataloader, model, criterion, optimizer, lr_scheduler_cosann, logger)
    evaluate_uicap(args, val_dataloader, model, criterion, optimizer, logger)

    # save ckp
    if (epoch+1)%5==0:
        if args.resume == True:
            torch.save(model.state_dict(),f'/root/autodl-tmp/UI_ITC/checkpoints_uicap/ckp_{logger_name}_epoch{epoch+1+int(resumed_epoch)}.pt')
            print(f"save dict for epoch : {epoch+1+int(resumed_epoch)} ")
        else:
            torch.save(model.state_dict(),f'/root/autodl-tmp/UI_ITC/checkpoints_uicap/ckp_{logger_name}_epoch{epoch+1}.pt')
            print(f"save dict for epoch : {epoch+1} ")