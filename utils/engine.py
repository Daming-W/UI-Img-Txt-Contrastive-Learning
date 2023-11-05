import clip_modified
import torch
from PIL import Image
import numpy as np
import argparse
import os
from tqdm import tqdm
import time
import torch.nn.functional as F
from transformers import get_cosine_schedule_with_warmup

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()

def train_epoch(args, dataloader, model, criterion, optimizer, lr_scheduler):

    total_loss = 0
    model.train()

    with tqdm(total=len(dataloader)) as pbar:
        for i, (imgs, txts) in enumerate(dataloader):
            # load to device
            imgs = imgs.cuda(non_blocking=True)
            txts = txts.cuda(non_blocking=True)

            # get img-txt embeddings
            #print(f"img and txt shape: {imgs.shape} ; {txts.shape}")
            with torch.cuda.amp.autocast(enabled=True):
                logits_per_img, logits_per_txt = model(imgs, txts)
            # compute itc loss
            ground_truth = torch.arange(len(imgs),dtype=torch.long,device=args.gpu_id)
            #print(f'shapes: {logits_per_img.shape} ; {logits_per_txt.shape} ; {ground_truth.shape}')
            loss = (criterion(logits_per_img, ground_truth)+criterion(logits_per_txt, ground_truth))/2

            optimizer.zero_grad()
            # loss backward
            loss.mean().backward()
            # fp32
            convert_models_to_fp32(model)
            # optimizer and scheduler step
            optimizer.step()
            clip_modified.model.convert_weights(model)
            lr_scheduler.step()

            # sum losses in a epoch
            total_loss += loss

            pbar.set_description('training')
            pbar.set_postfix({'loss' : '{0:1.5f}'.format(loss.detach().cpu())})
            time.sleep(0.05)
            pbar.update(1)

    epoch_loss = total_loss/len(dataloader)
    print(f'epoch_loss: {epoch_loss}')    


def evaluate(args, dataloader, model, criterion, optimizer):

    total_loss = 0
    model.eval()
    
    with tqdm(total=len(dataloader)) as pbar:
        for i, (imgs, txts) in enumerate(tqdm(dataloader)):
            # load to device
            imgs = imgs.cuda(non_blocking=True)
            txts = txts.cuda(non_blocking=True)
            # get img-txt embeddings
            with torch.cuda.amp.autocast(enabled=True):
                logits_per_img, logits_per_txt = model(imgs, txts)
            # compute itc loss
            ground_truth = torch.arange(len(imgs),dtype=torch.long,device=args.gpu_id)
            loss = (criterion(logits_per_img, ground_truth)+criterion(logits_per_txt, ground_truth))/2
            # sum losses in a epoch
            total_loss += loss
            
            pbar.set_description('evaluation')
            pbar.set_postfix({'loss' : '{0:1.5f}'.format(loss.detach().cpu())})
            time.sleep(0.05)
            pbar.update(1)

    epoch_loss = total_loss/len(dataloader)
    print(epoch_loss)    