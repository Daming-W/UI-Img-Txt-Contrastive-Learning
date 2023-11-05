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

def train_epoch(args, dataloader, model, criterion, optimizer, lr_scheduler, logger):

    total_loss = []
    model.train()

    with tqdm(total=len(dataloader)) as pbar: 
       for i, (imgs, txts) in enumerate(dataloader):
            # load to device
            imgs = imgs.cuda(non_blocking=True)
            txts = txts.cuda(non_blocking=True)
            optimizer.zero_grad()
            # get img-txt embeddings
            #print(f"img and txt shape: {imgs.shape} ; {txts.shape}")
            with torch.cuda.amp.autocast(enabled=True):
                logits_per_img, logits_per_txt = model(imgs, txts)
            # compute itc loss
            ground_truth = torch.arange(len(imgs),dtype=torch.long,device=args.gpu_id)
            #print(f'shapes: {logits_per_img.shape} ; {logits_per_txt.shape} ; {ground_truth.shape}')
            loss = (criterion(logits_per_img, ground_truth)+criterion(logits_per_txt, ground_truth))/2
            # loss backward
            loss.backward()
            # fp32
            convert_models_to_fp32(model)
            # optimizer and scheduler step
            optimizer.step()
            #clip_modified.model.convert_weights(model)
            if args.do_scheduler==True:
                lr_scheduler.step()
            # sum losses in an epoch
            total_loss.append(loss.detach().cpu())
            
            pbar.set_description('training')
            pbar.set_postfix({'loss(iter)':float(loss.detach().cpu()), 
                              'loss(mean)':np.mean(total_loss),
                              'lr':lr_scheduler.get_last_lr()})
            pbar.update(1)

    epoch_loss = np.mean(total_loss)
    #print(f'train_epoch_loss: {epoch_loss}')  
    logger.append(f'train_epoch_loss: {epoch_loss}')
    logger.append(f'train_lr : {lr_scheduler.get_last_lr()[0]}')
     

def evaluate(args, dataloader, model, criterion, optimizer, logger):

    total_loss = []
    model.eval()
    
    with tqdm(total=len(dataloader)) as pbar:
        for i, (imgs, txts) in enumerate(dataloader):
            # load to device
            imgs = imgs.cuda(non_blocking=True)
            txts = txts.cuda(non_blocking=True)
            # get img-txt embeddings    
            with torch.no_grad():
                logits_per_img, logits_per_txt = model(imgs, txts)
            # compute itc loss
            ground_truth = torch.arange(len(imgs),dtype=torch.long,device=args.gpu_id)
            img_loss,txt_loss = torch.nn.CrossEntropyLoss(),torch.nn.CrossEntropyLoss()
            loss = (img_loss(logits_per_img, ground_truth)+txt_loss(logits_per_txt, ground_truth))/2
            # sum losses in a epoch
            total_loss.append(loss.detach().cpu())
            pbar.set_description('evaluation')
            pbar.set_postfix({'loss(iter)':float(loss.detach().cpu()), 'loss(mean)':np.mean(total_loss)})
            pbar.update(1)

    epoch_loss = np.nanmean(total_loss)
    #print(f'evalutation_epoch_loss: {epoch_loss}') 
    logger.append(f'evalutation_epoch_loss: {epoch_loss}')

def train_uicap(args, dataloader, model, criterion, optimizer, lr_scheduler, logger):
    total_loss=[]
    model.train()
    with tqdm(total=len(dataloader)) as pbar:
        for i, (images, captions) in enumerate(dataloader):
            # load to device
            images = images.cuda(non_blocking=True)
            captions = captions.cuda(non_blocking=True)
            optimizer.zero_grad()
            # model computation
            #with torch.cuda.amp.autocast(enabled=True):
            outputs = model(images, captions[:,:-1])
            # compute cross entroppy loss
            print(outputs.flatten().shape)
            print(captions[:, 1:].flatten().shape)
            loss = criterion(outputs.flatten(), captions[:, 1:].flatten())
            loss = torch.tensor(loss, dtype=torch.float, requires_grad=True)
            # backward loss and step
            loss.backward()
            optimizer.step()
            # update lr scheduler:
            if args.do_scheduler==True:
                lr_scheduler.step()
            # sum losses in a epoch
            total_loss.append(loss.detach().cpu())
            pbar.set_description('train')
            pbar.set_postfix({'loss(iter)':float(loss.detach().cpu()), 'loss(mean)':np.mean(total_loss)})
            pbar.update(1)
        epoch_loss = np.mean(total_loss)

    #print(f'train_epoch_loss: {epoch_loss}')  
    logger.append(f'train_epoch_loss: {epoch_loss}')
    logger.append(f'train_lr : {lr_scheduler.get_last_lr()[0]}')

def evaluate_uicap(args, dataloader, model, criterion, optimizer, logger):

    total_loss = []
    model.eval()
    
    with tqdm(total=len(dataloader)) as pbar:
        for i, (images, captions) in enumerate(dataloader):
            # load to device
            images = images.cuda(non_blocking=True)
            captions = captions.cuda(non_blocking=True)
            # model computation
            with torch.no_grad():
                outputs = model(images, captions[:, :-1])
            # compute cross entroppy loss
            loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions[:, 1:].reshape(-1))
            # sum losses in a epoch
            total_loss.append(loss.detach().cpu())
            pbar.set_description('evaluation')
            pbar.set_postfix({'loss(iter)':float(loss.detach().cpu()), 'loss(mean)':np.mean(total_loss)})
            pbar.update(1)

    epoch_loss = np.nanmean(total_loss)
    #print(f'evalutation_epoch_loss: {epoch_loss}') 
    logger.append(f'evalutation_epoch_loss: {epoch_loss}')