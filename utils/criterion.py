import torch
import numpy as np
import argparse
import os
import torch.nn as nn

def cross_entropy(preds, targets, reduction='none'):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (-targets * log_softmax(preds)).sum(1)
    if reduction == "none":
        return loss
    elif reduction == "mean":
        return loss.mean()

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.2):
        super().__init__()
        self.margin = margin
    
    def forward(self, image_features, text_features, labels):
        # Compute the cosine similarity between image and text features
        similarities = F.cosine_similarity(image_features, text_features)
        
        # Compute the contrastive loss
        positive_similarities = similarities[labels == 1]
        negative_similarities = similarities[labels == 0]
        loss = torch.mean(torch.clamp(self.margin - positive_similarities + negative_similarities, min=0))
        return loss