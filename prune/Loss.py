"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, single_pos=False):
        super(SupervisedContrastiveLoss, self).__init__()
        self.temperature = temperature
        try:
            self.single_pos = single_pos
        except:
            self.single_pos = False
        
        self.sim = nn.CosineSimilarity(dim=1)
        
    def forward(self, model, num_positive, num_negative, contrastive_batch):
        # Compute negative similarities
        neg_indices = [0] + list(range(len(contrastive_batch))[-num_negative:])
        anchor_negatives = contrastive_batch[neg_indices]
        exp_neg = self.compute_exp_sim(model, anchor_negatives,
                                       return_sum=False)

        sum_exp_neg = exp_neg.sum(0, keepdim=True)
        #print('sum_exp_neg', sum_exp_neg)
        # Compute positive similarities
        anchor_positives = contrastive_batch[:1 + num_positive]
        exp_pos = self.compute_exp_sim(model, anchor_positives, 
                                       return_sum=False)
        
        if self.single_pos:
            log_probs = torch.log(exp_pos) - torch.log(sum_exp_neg + exp_pos)
        else:
            log_probs = (torch.log(exp_pos) - 
                         torch.log(sum_exp_neg + exp_pos.sum(0, keepdim=True)))
        loss = -1 * log_probs
        del exp_pos; del exp_neg; del log_probs
        return loss.mean()
    
    def compute_exp_sim(self, model, features, return_sum=True):
        """
        model: encoder (model.classifier(x, feature=True))
        feature: input batch
        Compute sum(sim(anchor, pos)) or sum(sim(anchor, neg))
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        features = features.to(device)
        model.classifier.pruning_switch(True)
            #outputs = model(features)
        pred, outputs = model.classifier(features, feature=True)
        sim = self.sim(outputs[0].view(1, -1), outputs[1:])
        #print('similarity', sim)
        exp_sim = torch.exp(torch.div(sim, self.temperature))
        # Should not detach from graph
        features = features.to(torch.device('cpu'))
        outputs = outputs.to(torch.device('cpu'))
        if return_sum:
            sum_exp_sim = exp_sim.sum(0, keepdim=True)
            exp_sim.detach_().cpu(); del exp_sim
            return sum_exp_sim
        return exp_sim

