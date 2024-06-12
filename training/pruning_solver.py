import os
from os.path import join as ospj
import time
import datetime
from munch import Munch
import logging

import numpy as np
import torch
from torch.utils import data
import torch.nn.functional as F
import torch.nn as nn

import util.utils as utils
from data.data_loader import InputFetcher
from data.data_loader import get_original_loader, get_val_loader, get_contrast_loader
from model.build_models import build_model
from training.solver import Solver
from prune.Loss import SupervisedContrastiveLoss


class PruneSolver(Solver):
    def __init__(self, args):
        super(PruneSolver, self).__init__(args)

        self.optims_main = Munch() # Used in retraining
        self.optims_mask = Munch() # Used in learning pruning parameter

        for net, m in self.nets.items():
            prune_param = [p for n,p in m.named_parameters() if 'gumbel_pi' in n]
            main_param = [p for n,p in m.named_parameters() if 'gumbel_pi' not in n]

            if args.optimizer == 'Adam':
                self.optims_main[net] = torch.optim.Adam(
                    params=main_param, #self.nets[net].parameters(),
                    lr=args.lr_main,
                    betas=(args.beta1, args.beta2),
                    weight_decay=0
                )
            elif args.optimizer == 'SGD':
                self.optims_main[net] = torch.optim.SGD(
                    main_param,
                    lr=args.lr_main,
                    momentum=0.9,
                    weight_decay=args.weight_decay
                )

            self.optims_mask[net] = torch.optim.Adam(
                prune_param,
                lr=args.lr_prune,
            )

        self.scheduler_main = Munch() # Used in retraining
        if not args.no_lr_scheduling:
            for net in self.nets.keys():
                self.scheduler_main[net] = torch.optim.lr_scheduler.StepLR(
                    self.optims_main[net], step_size=args.lr_decay_step_main, gamma=args.lr_gamma_main)

        self.con_prune_criterion = SupervisedContrastiveLoss()

    def sparsity_regularizer(self, token='gumbel_pi'):
        reg = 0.
        for n, p in self.nets.classifier.named_parameters():
            if token in n:
                reg = reg + p.sum()
        return reg

    def save_wrong_idx(self, loader): ##Use for DCWP
        self.nets.classifier.eval()
        self.nets.biased_classifier.eval()

        iterator = enumerate(loader)
        total_wrong, total_num = 0, 0
        wrong_idx = torch.empty(0).to(self.device)
        debias_idx = torch.empty(0).to(self.device)
        fname_full = []

        for _, (idx, data, attr, fname) in iterator:
            idx = idx.to(self.device)
            label = attr[:, 0].to(self.device)
            bias_label = attr[:, 1].to(self.device)
            data = data.to(self.device)

            with torch.no_grad():
                if self.args.select_with_GCE or self.args.data == 'coloredmnist' or self.args.data == 'isic':
                    logit = self.nets.biased_classifier(data)
                else:
                    logit = self.nets.classifier(data)

                pred = logit.data.max(1, keepdim=True)[1].squeeze(1)
                wrong = (pred != label).long()

                if self.args.data != 'coloredmnist' and self.args.data != 'isic':
                    debiased = (label != bias_label).long()
                else:
                    debiased = (label == bias_label).long()

                total_wrong += wrong.sum()
                total_num += wrong.shape[0]
                wrong_idx = torch.cat((wrong_idx, idx[wrong == 1])).long()
                debias_idx = torch.cat((debias_idx, idx[debiased == 1])).long()

            fname_full.append(fname)

        assert total_wrong == len(wrong_idx)
        print('Number of wrong samples: ', total_wrong)
        self.confirm_pseudo_label(wrong_idx, debias_idx, total_num)

    def confirm_pseudo_label(self, wrong_idx, debias_idx, total_num):
        wrong_label = torch.zeros(total_num).to(self.device)
        debias_label = torch.zeros(total_num).to(self.device)

        for idx in wrong_idx:
            wrong_label[idx] = 1
        for idx in debias_idx:
            debias_label[idx] = 1

        spur_precision = torch.sum(
                (wrong_label == 1) & (debias_label == 1)
            ) / torch.sum(wrong_label)
        print("Spurious precision", spur_precision)
        spur_recall = torch.sum(
                (wrong_label == 1) & (debias_label == 1)
            ) / torch.sum(debias_label)
        print("Spurious recall", spur_recall)

        wrong_idx_path = ospj(self.args.checkpoint_dir, 'wrong_index.pth')

        if not self.args.supervised:
            torch.save(wrong_label, wrong_idx_path)
        else:
            torch.save(debias_label, wrong_idx_path)
        print('Saved wrong index label.')
        self.nets.classifier.train()
        self.nets.biased_classifier.train()

    def train_PRUNE(self, epoch):
        args = self.args
        nets = self.nets
        optims = self.optims_mask # Train only pruning parameter
        #upweight = torch.ones_like(wrong_label)
        #upweight[wrong_label == 1] = 80

        #sampling_weight = upweight if not args.uniform_weight else torch.ones_like(wrong_label)
        anchor_list, balanced_loader = get_contrast_loader(return_dataset=False)

        #fetcher = InputFetcher(balanced_loader)
        fetcher_val = self.loaders.val
        start_time = time.time()

        self.nets.classifier.pruning_switch(True)

        for j in range(epoch):
            for batch in balanced_loader:
                index, x, label, cluster = batch
                x = x.to(self.device)
                label = label.to(self.device)
                cluster = cluster.to(self.device)
                #bias_label = torch.index_select(wrong_label, 0, idx.long())

                pred, feature = self.nets.classifier(x, feature=True)
                loss_main = self.criterion(pred, label).mean()
                loss_reg = self.sparsity_regularizer()
                loss_contrast = SupervisedContrastiveLoss()
                loss_con_batch = 0

                for i in range(len(index)):
                    anchor_idx = i
                    anchor_cluster = cluster[anchor_idx]
                    anchor_label = label[anchor_idx]

                    neg_idx = (cluster == anchor_cluster).nonzero().squeeze()
                    neg_y = (cluster == anchor_cluster).nonzero().squeeze()
                    similar_mask_neg = torch.eq(neg_idx.unsqueeze(1), neg_y.unsqueeze(0))
                    similar_items_neg = neg_idx[similar_mask_neg.any(dim=1)]
                    neg_in_batch = x[similar_items_neg]

                    pos_idx = (cluster != anchor_cluster).nonzero().squeeze()
                    pos_y = (label == anchor_label).nonzero().squeeze()
                    similar_mask_pos = torch.eq(pos_idx.unsqueeze(1), pos_y.unsqueeze(0))
                    similar_items_pos = pos_idx[similar_mask_pos.any(dim=1)]
                    pos_in_batch = torch.vstack((x[i].unsqueeze(0), x[similar_items_pos]))

                    contrastive_batch = torch.vstack((x[i].unsqueeze(0), neg_in_batch, pos_in_batch))
                    con_l = loss_contrast(nets, len(similar_items_pos)+1, len(similar_items_neg), contrastive_batch)
                    loss_con_batch += con_l.item()


                loss = loss_main + args.lambda_sparse * loss_reg + loss_con_batch * args.lambda_con_prune

                self._reset_grad()
                loss.backward()
                optims.classifier.step()

            # print out log info
            #if (i+1) % args.print_every == 0:
            elapsed = time.time() - start_time
            elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
            log = "Elapsed time [%s], LR [%.4f], "\
                    "Loss_main [%.6f] Loss_reg [%.6f] Loss_con [%.6f]" % (elapsed,
                                                                          optims.classifier.param_groups[-1]['lr'],
                                                                          loss_main.item(),
                                                                          loss_reg.item(),
                                                                          loss_con_batch)
            print(log)

            #if (i+1) % args.eval_every == 0:
            total = 0
            active = 0
            layerwise = {}
            for n, p in self.nets.classifier.named_parameters():
                if 'gumbel_pi' in n:
                    active_n = (p>=0).sum().item()
                    total_n = torch.ones_like(p).sum().detach().item()
                    layerwise[n] = active_n / total_n

                    total += total_n
                    active += active_n
                    if active_n == 0: print('Warning: Dead layer')

            ratio = active / total
            print('ratio:', ratio)
            print('layerwise', layerwise)
            logging.info(layerwise)
            self.valid_logger.append(ratio, which='ratio')
            self.valid_logger.append(layerwise, which='layerwise_ratio')

            self.nets.classifier.pruning_switch(False)
            self.nets.classifier.freeze_switch(True)
            total_acc, valid_attrwise_acc = self.validation(fetcher_val)
            print('pruning', valid_attrwise_acc)
            logging.info(valid_attrwise_acc)
            self.report_validation(valid_attrwise_acc, total_acc, i, which='prune')
            self.nets.classifier.pruning_switch(True)
            self.nets.classifier.freeze_switch(False)

            # save model checkpoints
            self._save_checkpoint(step=j+1, token='prune')

    def retrain(self, iters, freeze=True):
        args = self.args
        nets = self.nets
        optims = self.optims_main  # Train only weight parameter

        wrong_label = torch.load(ospj(self.args.checkpoint_dir, 'wrong_index.pth'))
        print('Number of wrong samples: ', wrong_label.sum())
        upweight = torch.ones_like(wrong_label)
        upweight[wrong_label == 1] = args.lambda_upweight

        upweight_loader = get_val_loader(args, split='val')

        # upweight_loader = get_original_loader(args, sampling_weight=upweight)

        fetcher = InputFetcher(upweight_loader)
        fetcher_val = self.loaders.val
        start_time = time.time()

        self.nets.classifier.pruning_switch(False)
        self.nets.classifier.freeze_switch(freeze)

        for i in range(iters):
            inputs = next(fetcher)
            idx, x, label, fname = inputs.index, inputs.x, inputs.y, inputs.fname
            bias_label = torch.index_select(wrong_label, 0, idx.long())

            pred, feature = self.nets.classifier(x, feature=True)
            loss_main = self.criterion(pred, label).mean()  
            loss = loss_main

            self._reset_grad()
            loss.backward()
            optims.classifier.step()

            # print out log info
            if (i + 1) % args.print_every == 0:
                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
                log = "Elapsed time [%s], Iteration [%i/%i], LR [%.4f], " \
                      "Loss_main [%.6f]" % (elapsed, i + 1, iters,
                                                             optims.classifier.param_groups[-1]['lr'],
                                                             loss_main.item())
                print(log)

            # save model checkpoints
            if (i + 1) % args.save_every_retrain == 0:
                self._save_checkpoint(step=i + 1, token='retrain_val')

            if (i + 1) % args.eval_every_retrain == 0:
                total_acc, valid_attrwise_acc = self.validation(fetcher_val)
                print('retrain', valid_attrwise_acc)
                logging.info(valid_attrwise_acc)
                print('average acc', total_acc)
                self.report_validation(valid_attrwise_acc, total_acc, i, which='retrain')
                self.valid_logger.append(total_acc.item(), which='retrain')
                self.valid_logger.append(valid_attrwise_acc, which='groupwise_acc')

            if not self.args.no_lr_scheduling:
                self.scheduler_main.classifier.step()

    def train(self):
        logging.info('=== Start training ===')
        """
        0. Pretrain model. Save pretrained ckpt
        1. Load pretrained model and pseudo bias label
        2. Build balanced dataset. Train pruning parameters
        3. Resume training with learned pruning parameters
        """

        args = self.args
        loader = self.loaders.train

        try:
            self._load_checkpoint(args.pretrain_iter, 'pretrain')
            print('Pretrained ckpt exists. Checking upweight index ckpt...')
        except:
            print('Start pretraining...')
            self.train_ERM(args.pretrain_iter)
            self._load_checkpoint(args.pretrain_iter, 'pretrain')

        if os.path.exists(ospj(args.checkpoint_dir, 'wrong_index.pth')):
            print('Upweight ckpt exists.')
        else:
            print('Upweight ckpt does not exist. Creating...')
            if args.pseudo_label_method == 'wrong' or args.mode == 'JTT':
                if args.earlystop_iter is not None: self._load_checkpoint(args.earlystop_iter, 'pretrain')
                self.save_wrong_idx(loader)
                self._load_checkpoint(args.pretrain_iter, 'pretrain')
            else:
                raise ValueError('No upweight ckpt')

        assert os.path.exists(ospj(args.checkpoint_dir, 'wrong_index.pth'))

        self.valid_logger.save()


        self.retrain(args.retrain_iter, freeze=True if args.mode != 'JTT' else False)
        self.valid_logger.save()
        print('Finished training')

    def evaluate(self):
        fetcher_val = self.loaders.val
        self._load_checkpoint(self.args.retrain_iter, 'retrain')
        print('Load model from ', ospj(self.args.checkpoint_dir, '{:06d}_{}_nets.ckpt'.format(self.args.retrain_iter, 'retrain')))
        self.nets.classifier.pruning_switch(False)
        self.nets.classifier.freeze_switch(True)

        total_acc, valid_attrwise_acc = self.validation(fetcher_val)
        self.report_validation(valid_attrwise_acc, total_acc, 0, which='Test', save_in_result=True)

        self._tsne(fetcher_val)

