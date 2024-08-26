import os
import argparse

from munch import Munch
from torch.backends import cudnn
import torch

from training.pruning_solver import PruneSolver
from util import setup, save_config, modify_args_for_baselines


def main(args):
    print(args)
    args = setup(args) # Making folders following exp_name
    save_config(args)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)

    solver = PruneSolver(args)


    if args.phase == 'train':
        solver.train()
    else:
        solver.evaluate()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='prune')

    # Data arguments
    parser.add_argument('--data', type=str, default='celebA',
                        choices=['celebA', 'isic', 'cub'])

    # weight for objective functions
    parser.add_argument('--lambda_con_prune', type=float, default=0.1)
    parser.add_argument('--lambda_con_retrain', type=float, required=False, default=0.1)
    parser.add_argument('--lambda_sparse', type=float, default=1e-8)

    # training arguments
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--no_lr_scheduling', default=False, action='store_true')

    # Pretraining
    parser.add_argument('--lr_decay_step_pre', type=int, default=60)
    parser.add_argument('--lr_gamma_pre', type=float, default=0.1)
    parser.add_argument('--lr_pre', type=float, default=1e-1)
    parser.add_argument('--pretrain_iter', type=int, default=10000)

    # Pruning
    parser.add_argument('--lr_prune', type=float, default=1e-2)
    parser.add_argument('--pruning_ep', type=int, default=20)

    # Retraining
    parser.add_argument('--optimizer', type=str, required=False,
                        choices=['Adam', 'SGD'], default='Adam')
    parser.add_argument('--lr_main', type=float, default=1e-2)
    parser.add_argument('--retrain_ep', type=int, default=5)
    parser.add_argument('--lr_decay_step_main', type=int, default=60)
    parser.add_argument('--lr_gamma_main', type=float, default=0.1)

    parser.add_argument('--weight_decay', type=float, default=1e-4)

    # misc
    parser.add_argument('--num_workers', type=int, default=1,
                        help='Number of workers used in DataLoader')
    parser.add_argument('--seed', type=int, default=99,
                        help='Seed for random number generator')
    parser.add_argument('--imagenet', default=True, action='store_true')


    # directory for training
    parser.add_argument('--train_root_dir', type=str, default='dataset')
    parser.add_argument('--val_root_dir', type=str, default='dataset')
    parser.add_argument('--checkpoint_dir', type=str, default='expr/checkpoints',
                        help='Directory for saving network checkpoints')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Nametag for the experiment')

    # save checkpoints
    parser.add_argument('--print_every', type=int, default=1000)
    parser.add_argument('--save_every', type=int, default=1000)
    parser.add_argument('--eval_every', type=int, default=500)
    parser.add_argument('--save_every_retrain', type=int, default=1)

    args = parser.parse_args()
    args = modify_args_for_baselines(args)

    main(args)
