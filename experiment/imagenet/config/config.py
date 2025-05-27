# -*- coding: utf-8 -*-
import argparse

parser = argparse.ArgumentParser(description='Training SNN')
parser.add_argument('--seed', default=60, type=int, help='random seed')

# model setting
parser.add_argument('--stu_arch', default="preact_resnet34", type=str, help="preact_resnet34")

parser.add_argument('--tea_arch', default="resnet34", type=str, help="resnet34")

parser.add_argument('--tea_path', default="your pt file", type=str, help='pth file that holds the model parameters')

# input data preprocess
parser.add_argument('--dataset', default="imagenet", type=str, help="CIFAR10|CIFAR100")
parser.add_argument('--data_path', default="./dataset/", type=str)
parser.add_argument('--log_path', default="./log", type=str, help="log path")
parser.add_argument('--auto_aug', default=True, action='store_true')
parser.add_argument('--cutout', default=True, action='store_true')

# learning setting
parser.add_argument('--optim', default='SGDM', type=str)
parser.add_argument('--scheduler', default='COSINE', type=str)
parser.add_argument('--train_batch_size', default=512, type=int)
parser.add_argument('--val_batch_size', default=512, type=int)
parser.add_argument('--lr', default=0.2, type=float)
parser.add_argument('--wd', default=2e-5, type=float)
parser.add_argument('--num_epoch', default=100, type=int)
parser.add_argument('--num_workers', default=16, type=int)

# spiking neuron setting
parser.add_argument('--decay', default=0.2, type=float)
parser.add_argument('--v_reset', default=None, type=float)
parser.add_argument('--thresh', default=1.0, type=float)
parser.add_argument('--T', default=4, type=int, help='num of time steps')
parser.add_argument('--step_mode', default='m', help='step mode')
parser.add_argument('--detach_reset', default=False, action='store_true')
parser.add_argument('--alpha', default=0.0, type=float)
parser.add_argument('--beta', default=0.0, type=float)
parser.add_argument("--local-rank", default=-1)

args = parser.parse_args()
