import sys

sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')

import time
import torch
import torch.nn as nn
from model import *
import os
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from util import Logger, Bar, AverageMeter, accuracy, load_dataset, setup_seed, get_model_name


def train(train_ldr, optimizer, model, evaluator, args):
    model.train()

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    end = time.time()

    bar = Bar('Processing', max=len(train_ldr))
    for idx, (ptns, labels) in enumerate(train_ldr):
        device = next(model.parameters()).device
        ptns, labels = ptns.to(device), labels.to(device)
        if 'dvs' in args.dataset.lower():
            ptns = ptns.mean(1)
        # measure data loading time
        data_time.update(time.time() - end)

        optimizer.zero_grad()

        out = model(ptns)

        loss = evaluator(out, labels)
        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        prec1, prec5 = accuracy(out.data, labels.data, topk=(1, 5))
        losses.update(loss.data.item(), ptns.size(0))
        top1.update(prec1.item(), ptns.size(0))
        top5.update(prec5.item(), ptns.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
            batch=idx + 1,
            size=len(train_ldr),
            data=data_time.avg,
            bt=batch_time.avg,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=top1.avg,
            top5=top5.avg,
        )
        bar.next()
    bar.finish()

    return top1.avg, losses.avg


def test(val_ldr, model, evaluator, args):
    model.eval()

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    end = time.time()
    bar = Bar('Processing', max=len(val_ldr))
    with torch.no_grad():
        for idx, (ptns, labels) in enumerate(val_ldr):
            device = next(model.parameters()).device
            ptns, labels = ptns.to(device), labels.to(device)
            if 'dvs' in args.dataset.lower():
                ptns = ptns.mean(1)

            out = model(ptns)

            loss = evaluator(out, labels)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(out.data, labels.data, topk=(1, 5))
            losses.update(loss.data.item(), ptns.size(0))
            top1.update(prec1.item(), ptns.size(0))
            top5.update(prec5.item(), ptns.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f} | top5: {top5: .4f}'.format(
                batch=idx + 1,
                size=len(val_ldr),
                data=data_time.avg,
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                top1=top1.avg,
                top5=top5.avg,
            )
            bar.next()
        bar.finish()

        return top1.avg, losses.avg


def main():
    device, dtype = torch.device(args.device), torch.float
    torch.cuda.set_device(device)

    log = Logger(args, args.log_path)
    log.info_args(args)
    writer = SummaryWriter(args.log_path)

    train_data, val_data, num_class = load_dataset(args.dataset, args.data_path, cutout=args.cutout,
                                                   auto_aug=args.auto_aug)
    train_ldr = DataLoader(dataset=train_data, batch_size=args.train_batch_size, shuffle=True,
                           pin_memory=True, num_workers=args.num_workers)
    val_ldr = DataLoader(dataset=val_data, batch_size=args.val_batch_size, shuffle=False,
                         pin_memory=True, num_workers=args.num_workers)

    evaluator = torch.nn.CrossEntropyLoss()
    from model import  ResNet_ANN

    ann_model = ResNet_ANN.__dict__[args.arch1](num_classes=num_class,
                                                   in_channels=2 if 'dvs' in args.dataset.lower() else 3)

    ann_model.to(device)


    ann_model.to(device)

    params = [
        {'params': ann_model.parameters(), 'weight_decay': args.wd},
    ]

    if args.optim.lower() == 'sgdm':
        optimizer = optim.SGD(params, lr=args.lr, momentum=0.9)
    elif args.optim.lower() == 'adamw':
        optimizer = optim.AdamW(params, lr=args.lr, )
    else:
        raise NotImplementedError()

    start_epoch = 0
    best_epoch = 0
    best_acc = 0.0
    if args.resume is not None:
        state = torch.load(args.resume, map_location=device)
        ann_model.load_state_dict(state['best_net'])
        optimizer.load_state_dict(state['optimizer'])
        start_epoch = state['best_epoch']
        best_acc = state['best_acc']
        log.info('Load checkpoint from epoch {}'.format(start_epoch))
        log.info('Best accuracy so far {}.'.format(best_acc))
        log.info('Test the checkpoint: {}'.format(test(val_ldr, ann_model, evaluator, args=args)))

    args.start_epoch = start_epoch
    if args.scheduler.lower() == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=args.num_epoch)
    else:
        raise NotImplementedError()

    for epoch in range(start_epoch, args.num_epoch):

        train_acc1, train_loss = train(train_ldr, optimizer, ann_model, evaluator, args)

        if scheduler is not None:
            scheduler.step()
        val_acc1, val_loss = test(val_ldr, ann_model, evaluator, args)
        if val_acc1 > best_acc:
            best_acc = val_acc1
            best_epoch = epoch
            state = {
                'best_acc': best_acc,
                'best_epoch': epoch,
                'best_net': ann_model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(state, os.path.join(args.log_path, 'model_weights.pth'))
        log.info(
            'Epoch %03d: train loss %.5f, test loss %.5f, train acc %.5f, test acc %.5f, Saved custom_model..  with acc %.5f in the epoch %03d' % (
                epoch, train_loss, val_loss, train_acc1, val_acc1, best_acc, best_epoch))

        # record in tensorboard
        writer.add_scalars('Loss', {'val': val_loss, 'train': train_loss}, epoch + 1)
        writer.add_scalars('Acc', {'val': val_acc1, 'train': train_acc1}, epoch + 1)

    log.info(
        'Finish training: the best validation accuracy of SNN is {} in epoch {}. \n The relate checkpoint path: {}'.format(
            best_acc, best_epoch, os.path.join(args.log_path, 'model_weights.pth')))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Training SNN')
    parser.add_argument('--seed', default=None, type=int, help='random seed')
    parser.add_argument('--arch', default="resnet18", type=str, )
    parser.add_argument('--dataset', default="CIFAR10_DVS_Aug", type=str, help="CIFAR10|CIFAR100")
    parser.add_argument('--data_path', default="~/data/", type=str)
    parser.add_argument('--log_path', default="./log", type=str, help="log path")
    parser.add_argument('--auto_aug', action='store_true')
    parser.add_argument('--cutout', action='store_true')
    parser.add_argument('--resume', default=None, type=str, help='pth file that holds the model parameters')
    parser.add_argument('--train_batch_size', default=128, type=int)
    parser.add_argument('--val_batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--wd', default=5e-4, type=float)
    parser.add_argument('--num_epoch', default=300, type=int)
    parser.add_argument('--device', default="cuda:0", type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--optim', default='SGDM', type=str)
    parser.add_argument('--decay', default=None, type=float)
    parser.add_argument('--scheduler', default='COSINE', type=str)

    args = parser.parse_args()

    seed = setup_seed(args.seed)
    args.print_freq = 100
    args.seed = seed
    import datetime

    ymdhms = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if not os.path.exists(args.log_path):
        os.mkdir(args.log_path)

    model_name = ''
    aug_str = '_'.join(['cut' if args.cutout else ''] + ['aug' if args.auto_aug else ''])
    if aug_str[0] != '_': aug_str = '_' + aug_str
    if aug_str[-1] != '_': aug_str = aug_str + '-'
    model_name += args.dataset.lower() + aug_str + 'ann' + '_' + args.arch.lower() + '_opt_' + args.optim.lower() + '_wd_' + str(
        args.wd)
    cas_num = len([one for one in os.listdir(args.log_path) if one.startswith(model_name)])
    model_name += '_cas_' + str(cas_num)

    args.log_path = os.path.join(args.log_path, ymdhms + '-' + model_name)
    if not os.path.exists(args.log_path):
        os.mkdir(args.log_path)
    main()
