# -*- coding: utf-8 -*-
import sys

import timm

sys.path.append('.')
sys.path.append('..')
sys.path.append('../..')

import time
import torch
from model import *
import os
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from util import Logger, Bar, AverageMeter, accuracy, load_dataset, warp_decay, split_params, init_config, bptt_model_setting
from spikingjelly.activation_based import functional
from torch.nn.parallel import DistributedDataParallel

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from model.layer import *
def train(train_ldr, optimizer, model, t_model, evaluator, args):
    model.train()
    t_model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    if dist.get_rank() == 0:
        bar = Bar('Processing', max=len(train_ldr))

    for idx, (ptns, labels) in enumerate(train_ldr):
        device = next(model.parameters()).device
        ptns, labels = ptns.to(device), labels.to(device)

        # measure data loading time
        data_time.update(time.time() - end)

        optimizer.zero_grad()
        functional.reset_net(model.module)

        in_data, _ = torch.broadcast_tensors(ptns, torch.zeros((args.T,) + ptns.shape))
        in_data = in_data.reshape(-1, *in_data.shape[2:])
        output = model(in_data)
        with torch.no_grad():
            t_avg_fr = t_model(ptns)
        avg_fr = output.mean(dim=0)
        loss_time = 0.0
        loss_time2 = 0.0

        teacher_labels = make_teacher(output, labels)


        for i in range(args.T):
            loss_time += kd_loss(output[i], teacher_labels.detach(), 1)
            loss_time2 += kd_loss(output[i], t_avg_fr.detach(), 1)
        loss_time = loss_time / args.T
        loss_time2 = loss_time2 / args.T
        hard_loss = cal_loss(output, labels, evaluator)
        loss = hard_loss + loss_time * args.beta + loss_time2 * args.alpha

        loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        prec1, prec5 = accuracy(avg_fr.data, labels.data, topk=(1, 5))

        losses.update(loss.data.item(), ptns.size(0))
        top1.update(prec1.item(), ptns.size(0))
        top5.update(prec5.item(), ptns.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        if dist.get_rank() == 0:
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

    if dist.get_rank() == 0:
        bar.finish()

    return top1.avg, losses.avg


def test(val_ldr, model,t_model, evaluator, args):
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()

    if dist.get_rank() == 0:
        bar = Bar('Processing', max=len(val_ldr))
    ann_acc = 0.0
    count = 0
    with torch.no_grad():
        for idx, (ptns, labels) in enumerate(val_ldr):
            device = next(model.parameters()).device
            ptns, labels = ptns.to(device), labels.to(device)

            data_time.update(time.time() - end)

            functional.reset_net(model.module)

            in_data, _ = torch.broadcast_tensors(ptns, torch.zeros((args.T,) + ptns.shape))
            in_data = in_data.reshape(-1, *in_data.shape[2:])
            output = model(in_data)
            avg_fr = output.mean(dim=0)
            t_avg_fr =t_model(ptns)
            loss = evaluator(avg_fr, labels)

            acc1, acc5 = accuracy(t_avg_fr.data, labels.data, topk=(1, 5))
            ann_acc += acc1*ptns.size(0)
            count += ptns.size(0)
            # measure accuracy and record loss
            prec1, prec5 = accuracy(avg_fr.data, labels.data, topk=(1, 5))
            losses.update(loss.data.item(), ptns.size(0))
            top1.update(prec1.item(), ptns.size(0))
            top5.update(prec5.item(), ptns.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress

            if dist.get_rank() == 0:
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
        if dist.get_rank() == 0:
            bar.finish()
        ann_acc /= count

        return top1.avg, losses.avg


def main():
    dist.init_process_group(backend='nccl')
    world_size = dist.get_world_size()
    rank = dist.get_rank()

    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    # set device, data type
    device, dtype = torch.device("cuda" if torch.cuda.is_available() else "cpu"), torch.float

    log = Logger(args, args.log_path)
    log.info_args(args)
    writer = SummaryWriter(args.log_path)

    train_data, val_data, num_class = load_dataset(args.dataset, args.data_path, cutout=args.cutout,
                                                   auto_aug=args.auto_aug)
    train_sampler = DistributedSampler(train_data, rank=rank)
    val_sampler = DistributedSampler(val_data, rank=rank)
    train_ldr = DataLoader(dataset=train_data, batch_size=args.train_batch_size // world_size, shuffle=False,
                           sampler=train_sampler,
                           pin_memory=True, num_workers=args.num_workers)
    val_ldr = DataLoader(dataset=val_data, batch_size=args.val_batch_size // world_size, shuffle=False,
                         sampler=val_sampler,
                         pin_memory=True, num_workers=args.num_workers)

    kwargs_spikes = {'v_reset': args.v_reset, 'thresh': args.thresh, 'decay': warp_decay(args.decay),
                     'detach_reset': args.detach_reset}
    model = eval(args.stu_arch + f'(num_classes={num_class}, **kwargs_spikes)')
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    t_model = timm.create_model(args.tea_arch, pretrained=False)
    checkpoint = torch.load(args.tea_path, map_location=device)
    t_model.load_state_dict(checkpoint, strict=True)
    t_model.to(device, dtype)

    bptt_model_setting(model, time_step=args.T, step_mode=args.step_mode)
    model.to(device, dtype)
    model = DistributedDataParallel(model, device_ids=[rank])

    params = split_params(model)
    params = [
        {'params': params[1], 'weight_decay': args.wd},
        {'params': params[2], 'weight_decay': 0}
    ]

    if args.optim.lower() == 'sgdm':
        optimizer = optim.SGD(params, lr=args.lr, momentum=0.9)
    elif args.optim.lower() == 'adam':
        optimizer = optim.Adam(params, lr=args.lr, amsgrad=False)
    else:
        raise NotImplementedError()

    evaluator = torch.nn.CrossEntropyLoss()


    start_epoch = 0
    best_epoch = 0
    best_acc = 0.0
    args.start_epoch = start_epoch
    if args.scheduler.lower() == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=0, T_max=args.num_epoch)
    else:
        raise NotImplementedError()
    for epoch in range(start_epoch, args.num_epoch):
        train_acc, train_loss = train(train_ldr, optimizer, model, t_model, evaluator, args=args)
        if args.scheduler != 'None':
            scheduler.step()
        val_acc, val_loss = test(val_ldr, model,t_model, evaluator, args=args)
        tensor_train_acc = torch.tensor(train_acc, device='cuda')
        tensor_train_loss = torch.tensor(train_loss, device='cuda')
        tensor_val_acc = torch.tensor(val_acc, device='cuda')
        tensor_val_loss = torch.tensor(val_loss, device='cuda')
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(tensor_train_acc, op=dist.ReduceOp.SUM)
            dist.all_reduce(tensor_train_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(tensor_val_acc, op=dist.ReduceOp.SUM)
            dist.all_reduce(tensor_val_loss, op=dist.ReduceOp.SUM)
            world_size = dist.get_world_size()
            tensor_train_acc /= world_size
            tensor_train_loss /= world_size
            tensor_val_acc /= world_size
            tensor_val_loss /= world_size
        train_acc = tensor_train_acc.item()
        train_loss = tensor_train_loss.item()
        val_acc = tensor_val_acc.item()
        val_loss = tensor_val_loss.item()
        if dist.get_rank() == 0:
            if val_acc > best_acc:  # saving checkpoint
                best_acc = val_acc
                best_epoch = epoch
                state = {
                    'best_acc': best_acc,
                    'best_epoch': epoch,
                    'best_net': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                torch.save(state, os.path.join(args.log_path, 'model_weights.pth'))


            log.info(
                'Epoch %03d: train loss %.5f, test loss %.5f, train acc %.5f, test acc %.5f, Saved custom_model..  with acc %.5f in the epoch %03d' % (
                    epoch, train_loss, val_loss, train_acc, val_acc, best_acc, best_epoch))
            # record in tensorboard
            writer.add_scalars('Loss', {'val': val_loss, 'train': train_loss}, epoch + 1)
            writer.add_scalars('Acc', {'val': val_acc, 'train': train_acc}, epoch + 1)



if __name__ == '__main__':
    from config.config import args

    init_config(args)
    main()
