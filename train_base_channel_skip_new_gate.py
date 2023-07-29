"""Training file of the original ResNet model"""
import argparse
import shutil
import time

import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision
import logging
import models_channel_skip_new_gate
import numpy as np
from functools import reduce
import sys

model_names = sorted(name for name in models_channel_skip_new_gate.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models_channel_skip_new_gate.__dict__[name]))


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('cmd', choices=['train', 'test', 'map', 'locate'])
    parser.add_argument('--dataset', '-d', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100'],
                        help='dataset choice')
    parser.add_argument('arch', default='resnet74', choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names) +
                             ' (default: resnet74)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=120, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument('--print-freq', '-p', default=1, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default='/home/zmx/DDI-main/save_checkpoints/densenet40_skip_channel_new_gate_minimum_100_Early_Exit_beta_1e-05xent/model_best.pth.tar', type=str, metavar='PATH',help='path to latest checkpoint (default: none)')
    # parser.add_argument('--resume', default=' ', type=str, metavar='PATH',
    #                     help='path to latest checkpoint (default: none)')
    parser.add_argument('-e', '--evaluate', dest='evaluate',
                        action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--save-folder', default='save_checkpoints', type=str,
                        help='folder to save the checkpoints')
    parser.add_argument('--lr-adjust', dest='lr_adjust',
                        choices=['linear', 'step'], default='step')
    parser.add_argument('--step-ratio', dest='step_ratio', type=float,
                        default=0.01)
    parser.add_argument('--minimum', default=100, type=float,
                        help='minimum')
    parser.add_argument('--computation_cost', default=True, type=bool,
                        help='using computation cost as regularization term')
    parser.add_argument('--beta', default=6e-1, type=float,
                        help='coefficient')
    parser.add_argument('--torchvision', default='False',
                        help='whether checkpoint is from torchvision')
    parser.add_argument('--proceed', default='False',
                        help='whether this experiment continues from a checkpoint')
    parser.add_argument('--loss', default='xent',
                        help='The loss function')
    args = parser.parse_args()
    return args


crop_size = 32
padding = 4


def prepare_train_data(dataset='cifar', batch_size=64,
                       shuffle=True, num_workers=1):
    if 'cifar' in dataset:
        transform_train = transforms.Compose([
            transforms.RandomCrop(crop_size, padding=padding),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        trainset = torchvision.datasets.__dict__[dataset.upper()](
            root='/home/zmx/skipnet-master/data', train=True, download=True,
            transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset,
                                                   batch_size=batch_size,
                                                   shuffle=shuffle,
                                                   num_workers=num_workers, drop_last=True)
    elif 'svhn' in dataset:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728),
                                 (0.1980, 0.2010, 0.1970)),
        ])
        trainset = torchvision.datasets.__dict__[dataset.upper()](
            root='/home/zmx/skipnet-master/data',
            split='train',
            download=True,
            transform=transform_train
        )

        transform_extra = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4300, 0.4284, 0.4427),
                                 (0.1963, 0.1979, 0.1995))

        ])

        extraset = torchvision.datasets.__dict__[dataset.upper()](
            root='/home/zmx/skipnet-master/data',
            split='extra',
            download=True,
            transform=transform_extra
        )

        total_data = torch.utils.data.ConcatDataset([trainset, extraset])

        train_loader = torch.utils.data.DataLoader(total_data,
                                                   batch_size=batch_size,
                                                   shuffle=shuffle,
                                                   num_workers=num_workers, drop_last=True)
    else:
        train_loader = None
    return train_loader


def prepare_test_data(dataset='cifar', batch_size=64,
                      shuffle=False, num_workers=1):
    if 'cifar' in dataset:
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        testset = torchvision.datasets.__dict__[dataset.upper()](
            root='/home/zmx/skipnet-master/data',    #/mnt/DDI/data/train
            train=False,
            download=True,
            transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  num_workers=num_workers, drop_last=True
                                                  )
    elif 'svhn' in dataset:
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4524, 0.4525, 0.4690),
                                 (0.2194, 0.2266, 0.2285)),
        ])
        testset = torchvision.datasets.__dict__[dataset.upper()](
            root='/home/zmx/skipnet-master/data',
            split='test',
            download=True,
            transform=transform_test)
        np.place(testset.labels, testset.labels == 10, 0)
        test_loader = torch.utils.data.DataLoader(testset,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  num_workers=num_workers, drop_last=True)
    else:
        test_loader = None
    return test_loader


def main():
    args = parse_args()

    args.save_path = save_path = os.path.join(args.save_folder, args.arch + '_skip_channel_new_gate_minimum_' + str(
        args.minimum) + '_Early_Exit_' + 'beta_' + str(args.beta) + args.loss)
    os.makedirs(args.save_path, exist_ok=True)
    args.logger_file = os.path.join(save_path, 'log_{}.txt'.format(args.cmd))
    handlers = [
        logging.FileHandler(args.logger_file, mode='w'),
        logging.StreamHandler()]

    logging.basicConfig(level=logging.INFO,
                        datefmt='%m-%d-%y %H:%M',
                        format='%(asctime)s:%(message)s',
                        handlers=handlers
                        )

    if args.cmd == 'train':
        logging.info("start training {}".format(args.arch))
        run_training(args)
    elif args.cmd == 'test':
        logging.info('start evaluating {} with checkpoints from {},'
                     .format(args.arch, args.resume))
        test_model(args)


def run_training(args):
    # create model
    model = models_channel_skip_new_gate.__dict__[args.arch](args.pretrained)

    model = torch.nn.DataParallel(model).cuda()

    best_prec1 = 0

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logging.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            if args.proceed == 'True':
                args.start_epoch = checkpoint['epoch']
            else:
                args.start_epoch = 0
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            logging.info("=> loaded checkpoint '{}' (epoch {})"
                         .format(args.resume, checkpoint['epoch']))
        else:
            logging.info("=> no checkpoint found at '{}'".format(args.resume))

    #     cudnn.benchmark = True

    # Data loading code

    train_loader = prepare_train_data(dataset=args.dataset,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      num_workers=args.workers)

    test_loader = prepare_test_data(dataset=args.dataset,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=args.workers)

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.loss == 'adjust':
        train_criterion = MyAdjustCrossEntropyLoss().cuda()
    else:
        train_criterion = nn.CrossEntropyLoss().cuda()

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(args, optimizer, epoch)

        # train for one epoch
        train(args, train_loader, model, criterion, train_criterion, optimizer, epoch)

        # evaluate on validation set
        prec1 = validate(args, test_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        checkpoint_path = os.path.join(args.save_path,
                                       'checkpoint_{:03d}.pth.tar'.format(
                                           epoch))


        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, filename=checkpoint_path)
        shutil.copyfile(checkpoint_path,
                        os.path.join(args.save_path, 'checkpoint_latest.pth.tar'))


def test_model(args):
    # create model

    #     t1 = time.time()
    model = models_channel_skip_new_gate.__dict__[args.arch](args.pretrained)
    model = torch.nn.DataParallel(model).cuda()



    if args.resume:
        if os.path.isfile(args.resume):
            logging.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            logging.info("=> loaded checkpoint '{}' (epoch {})"
                         .format(args.resume, checkpoint['epoch']))
        else:
            logging.info("=> no checkpoint found at '{}'".format(args.resume))


    # Data loading code
    test_loader = prepare_test_data(dataset=args.dataset,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=args.workers)

    criterion = nn.CrossEntropyLoss().cuda()

    validate(args, test_loader, model, criterion)


def train(args, train_loader, model, criterion, train_criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    top1_list = []
    for idx in range(7):
        top1_list.append(AverageMeter())
    cp_record = AverageMeter()


    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # target = target.cuda(async=True)
        target = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output, masks, gprobs = model(input_var)



        computation_cost = 0


        computation_cost *= args.beta


        reg = 1

        if args.computation_cost:
            loss = criterion(output, target_var) + computation_cost * reg

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))
        # cp_record.update(cp_ratio, 1)
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logging.info(
                'Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Prec_Main@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                'Prec_Main@5 {top5.val:.3f} ({top5.avg:.3f})\t'
                'Energy_ratio: {cp_record.val:.3f}({cp_record.avg:.3f})\t'.format(
                    epoch,
                    i,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    top1=top1,
                    top5=top5,
                    cp_record=cp_record))


def validate(args, test_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    top1_list = []
    for idx in range(7):
        top1_list.append(AverageMeter())



    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target) in enumerate(test_loader):
        # target = target.cuda(async=True)
        target = target.cuda()
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output, masks, gprobs = model(input_var)

        computation_cost = 0
        computation_all = 0



        loss = criterion(output, target_var)

        computation_cost *= args.beta


        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))
        # cp_record.update(cp_ratio, 1)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logging.info(
                'Test: [{0}/{1}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Prec_Main@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                'Prec_Main@5 {top5.val:.3f} ({top5.avg:.3f})\t'.format(
                    i,
                    len(test_loader),
                    batch_time=batch_time,
                    loss=losses,
                    top1=top1,
                    top5=top5,
                     ))  # cp_record=cp_record))

    logging.info(' * Prec_Main@1 {top1.avg:.3f} Prec_Main@5 {top5.avg:.3f}\t'
                 .format(top1=top1,
                         top5=top5
                        ))

    return top1.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        save_path = os.path.dirname(filename)
        shutil.copyfile(filename, os.path.join(save_path, 'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(args, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (args.step_ratio ** (epoch // 30))
    logging.info('Epoch [{}] Learning rate: {}'.format(epoch, lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class MyAdjustCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(MyAdjustCrossEntropyLoss, self).__init__()

    def forward(self, output, target):
        eplison = 1e-5
        batch_size = output.size()[0]  # batch_size
        logits = F.log_softmax(output, dim=1)  # compute the log of softmax values
        loss = -logits[range(batch_size), target]  # pick the values corresponding to the labels
        adjust_weight = Variable(torch.reciprocal(torch.mean(loss) + eplison), requires_grad=False)
        return (torch.sum(loss) * adjust_weight) / batch_size


if __name__ == '__main__':
    main()
