from __future__ import print_function, division, absolute_import
import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import sys

sys.path.append('.')
import pretrainedmodels
import pretrainedmodels.utils

from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import Dataset
import pandas as pd
from albumentations import *
import numpy as np
from os.path import isfile
import cv2

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

model_names = sorted(name for name in pretrainedmodels.__dict__
                     if not name.startswith("__")
                     and name.islower()
                     and callable(pretrainedmodels.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--data', metavar='DIR', default="path_to_imagenet",
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='nasnetamobile',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: fbresnet152)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=1256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', default=True,
                    action='store_true', help='evaluate model on validation set')
parser.add_argument('--pretrained', default='imagenet', help='use pre-trained model')
parser.add_argument('--do-not-preserve-aspect-ratio',
                    dest='preserve_aspect_ratio',
                    help='do not preserve the aspect ratio when resizing an image',
                    action='store_false')
parser.set_defaults(preserve_aspect_ratio=True)
best_prec1 = 0


def main():
    global args, best_prec1
    args = parser.parse_args()
    #args.arch = 'vgg19'
    args.lr = 0.1
    args.epochs = 200
    args.data = '/home/lixiaoxing/github/EfficientNet-PyTorch/data_cha/img/' 
    img_size = 224
    
    # data
    train_csv = pd.read_csv('/home/lixiaoxing/github/EfficientNet-PyTorch/data_cha/csv_20191219/CHA_GCN_left.csv')
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation((-120, 120)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    train_df, val_df = train_test_split(train_csv, test_size=0.3, random_state=2018, stratify=train_csv.SCORE)
    trainset = MyDataset(train_df.values, img_size, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=6)
    valset = MyDataset(val_df.values, img_size, transform=train_transform)
    val_loader = torch.utils.data.DataLoader(valset, batch_size=16, shuffle=False, num_workers=6)


    # create model
    print("=> creating model '{}'".format(args.arch))
    if args.pretrained.lower() not in ['false', 'none', 'not', 'no', '0']:
        print("=> using pre-trained parameters '{}'".format(args.pretrained))
        model = pretrainedmodels.__dict__[args.arch](num_classes=1000,
                                                     pretrained=args.pretrained)
    else:
        model = pretrainedmodels.__dict__[args.arch]()
    
    in_features = model.last_linear.in_features
    model.last_linear = nn.Linear(in_features, 1)
    #model.load_state_dict(torch.load('/home/lixiaoxing/github/EfficientNet-PyTorch/data_cha/cv2_3.pt'))
    model.last_linear = nn.Linear(in_features, 3)
  
    model.cuda()
    print(model)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True


    '''
    # Data loading code
    #traindir = os.path.join(args.data, 'train')
    valdir = args.data

    #train_loader = torch.utils.data.DataLoader(
    #    datasets.ImageFolder(traindir, transforms.Compose([
    #        transforms.RandomSizedCrop(max(model.input_size)),
    #        transforms.RandomHorizontalFlip(),
    #        transforms.ToTensor(),
    #        normalize,
    #    ])),
    #    batch_size=args.batch_size, shuffle=True,
    #    num_workers=args.workers, pin_memory=True)



    #if 'scale' in pretrainedmodels.pretrained_settings[args.arch][args.pretrained]:
    #    scale = pretrainedmodels.pretrained_settings[args.arch][args.pretrained]['scale']
    #else:
    #    scale = 0.875
    '''
    scale = 0.875

    print('Images transformed from size {} to {}'.format(
        int(round(max(model.input_size) / scale)),
        model.input_size))

    '''
    val_tf = pretrainedmodels.utils.TransformImage(
        model,
        scale=scale,
        preserve_aspect_ratio=args.preserve_aspect_ratio
    )

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, val_tf),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
    '''

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=args.lr, 
                                 weight_decay=args.weight_decay)

    #optimizer = torch.optim.SGD(model.parameters(), args.lr,
    #                            momentum=args.momentum,
    #                            weight_decay=args.weight_decay)

    model = torch.nn.DataParallel(model).cuda()

    '''
    if args.evaluate:
        validate(val_loader, model, criterion)
        return
    '''

    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch)

        # evaluate on validation set
        prec1, prec5 = validate(val_loader, model, criterion)

        # remember best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
        }, is_best)


class MyDataset(Dataset):
    def __init__(self, dataframe, img_size, transform=None):
        self.df = dataframe
        self.transform = transform
        
        self.IMG_SIZE = img_size
        self.aug1 = OneOf([Rotate(p=0.6, limit=160, border_mode=0, value=0), Flip(p=0.6)], p=1)
        self.aug1_1 = Rotate(p=0.5, limit=160, border_mode=0, value=0)
        self.aug1_2 = Flip(p=0.5)
        self.aug2 = RandomBrightnessContrast(brightness_limit=0.45, contrast_limit=0.45, p=0.6)
        self.h_min = np.round(self.IMG_SIZE * 0.72).astype(int)
        self.h_max = np.round(self.IMG_SIZE * 0.9).astype(int)
        self.aug3 = RandomSizedCrop((self.h_min, self.h_max), self.IMG_SIZE, self.IMG_SIZE, w2h_ratio=1, p=0.)
        self.max_hole_size = int(self.IMG_SIZE / 10)
        self.aug4 = Cutout(p=0.6, max_h_size=self.max_hole_size, max_w_size=self.max_hole_size, num_holes=6)
        self.aug5 = RandomSunFlare(src_radius=self.max_hole_size, num_flare_circles_lower=10,
                                   num_flare_circles_upper=20, p=0.5)
        self.aug6 = ShiftScaleRotate(p=0.5)
        self.aug_ultimate = Compose([self.aug1_1, self.aug1_2, self.aug2, self.aug4,self.aug6], p=1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        label = self.df[idx][1]
        #label = np.expand_dims(label, -1)
        p = self.df[idx][4]
        p_path = expand_path(p)
        image = cv2.imread(p_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = crop_image_from_gray(image)
        image = cv2.resize(image, (self.IMG_SIZE, self.IMG_SIZE))
        #image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), 30), -4, 128)
        im = image
        image = transforms.ToPILImage()(image)

        if self.transform:
            image = self.transform(image)

        return image, label, p_path

def crop_image_from_gray(img, tol=7):
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if (check_shape == 0):  # image is too dark so that we crop out everything,
            return img  # return original image
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            #         print(img1.shape,img2.shape,img3.shape)
            img = np.stack([img1, img2, img3], axis=-1)
        #         print(img.shape)
        return img

def expand_path(p):
    p = str(p)

    if isfile(args.data + p):
        return args.data + (p)
    if isfile(args.data + p):
        return args.data + (p)
    return args.data+p

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input_imgs, target,p) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_imgs = input_imgs.cuda()
        #input_var = torch.autograd.Variable(input_imgs)
        #target_var = torch.autograd.Variable(target)
        input_var = input_imgs
        target_var = target

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 2))
        losses.update(loss.item(), input_imgs.size(0))
        top1.update(prec1.item(), input_imgs.size(0))
        top5.update(prec5.item(), input_imgs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))


def validate(val_loader, model, criterion):
    with torch.no_grad():
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        # switch to evaluate mode
        model.eval()

        end = time.time()
        for i, (input, target,p) in enumerate(val_loader):
            target = target.cuda()
            input = input.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target.data, topk=(1, 2))
            losses.update(loss.data.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        return top1.avg, top5.avg


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


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


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
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


if __name__ == '__main__':
    main()

