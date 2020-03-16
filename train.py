import sys
import matplotlib.pyplot as plt
from os.path import isfile
import torch.nn.init as init
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
from PIL import Image, ImageFilter
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import Dataset
from torchvision import transforms
from torch.optim import Adam, SGD, RMSprop
import time
from torch.autograd import Variable
import torch.functional as F
from tqdm import tqdm
from sklearn import metrics
import urllib
import pickle
import cv2
import torch.nn.functional as F
from torchvision import models
import seaborn as sns
import random
sys.path.append('./apex/')
from apex import amp
import sys
import Kappa
from torch.utils.data.sampler import WeightedRandomSampler
from albumentations import *

sys.path.append('..')
from efficientnet_pytorch import EfficientNet

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import cv2
import random_col
import json

lr = 1e-4
IMG_SIZE = 513
num_classes = 3
n_epochs = 200
num_samples = 600
random_eye = False
train_csv = pd.read_csv('/home/lixiaoxing/github/EfficientNet-PyTorch/data_cha/csv_20191219/CHA_GCN_right.csv')
#val_csv = pd.read_csv('/home/lixiaoxing/github/EfficientNet-PyTorch/data_cha/imgval_left_02.csv')
train = '/home/lixiaoxing/github/EfficientNet-PyTorch/data_cha/img'
test = '/home/lixiaoxing/github/EfficientNet-PyTorch/data_cha/img/'
save_model = '/home/lixiaoxing/github/EfficientNet-PyTorch/model/'
if os.path.exists(save_model) is False:
    os.makedirs(save_model)

if random_eye:
    csv_1_path = '/home/lixiaoxing/github/EfficientNet-PyTorch/data_cha/label_people_2.csv'
    csv_2_path = '/home/lixiaoxing/github/EfficientNet-PyTorch/data_cha/random_img/'+'label_random_'+str(num_classes)+'.csv'
    random_col.rand_img(csv_1_path, csv_2_path)
    train_csv = pd.read_csv(csv_2_path)


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

    if isfile(train + p):
        return train + (p)
    if isfile(test + p):
        return test + (p)
    return test+p


class MyDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe
        self.transform = transform
        
        self.IMG_SIZE = IMG_SIZE
        self.aug1 = OneOf([Rotate(p=0.6, limit=160, border_mode=0, value=0), Flip(p=0.6)], p=1)
        self.aug1_1 = Rotate(p=0.5, limit=160, border_mode=0, value=0)
        self.aug1_2 = Flip(p=0.5)
        self.aug2 = RandomBrightnessContrast(brightness_limit=0.45, contrast_limit=0.45, p=0.6)
        self.h_min = np.round(IMG_SIZE * 0.72).astype(int)
        self.h_max = np.round(IMG_SIZE * 0.9).astype(int)
        self.aug3 = RandomSizedCrop((self.h_min, self.h_max), IMG_SIZE, IMG_SIZE, w2h_ratio=1, p=0.)
        self.max_hole_size = int(IMG_SIZE / 10)
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
        image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
        #image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), 30), -4, 128)
        im = image
        image = transforms.ToPILImage()(image)

        if self.transform:
            image = self.transform(image)

        return image, label, p_path

def listToJson(data, json_save):
    jsonData = json.dumps(data)
    fileObject = open(json_save, 'w')
    fileObject.write(jsonData)
    fileObject.close()


train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation((-120, 120)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
train_df, val_df = train_test_split(train_csv, test_size=0.3, random_state=2018, stratify=train_csv.SCORE)
#train_df = train_csv
#val_df = val_csv

# Random sampling
#train_labels=train_df.values[:,1]
#sampler_count=[len(np.where(train_labels==i)[0])  for i in range(num_classes)]
#weight = np.array(1./np.array(sampler_count))
#weights = [weight[train_label[1]] for train_label in train_df.values ]
#sampler = WeightedRandomSampler(weights, num_samples=num_samples, replacement=True)

trainset = MyDataset(train_df.values, transform=train_transform)
#train_loader = torch.utils.data.DataLoader(trainset, batch_size=8, sampler=sampler, num_workers=6)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True, num_workers=6)
valset = MyDataset(val_df.values, transform=train_transform)
val_loader = torch.utils.data.DataLoader(valset, batch_size=8, shuffle=False, num_workers=6)

model = EfficientNet.from_name('efficientnet-b5')
#print(model)
#print(model._blocks[38]._depthwise_conv)
in_features = model._fc.in_features
model._fc = nn.Linear(in_features, 1)
model.load_state_dict(torch.load('/home/lixiaoxing/github/EfficientNet-PyTorch/data_cha/cv2_3.pt'))
model._fc = nn.Linear(in_features, num_classes)
#### freeze
#for param in list(model.parameters())[:-1]:
#    param.requires_grad=False

model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
#### freeze
#optimizer = torch.optim.Adam(model._fc.parameters(), lr=lr, weight_decay=1e-6)
criterion = nn.CrossEntropyLoss()

##### 学习率调整:https://www.jianshu.com/p/9643cba47655

### 每个一定的epoch，lr会自动乘以gamma
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

### 当指标不再提升时降低学习率, 注：scheduler.step(指标)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1,mode='min',patience=10,verbose=True)

### lambda自定义衰减
lambda1 = lambda epoch:np.sin(epoch) / epoch
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

### 三段式lr，epoch进入milestones范围内即乘以gamma，离开milestones范围之后再乘以gamma
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[10,50],gamma = 0.9)

### 连续衰减，每个epoch中lr都乘以gamma
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

### 余弦式调整，T_max 对应1/2个cos周期所对应的epoch数值
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 20)

model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)


def train_model(epoch):
    model.train()
    train_labels = []
    train_pre = []
    train_name = []
    avg_loss = 0.0
    optimizer.zero_grad()
    for idx, (imgs, labels, p) in enumerate(train_loader):
        #print(idx, labels, p)
        imgs_train, labels_train = imgs.cuda(), labels.cuda()
        #print(labels_train)
        #_,output_train = torch.max(model(imgs_train),0)

        #2019-12-13 model改，对应改
        #output_train = model(imgs_train)
        output_train, feat_train = model(imgs_train)
        feat_train_np = feat_train.cpu().detach()

        #ke = model.extract_features_up(imgs_train)
        #ke = ke.detach().cpu()
        #feature=ke.data.numpy()

        ##use sigmod to [0,1]
        #feature= 1.0/(1+np.exp(-1*feature))

        ## to [0,255]
        #feature=np.round(feature*255)
        #print(feature[0,0,:,:].shape)

        #cv2.imwrite('./img.jpg',feature[0,1,:,:])

        #ke = model.eppxtract_features(imgs_train)
        #ke = ke[1]*255
        #ke = ke.detach().cpu()
        #img_logit = transforms.ToPILImage()(ke)
        #img_logit.save('a.png')
      
        train_name.extend(p)

        loss = criterion(output_train,labels_train)
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        avg_loss += loss.item() / len(train_loader)
        _, preds_train = torch.max(output_train, 1)
        train_labels.extend(labels_train.data.cpu().numpy())
        train_pre.extend(preds_train.data.cpu().numpy())
        ### 2019-12-16
        if idx==0:     
            feats_train = feat_train_np
        else:
            feats_train = np.vstack((feats_train,feat_train_np))

    np.save('./features/feats_train_right_'+str(epoch)+'.npy', feats_train)
    listToJson(train_name, './features/name_train_right_'+str(epoch)+'.json')
    train_kappa,conf_t = Kappa.quadratic_weighted_kappa(np.mat(train_labels), np.mat(train_pre))
    return avg_loss, train_kappa


def eval_model(epoch):
    avg_val_loss = 0.
    model.eval()
    val_labels = []
    val_pre = []
    val_name = []
    with torch.no_grad():
        for idx, (imgs, labels, p) in enumerate(val_loader):
     
            imgs_vaild, labels_vaild = imgs.cuda(), labels.cuda()
            ## 2019-12-13
            #output_test = model(imgs_vaild)
            output_test, feat_val = model(imgs_vaild)
            feat_val_np = feat_val.cpu().numpy()

            #ke = model.extract_features_up(imgs_vaild)
            #ke = ke.detach().cpu()
            #feature=ke.data.numpy()

            #use sigmod to [0,1]
            #feature= 1.0/(1+np.exp(-1*feature))

            # to [0,255]
            #feature=np.round(feature*255)
            #feature=feature[0,:,:,:]
            #feature=feature.transpose(1,2,0)
            #print(feature.shape)

            #cv2.imwrite('./vie/'+str(epoch)+'_'+str(idx)+'.png',feature)
            #im = im[0].detach().cpu()
            #im = im.data.numpy()
            ##im = im.transpose(1,2,0)
            #cv2.imwrite('./vie/'+str(epoch)+'_'+str(idx)+'.jpg',im)

            avg_val_loss += criterion(output_test, labels_vaild).item() / len(val_loader)
            val_labels.extend(labels_vaild.data.cpu().numpy())
            val_name.extend(p)
            _, preds_test = torch.max(output_test, 1)
            val_pre.extend(preds_test.data.cpu().numpy())
            ### 2019-12-16
            if idx==0:     
                feats_val = feat_val_np
            else:
                feats_val = np.vstack((feats_val,feat_val_np))

        np.save('./features/feats_val_right_'+str(epoch)+'.npy', feats_val)
        listToJson(val_name, './features/name_val_right_'+str(epoch)+'.json')
        val_kappa,conf_mat = Kappa.quadratic_weighted_kappa(np.mat(val_labels), np.mat(val_pre))
    return avg_val_loss, val_kappa,conf_mat, val_name, val_labels, val_pre


val_kappa=0.0
best_avg_kappa = 0.0
best_avg_loss = 100.0
print('**'*50)
for epoch in range(n_epochs):

    #print('lr:', scheduler.get_lr()[0])
    start_time = time.time()
    avg_loss, train_kappa= train_model(epoch)
    avg_val_loss, val_kappa, conf_mat, path, val_label, val_pred = eval_model(epoch)
    elapsed_time = time.time() - start_time
    val_acc=0
    for j in range(len(conf_mat)):
        val_acc+=conf_mat[j][j]
    val_acc=val_acc/np.array(conf_mat).sum()
    print (np.array(conf_mat))
    print('cv-Epoch {}/{} \t loss={:.4f} \t train_kappa={:.4f} \t val_loss={:.4f} \t val_acc={:.4f} \t val_kappa={:.4f} \t time={:.2f}s'.format(
            epoch + 1, n_epochs, avg_loss, train_kappa, avg_val_loss, val_acc, val_kappa, elapsed_time))

    #if avg_val_loss < best_avg_loss :
    #    best_avg_loss = avg_val_loss
    torch.save(model.state_dict(), save_model+'cv_'+str(epoch)+'.pt')
    file = open(save_model+'cv_'+str(epoch)+'.txt','w')
    for line in range(len(path)):
        file.write(path[line]+','+str(val_label[line])+','+str(val_pred[line])+'\n') 
    file.close() 
        
    scheduler.step(avg_val_loss)
