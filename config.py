import numpy as np
import time
import random
import math
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import os
import json
import xlrd
from dataset import ImageList, IMBALANCECIFAR100
from sklearn import manifold
from torchvision import *
from torch.utils import *

parser = argparse.ArgumentParser(description = 'retrieval')
parser.add_argument('--dataset', type = str, default = 'cifar10', help = "dataset name")    #cifar10, cifar100, coco, imagenet, cifar100-LT
parser.add_argument('--datatype', type = str, default = 'full', help = "datatype")      #full, mini
parser.add_argument('--hash_bit', type = int, default = 48, help = "number of hash code bits")      #12, 16, 24, 32, 48, 64
parser.add_argument('--batch_size', type = int, default = 85, help = "batch size")
parser.add_argument('--epochs', type = int, default = 100, help = "epochs")
parser.add_argument('--cuda', type = int, default = 0, help = "cuda id")
parser.add_argument('--method', type = str, default = 'anchor', help = "methods")       #anchor, NCA, DHN
parser.add_argument('--origin', action = 'store_true', default = False, help = "without HHF method")
parser.add_argument('--irreg', action = 'store_true', default = False, help = "No regularization")
parser.add_argument('--alpha', type = float, default = 16, help = "alpha")
parser.add_argument('--beta', type = float, default = 0.001, help = "beta")
parser.add_argument('--delta', type = float, default = 0.2, help = "delta")
parser.add_argument('--imb_factor', type=float, default=0.01, help='how LT the dataset is ...') #0.1, 0.01, 0.001

# for testing
parser.add_argument('--test', action = 'store_true', default = False, help = "testing")
parser.add_argument('--tsne', action = 'store_true', default = False, help = "save tsne")
parser.add_argument('--dist', action = 'store_true', default = False, help = "calculate distance")
parser.add_argument('--figure', action = 'store_true', default = False, help = "Top-N and P-R curve")
parser.add_argument('--visual', action = 'store_true', default = False, help = "visualization")

args = parser.parse_args()

# Hyper-parameters
train_flag = bool(1 - args.test)
tsne_flag = args.tsne
reg_flag = bool(1 - args.irreg)
dist_flag = args.dist
figure_flag = args.figure
method = args.method
HHF_flag = bool(1 - args.origin)
visual_flag = args.visual

if method == 'DHN':
    based_method = 'pair'
else:
    based_method = 'proxy'

dataset = args.dataset
datatype = args.datatype    #full, mini
num_epochs = args.epochs

batch_size = args.batch_size
feature_rate = 0.001
criterion_rate = 0.01
num_bits = args.hash_bit

if dataset  ==  'cifar10':
    num_classes = 10
elif dataset  ==  'cifar100' or dataset  == 'cifar100-LT' or dataset == 'imagenet':
    num_classes = 100
elif dataset == 'coco':
    num_classes = 80

# find the value of ζ
if HHF_flag:
    sheet = xlrd.open_workbook('codetable.xlsx').sheet_by_index(0)
    threshold = sheet.row(num_bits)[math.ceil(math.log(num_classes, 2))].value
    print(threshold)

# hyper-parameters
alpha = args.alpha
beta = args.beta
delta = args.delta

# path for loading and saving models
path = './result/' + dataset + '_' + datatype + '_' + str(num_bits) + '_' + str(batch_size) + '_' + method + '_' + str(HHF_flag) + '_' + str(alpha + beta + delta)
model_path = path + '.ckpt'

if train_flag:
    file_path = path + '.txt'
    f = open(file_path, 'w')

if tsne_flag:
    fig_path = path + '.pdf'

# Device configuration
device = torch.device('cuda:'+str(args.cuda) if torch.cuda.is_available() else 'cpu')

#  data pre-treatment
data_transform = {
    "train": transforms.Compose([transforms.Resize((299, 299)),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
    "val": transforms.Compose([transforms.Resize((299, 299)),
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
# load train data
if dataset  ==  'cifar10':
    if datatype == 'full':
        retrieve = 50000
        trainset = torchvision.datasets.CIFAR10(root = './data',
                                                train = True,
                                                download = True,
                                                transform = data_transform['train'])
        # load test data
        testset = torchvision.datasets.CIFAR10(root = './data',
                                            train = False,
                                            download = True,
                                            transform = data_transform['val'])
        database = trainset
    elif datatype == 'mini':
        retrieve = 5000
        trainset = torchvision.datasets.CIFAR10(root = './data',
                                                train = True,
                                                download = True,
                                                transform = data_transform['val'])
        # load test data
        testset = torchvision.datasets.CIFAR10(root = './data',
                                            train = False,
                                            download = True,
                                            transform = data_transform['val'])

        loadfile = open('mini.txt', 'r').read()
        train_list, test_list = json.loads(loadfile)
        database = trainset + torch.utils.data.Subset(testset, list(set(range(10000)) - set(test_list)))
        if based_method == 'pair':
            trainset_ = torch.utils.data.Subset(trainset, train_list)
        trainset = torch.utils.data.Subset(trainset, train_list)
        testset = torch.utils.data.Subset(testset, test_list)

elif dataset  ==  'cifar100':
    retrieve = 50000
    trainset = torchvision.datasets.CIFAR100(root = './data',
                                            train = True,
                                            download = True,
                                            transform = data_transform['train'])
    if based_method == 'pair':
        trainset_ = torchvision.datasets.CIFAR100(root = './data',
                                            train = True,
                                            download = True,
                                            transform = data_transform['train'])
    # load test data
    testset = torchvision.datasets.CIFAR100(root = './data',
                                        train = False,
                                        download = True,
                                        transform = data_transform['val'])
    database = trainset

elif dataset  ==  'cifar100-LT':
    '''cifar100 long tailed version'''
    retrieve = 5000
    trainset = IMBALANCECIFAR100(root='./data', 
                                    imb_factor=args.imb_factor,
                                    train=True, 
                                    download=True, 
                                    transform=data_transform['train'])
    if based_method == 'pair':
        trainset_ = IMBALANCECIFAR100(root='./data', 
                                    imb_factor=args.imb_factor,
                                    train=True, 
                                    download=True, 
                                    transform=data_transform['train'])
    # load test data
    testset = torchvision.datasets.CIFAR100(root = './data',
                                        train = False,
                                        download = True,
                                        transform = data_transform['val'])
    
    database = torchvision.datasets.CIFAR100(root = './data',
                                        train = True,
                                        download = True,
                                        transform = data_transform['train'])

elif dataset == 'coco':
    retrieve = 5000
    trainset = ImageList(open('./data/coco/train.txt', 'r').readlines(), transform = data_transform['train'])
    if based_method == 'pair':
        trainset_ = ImageList(open('./data/coco/train.txt', 'r').readlines(), transform = data_transform['train'])
    testset = ImageList(open('./data/coco/test.txt', 'r').readlines(), transform = data_transform['val'])
    database = ImageList(open('./data/coco/database.txt', 'r').readlines(), transform = data_transform['val'])

elif dataset == 'imagenet':
    retrieve = 1000
    trainset = ImageList(open('./data/imagenet/train.txt', 'r').readlines(), transform = data_transform['train'])
    if based_method == 'pair':
        trainset_ = ImageList(open('./data/imagenet/train.txt', 'r').readlines(), transform = data_transform['train'])
    testset = ImageList(open('./data/imagenet/test.txt', 'r').readlines(), transform = data_transform['val'])
    database = ImageList(open('./data/imagenet/database.txt', 'r').readlines(), transform = data_transform['val'])

train_num = len(trainset)
test_num = len(testset)
database_num = len(database)

trainloader = data.DataLoader(dataset = trainset,
                            batch_size = batch_size,
                            shuffle = True,
                            num_workers = 0)

if based_method == 'pair':
    trainloader_ = data.DataLoader(dataset = trainset_,
                            batch_size = batch_size,
                            shuffle = True,
                            num_workers = 0)

testloader = data.DataLoader(dataset = testset,
                            batch_size = batch_size,
                            shuffle = False,
                            num_workers = 0)

databaseloader = data.DataLoader(dataset = database,
                            batch_size = batch_size,
                            shuffle = False,
                            num_workers = 0)

print('------------- data prepared -------------')