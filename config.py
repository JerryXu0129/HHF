import time
import random
import math
import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
import sys
import argparse
import torch.nn as nn
import torch.nn.functional as F
import json
import xlrd
from dataset import *
from sklearn import manifold
from torch.utils import *
from scipy.spatial.distance import cdist

parser = argparse.ArgumentParser(description = 'retrieval')
parser.add_argument('--dataset', type = str, default = 'imagenet', help = "dataset name")    #imagenet, coco, cifar10, cifar100 
parser.add_argument('--datatype', type = str, default = 'full', help = "datatype")      #full, mini
parser.add_argument('--hash_bit', type = int, default = 48, help = "number of hash code bits")      #12, 16, 24, 32, 48, 64
parser.add_argument('--batch_size', type = int, default = 85, help = "batch size")
parser.add_argument('--epochs', type = int, default = 100, help = "epochs")
parser.add_argument('--cuda', type = int, default = 0, help = "cuda id")
parser.add_argument('--method', type = str, default = 'anchor', help = "methods")       #anchor, NCA, DHN
parser.add_argument('--backbone', type = str, default = 'googlenet', help = "backbone")     #googlenet, resnet
parser.add_argument('--origin', action = 'store_true', default = False, help = "without HHF method")
parser.add_argument('--irreg', action = 'store_true', default = False, help = "without quantization")
parser.add_argument('--alpha', type = float, default = 16, help = "alpha")
parser.add_argument('--beta', type = float, default = 0.001, help = "beta")
parser.add_argument('--delta', type = float, default = 0.2, help = "delta")
parser.add_argument('--retrieve', type = int, default = 0, help = "retrieval number")
parser.add_argument('--seed', type = int, default = 0, help = "random seed for initializing proxies")
parser.add_argument('--test', action = 'store_true', default = False, help = "testing")

args = parser.parse_args()

# Hyper-parameters
train_flag = bool(1 - args.test)
reg_flag = bool(1 - args.irreg)
method = args.method
backbone = args.backbone
HHF_flag = bool(1 - args.origin)
retrieve = args.retrieve

if method in ['DHN']:
    based_method = 'pair'
else:
    based_method = 'proxy'

dataset = args.dataset
datatype = args.datatype   
num_epochs = args.epochs

batch_size = args.batch_size
feature_rate = 0.001
criterion_rate = 0.01
num_bits = args.hash_bit

# hyper-parameters
alpha = args.alpha
beta = args.beta
delta = args.delta

seed = args.seed
# path for loading and saving models


path = 'result/' + dataset + '_' + datatype + '_' + backbone + '_' + str(num_bits) + '_' + str(batch_size) + '_' + method + '_' + str(HHF_flag) + '_' + str(alpha + beta + delta)
model_path = path + '.ckpt'

if train_flag and datatype != 'toy':
    file_path = path + '.txt'
    f = open(file_path, 'w')

# Device configuration
device = torch.device('cuda:'+str(args.cuda) if torch.cuda.is_available() else 'cpu')

#  data pre-treatment   
if backbone == 'googlenet':
    data_transform = {
        "train": transforms.Compose([transforms.Resize((299, 299)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),                
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),    
        "val": transforms.Compose([transforms.Resize((299, 299)),               
                                    transforms.ToTensor(),                                                  
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}   
elif backbone in ['resnet', 'alexnet']:                                                          
    data_transform = {                                                                          
        "train": transforms.Compose([transforms.Resize((224, 224)),                           
                                    transforms.RandomHorizontalFlip(),                     
                                    transforms.ToTensor(),                             
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]), 
        "val": transforms.Compose([transforms.Resize((224, 224)),                                             
                                    transforms.ToTensor(),                                                
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}
# load train data
if dataset  ==  'cifar10':
    num_classes = 10
    if datatype == 'full':
        if retrieve == 0:
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
        if retrieve == 0:
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
    num_classes = 100
    if retrieve == 0:
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
    
    database = torchvision.datasets.CIFAR100(root = './data',
                                        train = True,
                                        download = True,
                                        transform = data_transform['train'])
    
elif dataset == 'coco':
    if retrieve == 0:
        retrieve = 5000
    num_classes = 80
    trainset = ImageList(open('./data/coco/train.txt', 'r').readlines(), transform = data_transform['train'])
    if based_method == 'pair':
        trainset_ = ImageList(open('./data/coco/train.txt', 'r').readlines(), transform = data_transform['train'])
    testset = ImageList(open('./data/coco/test.txt', 'r').readlines(), transform = data_transform['val'])
    database = ImageList(open('./data/coco/database.txt', 'r').readlines(), transform = data_transform['val'])

elif dataset == 'imagenet':
    if retrieve == 0:
        retrieve = 1000
    num_classes = 100
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
                            num_workers = 8)

if based_method == 'pair':
    trainloader_ = data.DataLoader(dataset = trainset_,
                            batch_size = batch_size,
                            shuffle = True,
                            num_workers = 8)

testloader = data.DataLoader(dataset = testset,
                            batch_size = batch_size,
                            shuffle = False,
                            num_workers = 8)

databaseloader = data.DataLoader(dataset = database,
                            batch_size = batch_size,
                            shuffle = False,
                            num_workers = 8)

# find the value of ζ
if HHF_flag:
    sheet = xlrd.open_workbook('codetable.xlsx').sheet_by_index(0)
    threshold = sheet.row(num_bits)[math.ceil(math.log(num_classes, 2))].value
    print('threshold:', threshold)

print('------------- data prepared -------------')

