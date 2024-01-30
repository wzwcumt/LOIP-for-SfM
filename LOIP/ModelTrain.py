from numpy import mean
import argparse
import math
import os
import sys

from math import log10, ceil
import random, shutil, json
from multiprocessing import pool
from os.path import join, exists, isfile, realpath, dirname
from os import makedirs, remove, chdir, environ
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.data.dataset import Subset
import torchvision.transforms as transforms
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # 或者设置为一个更高的值
from datetime import datetime
import torchvision.datasets as datasets
import torchvision.models as models
import h5py
#import faiss
from tensorboardX import SummaryWriter
import numpy as np
from joblib import Parallel, delayed
import time
import multiprocessing
import timm
import dataset
import NetVlad
#from ResNet import resnet101

threads          = 8            # 每个数据加载器使用的线程数
randomSeed       = 42
batchSize        = 40           # 三元组的数量 (query, pos, negs). 每个三元组包含12张影像，4个批次.
cacheBatchSize   = 64          # 测试和cache时的batchSize
cacheRefreshRate = 0            # 多久刷新一次cache, in number of queries. 0 for off
margin           = 2.5          # Margin for triplet loss. Default=0.1, try 2.5
nGPU             = 1            # 参与训练的GPU个数
LearnRate        = 5e-7         # 学习率
LearnRateStep    = 3            # Decay LR ever N steps
LearnRateGamma   = math.exp(-0.01)  # Multiply LR by Gamma for decaying
momentum         = 0.9          # Momentum for SGD
weightDecay      = 1e-6         # Weight decays for SGD
nEpochs          = 30           # number of epochs to train for
evalEvery        = 1            # 每完成多少个 epoch 运行一个验证集并保存
patience         = 0           # Patience for early stopping. 0 is off
RamdonNum        = 50000
NewWidth         = 224
NewHeight        = 224
IsSplit       = False
num_clusters  = 64
p = 2.9208
eps = 1e-6

DatasetPath = "D:/WZW/TraiDataXYW/DownSample_alldata"  #"D:/WZW/ROT_RETRIEVAL/SWIN_TRAIN/train_2/Images_all"
TrainDataPath = "D:/WZW/ROT_RETRIEVAL/SWIN_TRAIN/三元组数据-random-train/Train数据"  #"D:/WZW/ROT_RETRIEVAL/SWIN_TRAIN/train_2/三元组/Train数据"
ValidateDataPath  = "D:/WZW/ROT_RETRIEVAL/SWIN_TRAIN/三元组数据-random-train/Validate数据"  #"D:/WZW/ROT_RETRIEVAL/SWIN_TRAIN/train_2/三元组/Validate数据"
runsPath     = "D:/WZW/ROT_RETRIEVAL/SWIN_TRAIN/train_gem/runs"
hdf5Path     = "D:/WZW/ROT_RETRIEVAL/SWIN_TRAIN/ModelTrain/centroids/VGG16_64_desc_cen.hdf5"
#cachePath    = "D:/Retrieval/ModelTrain/cache/"


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, input):
        return F.normalize(input, p=2, dim=self.dim)

def TrainOneEpoch(epoch):
    T1 = time.perf_counter()
    epoch_loss = 0
    startIter = 0
    nBatches = (len(TrainData) + batchSize - 1) // batchSize  # 有多少个batch
    print("Batch数: ", nBatches)

    model.train()
    TrainData.GetDataType = "Triplet"
    TrainDataLoader = DataLoader(dataset=TrainData, num_workers=threads, batch_size=batchSize, shuffle=True, collate_fn=dataset.collate_fn, pin_memory=True)
    for iteration, (query, positives, negatives, index) in enumerate(TrainDataLoader, startIter): #取一个batch的三元组
        # some reshaping to put query, pos, negs in a single (N, 3, H, W) tensor
        # where N = batchSize * (nQuery + nPos + nNeg)
        if query is None: continue  # in case we get an empty batch
        B, C, H, W = query.shape
        query = query.to(device)
        positives = positives.to(device)
        negatives = negatives.to(device)
        query_encoding = model(query)  # query送入vgg/Resnet...
        positives_encoding = model(positives)
        negatives_encoding = model(negatives)

        vladQ = query_encoding
        vladP = positives_encoding
        vladN = negatives_encoding

        # vladQ = F.avg_pool2d(query_encoding.clamp(min=eps).pow(p), (query_encoding.size(-2), query_encoding.size(-1))).pow(1./p)
        # vladP = F.avg_pool2d(positives_encoding.clamp(min=eps).pow(p), (positives_encoding.size(-2), positives_encoding.size(-1))).pow(1./p)
        # vladN = F.avg_pool2d(negatives_encoding.clamp(min=eps).pow(p), (negatives_encoding.size(-2), negatives_encoding.size(-1))).pow(1./p)

        vladQ = torch.squeeze(vladQ)
        vladQ = F.normalize(vladQ, p=2, dim=0)
        vladP = torch.squeeze(vladP)
        vladP = F.normalize(vladP, p=2, dim=0)
        vladN = torch.squeeze(vladN)
        vladN = F.normalize(vladN, p=2, dim=0)

        optimizer.zero_grad()

        loss = 0
        loss = criterion(vladQ, vladP, vladN)
        loss = loss / vladQ.size(0)
        loss = loss.to(device)
        # for param in model.features.parameters():
        #     param.requires_grad = False
        loss.backward()
        optimizer.step()
        del vladQ, vladP, vladN, query, positives, negatives

        Iteration_loss = loss.item()
        epoch_loss += Iteration_loss
        #if iteration % 100 == 0 or nBatches <= 10:
        print("==> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch+1, iteration, nBatches, Iteration_loss), flush=True)
    torch.cuda.empty_cache()
    TrainData.GetDataType = 'None'
    avg_loss = epoch_loss / nBatches
    T2 = time.perf_counter()
    TimeConsuming = (T2 - T1)  / 60.0
    print("===> ["+ str(round(TimeConsuming, 1))+ " min] 第 " + str(epoch+1) + " 个Epoch已完成! 平均Loss: " + str(round(avg_loss, 4)))
    writer.add_scalar('Train/AvgLoss', avg_loss, epoch+1)
    return avg_loss

def Validate(TestData, epoch=0, write_tboard=False):
    T1 = time.perf_counter()
    model.eval()
    NQuery = len(TestData.Query)
    NDatabase = len(TestData.TestDatabase)

    DbImageFeat = torch.zeros(0).to(device)
    QueryFeat = torch.zeros(0).to(device)

    TestData.GetDataType = 'Database'
    Databaseloader = DataLoader(dataset=TestData, num_workers=threads, batch_size=cacheBatchSize, shuffle=False, pin_memory=True)
    with torch.no_grad():
        for iteration, (DbImg, Index) in enumerate(Databaseloader, 1):
            torch.cuda.empty_cache()
            DbImg = DbImg.to(device)
            DbImg_Encoding = model(DbImg)
            #DbImg_Encoding = F.avg_pool2d(DbImg_Encoding.clamp(min=eps).pow(p), (DbImg_Encoding.size(-2), DbImg_Encoding.size(-1))).pow(1./p)
            DbImg_Encoding = torch.squeeze(DbImg_Encoding)
            DbImg_Encoding = F.normalize(DbImg_Encoding, p=2, dim=0)
            DbImageFeat = torch.cat((DbImageFeat, DbImg_Encoding), 0)
            del DbImg, Index

    TestData.GetDataType = 'TestQuery'
    Databaseloader = DataLoader(dataset=TestData, num_workers=threads, batch_size=cacheBatchSize, shuffle=False, pin_memory=True)
    with torch.no_grad():
        for iteration, (QueryImg, Index) in enumerate(Databaseloader, 1):
            torch.cuda.empty_cache()
            QueryImg = QueryImg.to(device)
            QueryImg_Encoding = model(QueryImg)
            #QueryImg_Encoding = F.avg_pool2d(QueryImg_Encoding.clamp(min=eps).pow(p), (QueryImg_Encoding.size(-2), QueryImg_Encoding.size(-1))).pow(1./p)
            QueryImg_Encoding = torch.squeeze(QueryImg_Encoding)
            QueryImg_Encoding = F.normalize(QueryImg_Encoding, p=2, dim=0)
            QueryFeat = torch.cat((QueryFeat, QueryImg_Encoding), 0)
            del QueryImg, Index

    DistanceMatrix = torch.cdist(QueryFeat, DbImageFeat)
    MinDis, PredictTop101Index = torch.topk(DistanceMatrix, 101, dim=1, largest=False)
    PredictTop100Index = PredictTop101Index[:, 1:]
    TrueTop100 = TestData.TopN
    MeanTopN = []

    for i in range(NQuery):
        ThisPredict = PredictTop100Index[i]
        #ThisPredict.remove(i)
        ThisTrue = TrueTop100[i]
        TopN = []
        N = []
        for j in range(len(ThisPredict)):
            if ThisPredict[j].item() in ThisTrue:
                if len(TopN) == 0:
                    TopN.append(1.0 / (j + 1))
                    N.append(j + 1)
                else:
                    TopN.append((TopN[-1] * N[-1] + 1) / (j + 1))
                    N.append(j + 1)
        if len(TopN) == 0:
            TopN = [0]
        MeanTopN.append(mean(TopN))
    mAP = mean(MeanTopN)

    N_values = [1, 5, 10]
    Recall = [0] * len(N_values)  # 查全率
    for N_Index in range(len(N_values)):
        N = N_values[N_Index]
        RecallN = [0] * NQuery
        for QueryIndex in range(NQuery):
            PredictTopN = PredictTop100Index[QueryIndex][:N]
            TrueTopN = TrueTop100[QueryIndex]
            RecallN[QueryIndex] = np.sum(np.in1d(PredictTopN.cpu(), TrueTopN)) * 1.0 / N
        Recall[N_Index] = mean(RecallN)
    T2 = time.perf_counter()
    TimeConsuming = (T2 - T1) / 60.0
    print("===============================================")
    print("===> [" + str(round(TimeConsuming, 1)) + " min] mAP: " + str(round(mAP, 4)))
    for N_Index in range(len(N_values)):
        N = N_values[N_Index]
        print("Top " + str(N) + ": " + str(round(Recall[N_Index], 4)))
    torch.cuda.empty_cache()
    print("===============================================")
    return mAP, Recall[2]



def save_checkpoint(state, is_best, filename):
    model_out_path = join(savePath, filename)
    torch.save(state, model_out_path)
    if is_best:
        shutil.copyfile(model_out_path, join(savePath, 'model_best.pth.tar'))

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("GPU可用")
    else:
        device = torch.device("cpu")
        print("未找到GPU，将使用CPU")
    print("Encoder:", EncoderType,)
    device = torch.device("cuda")

    # 设置种子
    random.seed()
    np.random.seed()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    print('加载数据...')
    TrainData = dataset.Dataset(TrainDataPath, DatasetPath, True, True, RamdonNum)
    TrainDataLoader = DataLoader(dataset=TrainData, num_workers=threads, batch_size=cacheBatchSize, shuffle=False, pin_memory=True)

    TestData = dataset.Dataset(ValidateDataPath, DatasetPath, False, False, 50)
    TestDataLoader = DataLoader(dataset=TestData, num_workers=threads, batch_size=cacheBatchSize, shuffle=False, pin_memory=True)
    #print('原三元组数量:', len(TrainData.AllQuery))
    print('筛选后三元组数量:', len(TrainData))
    print('数据库影像数量:', len(TrainData.Database))
    print('构建模型...')
    # 加载预训练的 VGG16 模型
    vgg_model = models.vgg16(pretrained=True)

    # 获取 VGG16 的卷积层
    features = vgg_model.features

    # 移除最后的全连接层，仅保留卷积层
    conv_base = nn.Sequential(*list(features.children()))


    # 添加全局平均池化层(GEM)
    class GemPool(nn.Module):
        def __init__(self, p=3, eps=1e-6):
            super(GemPool, self).__init__()
            self.p = nn.Parameter(torch.Tensor([p]))  # 将p定义为可训练参数
            self.eps = eps

        def forward(self, x):
            # 进行Gem层的池化操作
            return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1. / self.p)


    gem_pooling = GemPool()

    # 构建新的模型
    model = nn.Sequential(
        conv_base,
        gem_pooling
    )

    # 将模型设置为评估模式
    model.eval()

    # if EncoderType == "Alexnet":
    #     encoder_dim = 256
    #     encoder = models.alexnet(pretrained=True)
    #     layers = list(encoder.features.children())
    #     for l in layers:
    #         for p in l.parameters():
    #             p.requires_grad = True
    #     encoder = nn.Sequential(*layers)
    #     model = nn.Module()
    #     model.add_module('encoder', encoder)
    # elif EncoderType == "VGG16":
    #     encoder_dim = 512
    #     encoder = models.vgg16(pretrained=True)
    #     layers = list(encoder.features.children())
    #     for l in layers:
    #         for p in l.parameters():
    #             p.requires_grad = True
    #     encoder = nn.Sequential(*layers)
    #     model = nn.Module()
    #     model.add_module('encoder', encoder)
        # model = models.vgg16(pretrained=False).features
        # model.load_state_dict(torch.load('D:/WZW/weights/vgg_conv_weights.pth'))
    # elif EncoderType == "Resnet":
    #     encoder_dim = 2048
    #     model = resnet101(pretrained=True)
    #     layers = list(model.children())

    # if PoolingType == "MaxPooling":
    #     max = nn.AdaptiveMaxPool2d((4, 4))
    #     model.add_module('pool', nn.Sequential(*[max, Flatten(), L2Norm()]))
    # elif PoolingType == "Gem":
    #     p = 3
    #     eps = 1e-6
    #     model.add_module("GAP",  F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1./p))
    # elif PoolingType == "NetVlad":
    #     net_vlad = NetVlad.NetVLAD(num_clusters=num_clusters, dim=encoder_dim, vladv2=True)
    #     if not os.path.exists(hdf5Path):
    #         raise FileNotFoundError("找不到hdf5文件,先运行cluster.py!")
    #     with h5py.File(hdf5Path, mode='r') as h5:
    #         clsts = h5.get("centroids")[...]
    #         traindescs = h5.get("descriptors")[...]
    #         net_vlad.init_params(clsts, traindescs)
    #         del clsts, traindescs
    #     model.add_module('pool', net_vlad)
    # if PoolingType == "GemPooling":
    #     max = nn.AdaptiveMaxPool2d((4, 4))
    #     model.add_module('pool', nn.Sequential(*[max, Flatten(), L2Norm()]))

    # vgg16 = torchvision.models.vgg16(pretrained=True)
    # #冻结 VGG 模型的参数
    # for param in vgg16.parameters():
    #     param.requires_grad = False
    # # 替换 VGG 的全连接层
    # num_features = vgg16.classifier[6].in_features
    # vgg16.classifier[6] = nn.Linear(num_features, 4096)
    # model = vgg16

    model = model.to(device)
    model.train()
    print("网络结构:")
    print(model)

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=LearnRate, momentum=momentum, weight_decay=weightDecay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LearnRateStep, gamma=LearnRateGamma)
    criterion = nn.TripletMarginLoss(margin=margin, p=2, reduction='sum').to(device)

    print('训练模型...')
    not_improved = 0
    best_score = 0


    OutputFile = EncoderType + ".txt"
    Output = open(OutputFile, mode='w+')
    Output.write(EncoderType + "\n")
    writer = SummaryWriter(log_dir=join(runsPath, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + EncoderType))
    logdir = writer.file_writer.get_logdir()
    savePath = join(logdir, "checkpoints")
    if os.path.exists(savePath):
        shutil.rmtree(savePath)
    makedirs(savePath)

    BestRecall20 = 0
    for epoch in range(nEpochs):
        scheduler.step(epoch)
        AveLoss = TrainOneEpoch(epoch)
        mAP, Recall20 = Validate(TestData, epoch, write_tboard=True)
        if Recall20 > BestRecall20:
            BestRecall20 = Recall20
            IsBestFlag = True
        else:
            IsBestFlag = False

        #IsBestFlag, Condition = IsBest(TopN, BestTop10, BestTop5, BestTop1)
        Output.write(str(epoch) + "\t" + str(AveLoss) + "\t" + str(Recall20) + "\n")
        CheckPointFile = "CheckPoint_" + EncoderType + "_" + str(epoch+1) + ".pth.tar"
        save_checkpoint({'epoch': epoch+1, 'state_dict': model.state_dict(), 'mAP': mAP, 'best_score': BestRecall20, 'optimizer': optimizer.state_dict(), }, IsBestFlag, CheckPointFile)
