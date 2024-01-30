import copy
import heapq
import os
from glob import glob
import random

import joblib
import numpy as np
import warnings
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import NetVlad
import h5py
from treelib import Tree
import time

def ParseVocTree(VocTreePath):
    f = open(VocTreePath)
    VocTree = Tree()
    for line in f:
        NodeInfoList = line.strip().split(',')
        NodeID = NodeInfoList[0]
        NodeData = np.asarray(NodeInfoList[1].split(';')).astype(float)
        ChildrenIDs = []
        if len(NodeInfoList[2]) != 0:
            ChildrenIDs = NodeInfoList[2].split(';')
        if not VocTree.contains(NodeID):
            VocTree.create_node(identifier=NodeID, data=NodeData)
        else:
            VocTree.update_node(NodeID, data=NodeData)
        for ChildID in ChildrenIDs:
            VocTree.create_node(identifier=ChildID, data=None, parent=NodeID)
    return VocTree

def GetMinDist(data, tree, parent):
    ChildrenList = tree.children(parent)
    ChildrenDist = [0] * len(ChildrenList)
    for ChildIndex in range(len(ChildrenList)):
        ChildrenDist[ChildIndex] = np.linalg.norm(data - ChildrenList[ChildIndex].data)
    MinIndex = ChildrenDist.index(min(ChildrenDist))
    return ChildrenList[MinIndex].identifier


# 获取tree的最小距离路径
def GetMinDistPath(data, tree):
    MinDistPath = [tree.root]  # 树的根节点一定在最小距离路径中
    # 如果tree的根节点就是叶子节点, 那么直接返回当前的最小距离路径(只有根节点)
    if tree.get_node(tree.root).is_leaf():
        return MinDistPath
    # 开始递归
    TempParent = tree.root
    while True:
        MinDistId = GetMinDist(data, tree, TempParent)
        MinDistPath.append(MinDistId)  # 加入最小距离路径
        if tree.get_node(MinDistId).is_leaf():  # 如果已经递归到叶子节点, 返回
            return MinDistPath
        TempParent = MinDistId

# 在tree中, 获取Id号节点的所有兄弟节点的所有叶子节点
def GetBrotherLeaves(tree, Id):
    BrotherLeaves = []
    BrotherList = tree.siblings(Id)  # BrotherList为Id号节点的所有兄弟节点
    for BrotherId in BrotherList:  # 依次找每一个兄弟节点的叶子节点
        BrotherLeaves = BrotherLeaves + tree.subtree(BrotherId.identifier).leaves()
    return BrotherLeaves


class RetrievalMatchClass:
    def __init__(self, PthPath, HDF5Path, CheckPointPath, PCA_Model, VocTreePath):
        random.seed()
        np.random.seed()
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        self.num_clusters = 64
        self.encoder_dim = 512
        self.NewWidth = 224
        self.NewHeight = 224
        margin = 1
        LearnRate = 0.0001
        LearnRateStep = 5
        LearnRateGamma = 0.5
        momentum = 0.9
        weightDecay = 0.001
        self.PthPath = PthPath
        self.HDF5Path = HDF5Path
        self.CheckPointPath = CheckPointPath
        self.device = torch.device("cuda")
        self.DatabaseFeature = None
        self.pca = joblib.load(PCA_Model)
        self.ImageID2ImageName = {}
        self.VocTree = None
        self.LeafID2ImageID = {}  # 叶子ID -> 影像序号
        self.ImageNum = 0  # 影像总数
        if (VocTreePath is not None) and len(VocTreePath) > 0:
            self.VocTree = ParseVocTree(VocTreePath)
        warnings.filterwarnings("ignore", category=UserWarning)
        self.encoder = models.vgg16(pretrained=False)
        self.encoder.load_state_dict(torch.load(self.PthPath))
        layers = list(self.encoder.features.children())[:-2]
        self.encoder = nn.Sequential(*layers)
        self.model = nn.Module()
        self.model.add_module('encoder', self.encoder)
        net_vlad = NetVlad.NetVLAD(num_clusters=self.num_clusters, dim=self.encoder_dim, vladv2=False)
        with h5py.File(self.HDF5Path, mode='r') as h5:
            clsts = h5.get("centroids")[...]
            traindescs = h5.get("descriptors")[...]
            net_vlad.init_params(clsts, traindescs)
            del clsts, traindescs
        self.model.add_module('pool', net_vlad)
        self.model.encoder = nn.DataParallel(self.model.encoder)
        self.model.pool = nn.DataParallel(self.model.pool)
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=LearnRate,
                              momentum=momentum, weight_decay=weightDecay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LearnRateStep, gamma=LearnRateGamma)
        criterion = nn.TripletMarginLoss(margin=margin, p=2, reduction='sum').to(self.device)
        checkpoint = torch.load(self.CheckPointPath, map_location=lambda storage, loc: storage)
        TestEpoch = checkpoint['epoch']
        BestMetric = checkpoint['best_score']
        self.model.load_state_dict(checkpoint['state_dict'], strict=False)
        self.model = self.model.to(self.device)

    def Match(self, ImagePath, N):
        start_time = time.time()
        ImagePath = ImagePath.replace("\\", "/")
        image = Image.open(ImagePath)
        try:
            if len(image.split()) != 3:
                image = image.convert('RGB')
        except:
            return [], 0, 0
        image = image.resize((self.NewWidth, self.NewHeight))
        image = transforms.Compose([transforms.ToTensor()])(image).to(self.device)
        self.model.eval()
        with torch.no_grad():
            Feature = self.model.encoder(image)  # 经过VGG16
            Feature = Feature.unsqueeze(0)
            PoolingFeature = self.model.pool(Feature)  # 经过NetVlad
            PoolingFeature = self.pca.transform(PoolingFeature.to(torch.device("cpu")))  # tensor转numpy, 再做PCA降维
            CurrentImageID = self.ImageNum  # 当前影像的序号
            self.ImageID2ImageName[CurrentImageID] = os.path.basename(ImagePath)
            self.ImageNum = self.ImageNum + 1

            GlobalFeatureExtraction_Time = (time.time() - start_time) * 1000
            start_time = time.time()

            if self.VocTree is None:  # 用两两匹配
                PoolingFeature = torch.from_numpy(PoolingFeature).to(self.device)
                TopNIndex = []
                if self.DatabaseFeature is not None:
                    Database = self.DatabaseFeature.to(self.device)

                    PoolingFeature_np = PoolingFeature.cpu().numpy()[0]
                    Database_np = Database.cpu().numpy()
                    distances = []
                    for i in range(Database_np.shape[0]):
                        sum_of_squared_differences = sum((PoolingFeature_np[j] - Database_np[i, j]) ** 2 for j in range(Database_np.shape[1]))
                        distance = sum_of_squared_differences ** 0.5
                        distances.append(distance)
                    distances_np = np.array(distances)
                    TopNIndex = np.argsort(distances_np)[:N]
                ThisImageFeature = PoolingFeature
                if self.DatabaseFeature is None:
                    self.DatabaseFeature = ThisImageFeature.to(torch.device("cpu"))
                else:
                    self.DatabaseFeature = torch.cat((self.DatabaseFeature, ThisImageFeature.to(torch.device("cpu"))), 0)
                GlobalFeatureRetrieval_Time = (time.time() - start_time) * 1000
                return TopNIndex, round(GlobalFeatureExtraction_Time), round(GlobalFeatureRetrieval_Time)
            else:
                if self.DatabaseFeature is None:   #当前特征加入数据库
                    self.DatabaseFeature = [PoolingFeature.squeeze()]
                else:
                    self.DatabaseFeature.extend(PoolingFeature)
                MinDistPath = GetMinDistPath(PoolingFeature, self.VocTree)[::-1]  # 计算关键路径
                TopNIndex = []
                if MinDistPath[0] not in self.LeafID2ImageID:  # 影像所属的叶子节点还没有任何影像
                    self.LeafID2ImageID[MinDistPath[0]] = [CurrentImageID]  # 把当前影像ID加入该叶子节点中
                else:
                    TopNIndex = copy.deepcopy(self.LeafID2ImageID[MinDistPath[0]])  # 取出影像所属的叶子节点下的当前所有影像ID(不包含当前影像的ID)
                    self.LeafID2ImageID[MinDistPath[0]].append(CurrentImageID)  # 再把当前影像ID加入该叶子节点
                if len(TopNIndex) < N:
                    for BrotherNode in self.VocTree.siblings(MinDistPath[0]): #找所有的兄弟节点
                        BrotherID = BrotherNode.identifier
                        if BrotherID in self.LeafID2ImageID:
                            TopNIndex.extend(copy.deepcopy(self.LeafID2ImageID[BrotherID])) #把所有兄弟节点的所有影像ID加入
                    for i in range(1, len(MinDistPath)): #从叶子节点的父节点开始往上遍历关键路径(直到根节点), MinDistPath[i](i>0)全为非叶子节点
                        if len(TopNIndex) >= N:  #影像数已经够了, 不再继续遍历
                            break
                        MoreLeavesID = GetBrotherLeaves(self.VocTree, MinDistPath[i]) #找关键路径的当前非叶子节点的所有兄弟节点的叶子节点
                        for NodeID in MoreLeavesID:
                            if NodeID.identifier in self.LeafID2ImageID:
                                TopNIndex.extend(copy.deepcopy(self.LeafID2ImageID[NodeID.identifier])) #将关键路径的当前非叶子节点的所有兄弟节点的叶子节点中的影像ID加入
                AllDistance = [0] * len(TopNIndex)  # 计算得到的所有影像与当前影像的距离
                for i in range(len(TopNIndex)):
                    AllDistance[i] = np.linalg.norm(PoolingFeature - self.DatabaseFeature[TopNIndex[i]])
                TopNIndex_Index = heapq.nsmallest(N, range(len(AllDistance)), AllDistance.__getitem__)  # 按照距离排序取前N个, TopNIndex_Index是TopNIndex的索引(索引的索引)
                TopNIndex_Sorted = [0] * len(TopNIndex_Index)
                for i in range(len(TopNIndex_Index)): # TopNIndex_Index[0]对应的TopNIndex距离最小, TopNIndex_Index[1]对应的TopNIndex距离第二小...
                    TopNIndex_Sorted[i] = TopNIndex[TopNIndex_Index[i]]
                GlobalFeatureRetrieval_Time = (time.time() - start_time) * 1000
                return TopNIndex_Sorted, round(GlobalFeatureExtraction_Time), round(GlobalFeatureRetrieval_Time)

def ReadGroundTruthFile(groundTruthPath, nImages):
    with open(groundTruthPath, 'r') as file:
        lines = file.readlines()
    GT_TopN = []
    for line in lines:
        if len(GT_TopN) >= nImages:
            break
        # 使用空格分割字符串，然后转换每个元素为数字
        numbersInLine = [int(num) for num in line.split()]
        # 将转换后的数字列表添加到主列表中
        GT_TopN.append(numbersInLine)
    return GT_TopN

def CalculateMeanRecall(topN, GT_TopN):
    nImages = len(topN)
    recall = [1] * nImages
    for i in range(nImages):
        TopN_Set = set(topN[i])
        count = sum(1 for elem in GT_TopN[i] if elem in TopN_Set)
        if count > 0:
            recall[i] = count / len(GT_TopN[i])
    meanRecall = sum(recall) / nImages
    return meanRecall, recall

PthPath = "D:/Code/RTPS/RTPS Models/Network/vgg16-397923af.pth"
HDF5Path = "D:/Code/RTPS/RTPS Models/Network/Aerial/NoSplit/VGG16_64_desc_cen.hdf5"
CheckPointPath = "D:/Code/RTPS/RTPS Models/Network/Aerial/NoSplit/VGG16_NetVlad_NoSplit.pth.tar"
PCA_Model = "D:/Code/RTPS/RTPS Models/PCA/PCA_dims32768to2048.model"
VocTreePath = "D:/Code/RTPS/RTPS Models/2048_8x8.tree"
#VocTreePath = ""
directory_path = "D:/Dataset/WZW/GL3D1/images"
groundTruthPath = None
TopN = 30

RetrievalMatch = RetrievalMatchClass(PthPath, HDF5Path, CheckPointPath, PCA_Model, VocTreePath)
image_paths = []
RetrievalResult = []
Database = []
AllTimes = []
for extension in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
    image_paths.extend([os.path.abspath(path) for path in glob(os.path.join(directory_path, extension))])
nImages = len(image_paths)
print("影像数:" + str(nImages))


GT_TopN = None
if groundTruthPath is not None and len(groundTruthPath) > 0:
    GT_TopN = ReadGroundTruthFile(groundTruthPath, nImages)
    if len(GT_TopN) < len(image_paths):
        raise ValueError("TopN的真值貌似不正确!")


for i in range(len(image_paths)):
    print(f'\r{i + 1}/{len(image_paths)}', end='')
    image_path = image_paths[i]
    Database.append(image_path)

    start_time = time.time()
    TopNIndex, GlobalFeatureExtraction_Time, GlobalFeatureRetrieval_Time = RetrievalMatch.Match(image_path, TopN)
    ThisTime = (time.time() - start_time) * 1000
    RetrievalResult.append(TopNIndex)
    AllTimes.append(GlobalFeatureRetrieval_Time)
print("\n")

if GT_TopN is not None:
    meanRecall, recall = CalculateMeanRecall(RetrievalResult, GT_TopN)
    print("Recall=" + str(meanRecall))

print("输出...")

with open('RetrievalResult_VocTree_Ph1.txt', 'w') as f:
    for row in RetrievalResult:
        f.write(' '.join(map(str, row)) + '\n')
with open('Database_Ph1.txt', 'w') as f:
    for item in Database:
        f.write(item + '\n')
with open('AllTimes_VocTree_Ph1.txt', 'w') as f:
    for item in AllTimes:
        f.write(str(item) + '\n')
print("完成!")
