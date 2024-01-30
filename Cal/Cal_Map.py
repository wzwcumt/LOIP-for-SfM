import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os
import sys
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import cdist
import timm
# 将pvt.py所在的目录添加到sys.path
sys.path.append('D:\\WZW\\ROT_RETRIEVAL\\SWIN_TRAIN\\train_gem\\PVT-2\\classification')
# 现在可以从pvt中导入pvt_small
from pvt import pvt_small
p = 2.9208
eps = 1e-6


# 读取底库图像顺序信息
data_file = 'D:/WZW/ROT_RETRIEVAL/SWIN_TRAIN/三元组数据-random-train/Validate数据/Validate_Database.txt'
with open(data_file, 'r') as f:
    image_data = f.read().splitlines()
# 读取待查询图像顺序信息
query_file = 'D:/WZW/ROT_RETRIEVAL/SWIN_TRAIN/三元组数据-random-train/Validate数据/Validate_Query.txt'
with open(query_file, 'r') as f:
    image_query = f.read().splitlines()

# 加载预训练的VGG16模型
model = models.vgg16(pretrained=True).features
model.load_state_dict(torch.load('D:\\WZW\\weights\\new_vgg_weights.pth')['state_dict'])

device = torch.device("cuda")
model = model.to(device)
model.eval()

# 预处理图像的转换
preprocess = transforms.Compose([
    #transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

data_features = []
wzw = 0
#提取每个图像的特征并保存
for img_name in image_data:
    img_path = os.path.join('D:/WZW/TraiDataXYW/alldata_rot_224', img_name)  # 替换为你的图像文件夹路径
    img = Image.open(img_path).convert('RGB')
    img = preprocess(img)
    img = img.unsqueeze(0)  # 添加一个维度作为 batch
    img = img.to(device)

    with torch.no_grad():
        output = model(img)
        # output = output.permute(0, 3, 1, 2)
        # output = F.avg_pool2d(output.clamp(min=eps).pow(p),
        #                            (output.size(-2), output.size(-1))).pow(1. / p)
        output = torch.squeeze(output)
        output = F.normalize(output, p=2, dim=0)

        output = output.cpu().detach().numpy()
        data_features.append(output)
    wzw += 1
    print(wzw)
query_features = []
tax = 0
# 提取每个图像的特征并保存
for img_name in image_query:
    img_path = os.path.join('D:/WZW/TraiDataXYW/alldata_rot_224', img_name)  # 替换为你的图像文件夹路径
    img = Image.open(img_path).convert('RGB')
    img = preprocess(img)
    img = img.unsqueeze(0)  # 添加一个维度作为 batch
    img = img.to(device)
    with torch.no_grad():
        output = model(img)
        # output = output.permute(0, 3, 1, 2)
        output = F.avg_pool2d(output.clamp(min=eps).pow(p),
                              (output.size(-2), output.size(-1))).pow(1. / p)
        output = torch.squeeze(output)
        output = F.normalize(output, p=2, dim=0)

        output = output.cpu().detach().numpy()
        query_features.append(output)
    tax += 1
    print(tax)
print("特征提取好了！")

data_features = np.stack(data_features)  # 将特征列表转换为二维数组

query_features = np.stack(query_features)  # 将特征列表转换为二维数组
print(query_features.shape)
# 进行欧式距离计算
distances = cdist(query_features, data_features, 'euclidean')

# 对每一行距离进行排序，得到索引
sorted_indices = np.argsort(distances, axis=1)

# 获取距离较小的前100个图像特征的索引
top_100_indices = sorted_indices[:, 1:101]
print(top_100_indices)
np.savetxt('top_100_indices.txt', top_100_indices, fmt='%d')

# 读取 txt 文件
with open('D:/WZW/ROT_RETRIEVAL/SWIN_TRAIN/三元组数据-random-train/Validate数据/Validate_TopN.txt', 'r') as file:
    lines = file.readlines()

# 从 txt 文件中获取 ground truth 的索引
gd_100 = []
for line in lines:
    indices = line.strip().split('\t')
    gd_100.append([int(idx) for idx in indices])

gd_100 = np.array(gd_100)

# 计算 recall@1
recall_1 = np.mean(np.array([1 if top_100_indices[i, 0] in gd_100[i, :99] else 0 for i in range(len(top_100_indices))]))

# 计算 recall@5
recall_5 = np.mean(np.array([1 if np.any(np.isin(top_100_indices[i, :5], gd_100[i, :99])) else 0 for i in range(len(top_100_indices))]))

# 计算 recall@10
recall_10 = np.mean(np.array([1 if np.any(np.isin(top_100_indices[i, :10], gd_100[i, :99])) else 0 for i in range(len(top_100_indices))]))

# print(f"Recall@1: {recall_1}")
# print(f"Recall@5: {recall_5}")
# print(f"Recall@10: {recall_10}")


# 计算 Precision@K
def calculate_map_at_k(top_indices, ground_truth, k):
    scores = []
    for i in range(top_indices.shape[0]):
        # 取 top_indices 的每行前 k 个数值
        top_k = set(top_indices[i, :k])
        # 检查这些数值在 ground_truth 的对应行中出现的次数
        hits = sum([1 for idx in top_k if idx in ground_truth[i]])
        # 计算得分
        score = hits / k
        scores.append(score)
    # 计算所有行得分的平均值
    return np.mean(scores)

# 使用示例
map_3 = calculate_map_at_k(top_100_indices, gd_100, 3)
map_5 = calculate_map_at_k(top_100_indices, gd_100, 5)
map_10 = calculate_map_at_k(top_100_indices, gd_100, 10)
map_20 = calculate_map_at_k(top_100_indices, gd_100, 20)
print("map@3:", map_3)
print("map@5:", map_5)
print("map@10:", map_10)
print("map@20:", map_20)