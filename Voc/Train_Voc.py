import os
import shutil
import time
import joblib
import pickle
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from treelib import Tree, Node
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# config
vis = True
# vis = False
vis_row = 4
before_node = []
dis_all = []
name_all = []

# dis_all = torch.tensor(dis_all)
norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]
k = 7  # 每层节点数
l = 7  # 树深度
N = 20  # TOP-"N"
inference_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])


# classes = ["ants", "bees"]
class NodeData:
    def __init__(self, center):
        self.center = center
        self.name = None
        self.data = None

    def init_leaf_node(self, name, data):
        self.name = name
        self.data = data


# 指定query_txt的路径，test_pt的路径、目标文件夹路径，将待查询图像pt移至目标文件夹
def mk_query_dir(txt_path, pt_path, des_path):
    query_list = []
    files = []
    with open(txt_path, 'r') as f:
        for line in f:
            query_list.append(line.strip().split('.jpg')[0] + '.pt')
            #query_list.append(line.strip().split('.jpg')[0] + '.pt')
    for file in query_list:
        src = os.path.join(pt_path, file)
        dst = os.path.join(des_path, file)
        shutil.move(src, dst)

# 指定query_txt的路径，test_pt的路径、目标文件夹路径，将待查询图像移至目标文件夹
def mk_query_dir_pic(txt_path, pt_path, des_path):
    query_list = []
    files = []
    with open(txt_path, 'r') as f:
        for line in f:
            query_list.append(line.strip().split(' ')[0])
    for file in query_list:
        src = os.path.join(pt_path, file)
        dst = os.path.join(des_path, file)
        shutil.move(src, dst)

def transform_img(img_rgb, transform=None):
    """
    将数据转换为模型读取的形式
    :param img_rgb: PIL Image
    :param transform: torchvision.transform
    :return: tensor
    """

    if transform is None:
        raise ValueError("找不到transform！必须有transform对img进行处理")

    img_t = transform(img_rgb)
    return img_t


def get_img_name(img_dir, format="jpg"):
    """
    获取文件夹下format格式的文件名
    :param img_dir: str
    :param format: str
    :return: list
    """
    file_names = os.listdir(img_dir)
    # 使用 list(filter(lambda())) 筛选出 jpg 后缀的文件
    img_names = np.array(list(filter(lambda x: x.endswith(format), file_names)))

    if len(img_names) < 1:
        raise ValueError("{}下找不到{}格式数据".format(img_dir, format))
    return img_names


def get_model(vis_model=False):
    model = models.vgg16(pretrained=True)

    # 修改全连接层的输出
    model.classifier = torch.nn.Sequential()

    if vis_model:
        from torchsummary import summary
        summary(model, input_size=(3, 224, 224), device="cpu")

    return model


def get_feature(img_dir, pt_dir):
    time_total = 0
    img_list, img_pred = list(), list()

    # 1. data
    img_names = get_img_name(img_dir)
    num_img = len(img_names)

    # 2. model
    model = get_model(True)
    model.to(device)
    model.eval()

    # with torch.no_grad():
    for idx, img_name in enumerate(img_names):
        path_img = os.path.join(img_dir, img_name)

        img_rgb = Image.open(path_img).convert('RGB')

        img_tensor = transform_img(img_rgb, inference_transform)
        img_tensor.unsqueeze_(0)
        img_tensor = img_tensor.to(device)

        time_tic = time.time()
        outputs = model(img_tensor)
        time_toc = time.time()
        torch.save(outputs, pt_dir + "/" + img_name[:len(img_name) - 4] + ".pt")

        time_s = time_toc - time_tic
        time_total += time_s

        print('{:d}/{:d}: {} {:.3f}s '.format(idx + 1, num_img, img_name, time_s))

    print("\ndevice:{} total time:{:.1f}s mean:{:.3f}s".
          format(device, time_total, time_total / num_img))
    if torch.cuda.is_available():
        print("GPU name:{}".format(torch.cuda.get_device_name()))


def detach_pt(pt_dir):
    pt_names = get_img_name(pt_dir, "pt")
    num_img = len(pt_names)
    data = np.empty([num_img, len(torch.load(os.path.join(pt_dir, pt_names[0])).detach().cpu().numpy().flatten())])
    for idx, pt_name in enumerate(pt_names):
        data[idx] = torch.load(os.path.join(pt_dir, pt_name)).detach().cpu().numpy().flatten()
    return data


def get_sub_tree(tree, data, node, names, k, l, layer):
    if layer != l:
        layer += 1
        km = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=800, tol=0.0001, verbose=0, random_state=None,
                    copy_x=True, algorithm='auto')
        model = km.fit(data)
        pred = model.labels_
        for ind, clas in enumerate(np.unique(pred)):
            index_tmp = np.array(np.where(pred == clas)).ravel()
            names_sub = names[index_tmp]
            data_sub = data[index_tmp, :]
            node_data = NodeData((model.cluster_centers_)[ind].squeeze())
            if (len(data_sub) <= k) or (layer == l):
                node_data.init_leaf_node(names_sub, data_sub)

            node_sub = Node(data=node_data)
            tree.add_node(node_sub, parent=node.identifier)

            if len(data_sub) > k:
                get_sub_tree(tree, data_sub, node_sub, names_sub, k, l, layer)


def get_vocabulary_tree(data, pt_names, k, l):
    # 1. KMeans
    km = KMeans(n_clusters=8, init='k-means++', n_init=10, max_iter=300, tol=0.0001, verbose=0, random_state=None,
                copy_x=True, algorithm='auto')

    # 2. Tree
    time_tic = time.time()
    tree = Tree()
    model = km.fit(data)
    names = np.array(pt_names)
    node_data = NodeData(model.cluster_centers_.squeeze())
    node = Node(data=node_data)
    tree.add_node(node)

    get_sub_tree(tree, data, node, names, k, l, 1)
    joblib.dump(filename='D:/WZW/voc/tree_model/vocabulary_tree.tree', value=tree)

    tree.save2file('vocabulary_tree.tree')

    time_toc = time.time()
    time_total = time_toc - time_tic
    num_img = len(pt_names)
    print("\ndevice:{} total time:{:.1f}s mean:{:.3f}s".
          format(device, time_total, time_total / num_img))

    return tree


# # 节点的数据结构, 这里包括两个属性: 距离、数量
# class NodeData(object):
#     def __init__(self, Distance, Number):
#         self.Distance = Distance
#         self.Number = Number

# 在tree中, 获取parent节点的所有孩子节点与data的距离
def GetDist(query_pt,data_test,result_index):
    Pic_Dist = {}
    for key,value in result_index.items():
        Pic_Dist[key] = np.linalg.norm(query_pt - data_test[value])
    # ChildrenList = tree.children(parent)
    # ChildrenDist = [0] * len(ChildrenList)
    # for ChildIndex in range(len(ChildrenList)):
    #     ChildrenDist[ChildIndex] = np.linalg.norm(data - ChildrenList[ChildIndex].data.center)
    return Pic_Dist


# 在tree中, 获取parent节点的所有孩子节点中, 距离最小的那个节点
def GetMinDist(data, tree, parent):
    ChildrenList = tree.children(parent)
    ChildrenDist = [0] * len(ChildrenList)
    for ChildIndex in range(len(ChildrenList)):
        ChildrenDist[ChildIndex] = np.linalg.norm(data - ChildrenList[ChildIndex].data.center)
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


def SearchID(data, tree):
    MinDistPath = GetMinDistPath(data, tree)[::-1]  # 获取最短距离路径(反转)
    # TargetId = []  # 最后的结果(存储了节点的id)
    for CurNodeId in MinDistPath:  # 遍历最短距离路径中的每一个节点
        if tree.get_node(CurNodeId).is_leaf():  # 如果当前节点是叶子节点
            return CurNodeId


def NodetoPic(predict_id):
    IdtoPic = {}
    id_list = []
    for key, val in predict_id.items():
        if predict_id[key] not in id_list:
            id_list.append(predict_id[key])
    for i in range(len(id_list)):
        temp = []
        for key, val in predict_id.items():
            if predict_id[key] == id_list[i]:
                temp.append(key)
        IdtoPic[id_list[i]] = temp
    return IdtoPic


# 输入待查询影像的name，返回一系列相同id的影像的name
def Query(query_pt, Id_Pic, data_test, test_img_names_dict, N, tree):
    MinDistPath = GetMinDistPath(query_pt, tree)[::-1]  # 获取最短距离路径(反转)
    CurN = 0
    result = []
    result_index = {} #key是图片名 value是在test中的索引
    Pic_Dist = {}

    for CurNodeId in MinDistPath:
        if tree.get_node(CurNodeId).is_leaf():
            if CurNodeId in Id_Pic.keys():
                result = Id_Pic[CurNodeId]
                CurN = CurN + len(Id_Pic[CurNodeId])  # 更新CurN
        if CurN < N:
            BrotherLeaves = GetBrotherLeaves(tree, CurNodeId)
            for BrotherLeaf in BrotherLeaves:
                if BrotherLeaf.tag in Id_Pic.keys():
                    for i in range(len(Id_Pic[BrotherLeaf.tag])):
                        result.append(Id_Pic[BrotherLeaf.tag][i])
                    CurN = CurN + len(Id_Pic[BrotherLeaf.tag])
        else:
            break

    for i in range(len(result)):
        for index, name in test_img_names_dict.items():
            if name == result[i]:
                result_index[name] = index
                break
    Pic_Dist = GetDist(query_pt,data_test,result_index) #耗时0.005s
    return Pic_Dist


def order_dict(dicts, n):
    result = []
    result1 = []
    p = sorted([(k, v) for k, v in dicts.items()], reverse=False)
    s = set()
    for i in p:
        s.add(i[1])
    for i in sorted(s, reverse=False)[:n]:
        for j in p:
            if j[1] == i:
                result.append(j)
    for r in result:
        result1.append(r[0])
    return result1

def Map(ans,gd_path):
    count = 0
    precison = []
    f = open(gd_path, 'rb')
    data = pickle.load(f)
    gnd = data["gnd"]
    gnd_t = []
    for i in range(len(gnd)):
        gnd_t.append(gnd[i]["easy"])
    for i in range(len(gnd_t)):
        if len(gnd_t[i]) > 0:
            count += 1
            temp = []
            num = 0
            for j in range(len(ans[i])):
                if ans[i][j] in gnd_t[i]:
                    num += 1
                    temp.append(num/(j+1))
            if num > 0:
                precison.append(sum(temp)/num)
    map = sum(precison)/count
    return map

def npz_to_data(npz_dir,format="npz"):
    ans = []
    file_names = os.listdir(npz_dir)
    # 使用 list(filter(lambda())) 筛选出 jpg 后缀的文件
    npz_names = np.array(list(filter(lambda x: x.endswith(format), file_names)))
    for i in range(len(npz_names)):
        data = np.load(npz_dir+npz_names[i])
        output = data["global_descriptor"]
        ans.append(output)
    return ans
def load_vectors_from_txt(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        data = np.zeros((len(lines), 512))  # 初始化一个适当大小的数组
        for i, line in enumerate(lines):
            vector = np.fromstring(line, sep=' ')  # 将每行转换为一个NumPy数组
            data[i, :] = vector  # 将向量存储到数组中
    return data


def main():

    source_dir = "D:/WZW/Dataset/12wan/all_data"
    train_img_names = get_img_name(source_dir)
    # 使用示例
    txt_path = 'D:/WZW/Dataset/12wan/Features/D.txt'  # 替换为您的txt文件路径
    data_train = load_vectors_from_txt(txt_path)
    index_train = range(len(data_train))
    #生成树
    tree = get_vocabulary_tree(data_train[index_train], train_img_names[index_train], k, l)

if __name__ == "__main__":
    main()

