import os.path
import os
import random
import threading
from pathlib import Path
import torchvision.transforms as transforms
import h5py
import torch
from sklearn.neighbors import NearestNeighbors
from multiprocessing.dummy import Pool as ThreadPool

from PIL import Image, ImageFile
from scipy.io import loadmat
from torch.utils import data
from pathlib import Path
from PIL import Image
from io import BytesIO

from ModelTrain import NewWidth, NewHeight


#Angle = [0, 90, 180, 270]
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class Dataset(data.Dataset):
    def __init__(self, DataDir, DatasetPath, DataIsTrain, IsRandom, NRandom):
        super().__init__()
        if DataIsTrain:
            files = os.listdir(DataDir)
            for file in files:
                file_path = os.path.join(DataDir, file)
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                if file == 'Train_Database.txt':
                    self.Database = []
                    for line in lines:
                        self.Database.append(DatasetPath + "/" + line.strip())
                elif file == 'Train_Query.txt':
                    self.Query = []
                    for line in lines:
                        self.Query.append(DatasetPath + "/" + line.strip())
                        #self.Query = self.Query[:2000]
                elif file == 'Train_Positive.txt':
                    self.Positive = []
                    for line in lines:
                        self.Positive.append(DatasetPath + "/" + line.strip())
                        #self.Positive = self.Positive[:2000]
                elif file == 'Train_Negative.txt':
                    self.Negative = []
                    for line in lines:
                        self.Negative.append(DatasetPath + "/" + line.strip())
                        #self.Negative = self.Negative[:2000]
            self.GetDataType = 'None'
        else:
            files = os.listdir(DataDir)
            for file in files:
                file_path = os.path.join(DataDir, file)
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                if file == 'Validate_Database.txt':
                    self.TestDatabase = []
                    for line in lines:
                        self.TestDatabase.append(DatasetPath + "/" + line.strip())
                elif file == 'Validate_Query.txt':
                    self.Query = []
                    for line in lines:
                        self.Query.append(DatasetPath + "/" + line.strip())
                elif file == 'Validate_TopN.txt':
                    TopN_Str = []
                    for line in lines:
                        TopN_Str.append(line.strip().split('\t'))
                    self.TopN = [[int(j) for j in i] for i in TopN_Str]
            self.GetDataType = 'None'

    def __len__(self):
        if self.GetDataType == 'Database':
            return len(self.TestDatabase)
        elif self.GetDataType == 'Cluster':
            return len(self.Database)
        else:
            return len(self.Query)

    # def PreProcess(self, Image):
    #     HF = transforms.RandomHorizontalFlip() #随机水平翻转
    #     Image = HF(Image)
    #     VF = transforms.RandomVerticalFlip() #随机竖直翻转
    #     Image = VF(Image)
    #     if random.random() > 0.2:
    #         RR = transforms.RandomRotation(degrees=(0, 360))  # 随机旋转
    #         Image = RR(Image)
    #     if random.random() > 0.5:
    #         Image = transforms.ColorJitter(brightness=0.2)(Image)
    #     if random.random() > 0.5:
    #         Image = transforms.ColorJitter(contrast=0.2)(Image)
    #     if random.random() > 0.5:
    #         Image = transforms.ColorJitter(saturation=0.2)(Image)
    #     if random.random() > 0.5:
    #         Image = transforms.ColorJitter(hue=0.4)(Image)
    #     return Image
    def PreProcess(self, Image):
        # HF = transforms.RandomHorizontalFlip() #随机水平翻转
        # Image = HF(Image)
        # VF = transforms.RandomVerticalFlip() #随机竖直翻转
        # Image = VF(Image)
        # if random.random() > 0.2:
        #     RR = transforms.RandomRotation(degrees=(0, 360))  # 随机旋转
        #     Image = RR(Image)
        # if random.random() > 0.5:
        #     Image = transforms.ColorJitter(brightness=0.2)(Image)
        # if random.random() > 0.5:
        #     Image = transforms.ColorJitter(contrast=0.2)(Image)
        # if random.random() > 0.5:
        #     Image = transforms.ColorJitter(saturation=0.2)(Image)
        # if random.random() > 0.5:
        #     Image = transforms.ColorJitter(hue=0.4)(Image)
        # if random.random() > 0.5:
        #     RS = transforms.RandomResizedCrop(224)  # 随机缩放和裁剪
        #     Image = RS(Image)
        # if random.random() > 0.5:
        #     RC = transforms.RandomCrop(224)  # 随机裁剪
        #     Image = RC(Image)
        # if random.random() > 0.5:
        #     RA = transforms.RandomAffine(degrees=45, translate=(0.1, 0.1), scale=(0.8, 1.2))  # 随机仿射变换
        #     Image = RA(Image)
        return Image


    def __getitem__(self, index):
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        if self.GetDataType == 'Triplet':
            Query = Image.open(self.Query[index])
            if len(Query.split()) != 3:
                Query = Query.convert('RGB')
            Query = transform(Query)

            Positive = Image.open(self.Positive[index])
            if len(Positive.split()) != 3:
                Positive = Positive.convert('RGB')
            Positive = transform(Positive)
            # Positive = Positive.resize((NewWidth, NewHeight))
            # #Positive.save("Positive1.jpg")
            Positive = self.PreProcess(Positive)
            # #Positive.save("Positive2.jpg")
            # Positive = transforms.Compose([transforms.ToTensor()])(Positive)


            Negative = Image.open(self.Negative[index])
            if len(Negative.split()) != 3:
                Negative = Negative.convert('RGB')
            Negative = transform(Negative)
            # Negative = Negative.resize((NewWidth, NewHeight))
            # #Negative.save("Negative1.jpg")
            Negative = self.PreProcess(Negative)
            # #Negative.save("Negative2.jpg")
            # Negative = transforms.Compose([transforms.ToTensor()])(Negative)

            return Query, Positive, Negative, index

        elif self.GetDataType == 'Database':
            DbImg = Image.open(self.TestDatabase[index])
            if len(DbImg.split()) != 3:
                DbImg = DbImg.convert('RGB')
            DbImg = transform(DbImg)
            #DbImg = DbImg.resize((NewWidth, NewHeight))
            DbImg = self.PreProcess(DbImg)
            #DbImg = transforms.Compose([transforms.ToTensor()])(DbImg)
            return DbImg, index
        elif self.GetDataType == 'TestQuery':
            TestQueryImg = Image.open(self.Query[index])
            if len(TestQueryImg.split()) != 3:
                TestQueryImg = TestQueryImg.convert('RGB')
            TestQueryImg = transform(TestQueryImg)
            # TestQueryImg = TestQueryImg.resize((NewWidth, NewHeight))
            # TestQueryImg = transforms.Compose([transforms.ToTensor()])(TestQueryImg)
            return TestQueryImg, index
        elif self.GetDataType == 'Cluster':
            DbImg = Image.open(self.Database[index])
            if len(DbImg.split()) != 3:
                DbImg = DbImg.convert('RGB')
            DbImg = DbImg.resize((NewWidth, NewHeight))
            DbImg = transforms.Compose([transforms.ToTensor()])(DbImg)
            return DbImg, index
def collate_fn(batch):
    """Creates mini-batch tensors from the list of tuples (query, positive, negatives).

    Args:
        data: list of tuple (query, positive, negatives).
            - query: torch tensor of shape (3, h, w).
            - positive: torch tensor of shape (3, h, w).
            - negative: torch tensor of shape (n, 3, h, w).
    Returns:
        query: torch tensor of shape (batch_size, 3, h, w).
        positive: torch tensor of shape (batch_size, 3, h, w).
        negatives: torch tensor of shape (batch_size, n, 3, h, w).
    """

    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0: return None, None, None, None, None

    query, positive, negatives, indices = zip(*batch)

    query = data.dataloader.default_collate(query)
    positive = data.dataloader.default_collate(positive)
    negatives = data.dataloader.default_collate(negatives)
    #negCounts = data.dataloader.default_collate([x.shape[0] for x in negatives])
    #negatives = torch.cat(negatives, 0)
    import itertools
    indices = list(itertools.chain(*[indices]))

    return query, positive, negatives, indices