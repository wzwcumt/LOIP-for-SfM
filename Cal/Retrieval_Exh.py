import math

# 初始化
Descriptors_DirPath = r"D:\WZW\TraiDataXYW\GL3D\rawimage"
# Descriptors_DirList = ["2000_Descriptors", "2000_Descriptors_LastModel", "13667_Descriptors", "13667_Descriptors_LastModel"]
Descriptors_DirList = ["RawImageDescriptors"]
# Descriptors_FileList = ["Block7_Descriptors.txt", "Block11_Descriptors.txt", "Block13_Descriptors.txt",
                        # "Block21_Descriptors.txt", "Block25_Descriptors.txt", "Block27_Descriptors.txt", ]
Descriptors_FileList = ["RawImage1_Descriptors.txt", "RawImage2_Descriptors.txt", "RawImage3_Descriptors.txt"]
N = 50

# 对每一个文件进行处理
for DirName in Descriptors_DirList:
    DirPath = Descriptors_DirPath + '\\' + DirName
    for FileName in Descriptors_FileList:
        # 读入一个block中所有影像的特征矢量
        print("--------------------------------------{0}--------------------------------------".format(FileName.split("_")[0]))
        print("正在读取特征矢量！")
        FilePath = DirPath + '\\' + FileName
        InputFile = open(FilePath)
        ImageNum = int(InputFile.readline().split(' ')[-1])
        ImageName = []
        ImageDescriptors = []
        for i in range(ImageNum):
            tempstr = InputFile.readline().split(' ')
            ImageName.append(tempstr[0].split('\\')[-1])
            templist = []
            for j in range(len(tempstr) - 1):
                templist.append(float(tempstr[j + 1]))
            ImageDescriptors.append(templist)
            print(i + 1, '/', ImageNum)

        # 计算任意两张影像之间的欧氏距离
        print("--------------------------------------{0}--------------------------------------".format(FileName.split("_")[0]))
        print("正在计算任意两张影像之间的欧氏距离！")
        Distance = [[-1 for I in range(ImageNum)] for J in range(ImageNum)]
        for i in range(ImageNum):
            for j in range(i, ImageNum):
                Dis = 0
                for k in range(len(ImageDescriptors[i])):
                    Dis = Dis + (ImageDescriptors[i][k] - ImageDescriptors[j][k]) * (ImageDescriptors[i][k] - ImageDescriptors[j][k])
                Dis = math.sqrt(Dis)
                Distance[i][j] = Dis
                Distance[j][i] = Dis
            print(i + 1, '/', ImageNum)

        # 计算每一张影像的TopN
        print("--------------------------------------{0}--------------------------------------".format(FileName.split("_")[0]))
        print("正在计算任意影像的TopN！")
        TopN = []
        for i in range(ImageNum):
            TempTopN = []
            TempDistanceSortedList = sorted(Distance[i])
            for j in range(N):
                ThisImageIndex = Distance[i].index(TempDistanceSortedList[j + 1])
                TempTopN.append(ImageName[ThisImageIndex])
            TopN.append(TempTopN)
            print(i + 1, '/', ImageNum)

        # 输出TopN的结果
        print("--------------------------------------{0}--------------------------------------".format(FileName.split("_")[0]))
        print("正在输出计算结果！")
        OutputFile = open(DirPath + '\\' + FileName.split('.')[0] + "_Top" + str(N) + ".txt", "w")
        for i in range(ImageNum):
            OutputFile.write(ImageName[i] + " :\n")
            for j in range(N):
                OutputFile.write(TopN[i][j])
                if (j == N - 1):
                    OutputFile.write('\n')
                else:
                    OutputFile.write(' ')



