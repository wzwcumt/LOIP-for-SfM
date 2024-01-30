# 初始化
N = 50
# Descriptors_DirPath = r"D:\GL3D\Model_ImageDataset"
Descriptors_DirPath = r"D:\WZW\TraiDataXYW\GL3D\rawimage"
# Descriptors_DirList = ["2000_Descriptors", "2000_Descriptors_LastModel", "13667_Descriptors", "13667_Descriptors_LastModel"]
# Descriptors_DirList = ["VGG16_Origin_Descriptors"]
Descriptors_DirList = ["RawImageDescriptors"]
# Descriptors_FileList = ["Block7_Descriptors_Top50.txt", "Block11_Descriptors_Top50.txt", "Block13_Descriptors_Top50.txt",
                        # "Block21_Descriptors_Top50.txt", "Block25_Descriptors_Top50.txt", "Block27_Descriptors_Top50.txt"]
Descriptors_FileList = ["RawImage1_Descriptors_Top50.txt", "RawImage2_Descriptors_Top50.txt", "RawImage3_Descriptors_Top50.txt"]
ImageNumList = [836, 758, 751]

# 开始生成影像对
for DirName in Descriptors_DirList:
    DirPath = Descriptors_DirPath + '\\' + DirName
    I = 0
    for FileName in Descriptors_FileList:
        # 读入影像匹配结果
        print("--------------------------------------{0}--------------------------------------".format(FileName.split("_")[0]))
        print("正在读入影像匹配结果！")
        FilePath = DirPath + '\\' + FileName
        InputFile = open(FilePath)
        ImageList = []
        TopN = []
        for i in range(ImageNumList[I]):
            ImageList.append(InputFile.readline().split(" ")[0])
            TopN.append(InputFile.readline().split(" ")[1:])

         # 获取不重复的影像对
        print("正在获取不重复的影像对！")
        AlreadyExist = []
        for i in range(ImageNumList[I]):
            for j in range(N):
                if ((ImageList[i], TopN[i][j]) not in AlreadyExist and (TopN[i][j], ImageList[i]) not in AlreadyExist):
                    AlreadyExist.append((ImageList[i], TopN[i][j]))
                else:
                    continue

        # 输出结果
        print("正在输出结果！")
        OutputFile = open(DirPath + '\\' + FileName.split('.')[0] + "_ImagePairs" + ".txt", "w")
        for i in range(len(AlreadyExist)):
            OutputFile.write(AlreadyExist[i][0] + " " + AlreadyExist[i][1] + '\n')
        OutputFile.close()

        I = I + 1

