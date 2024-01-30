# LOIP for SfM
![image name](https://s2.loli.net/2024/01/30/sk9e6Qch7i2nDtp.png)

In this repository, to enhance the generality of several conventional backbone networks in detecting **overlapping image pairs** with different rotations, we propose a simple yet efficient fine-tuning solution by using images with referenced overlapping relationships. Additionally, we build a **hierarchical vocabulary tree** with fine-tuned global features to further improve the time efficiency of image retrieval. Furthermore, its feasibility for both offline and online image retrieval is demonstrated through comprehensive validation on offline and online Structure-from-Motion (SfM).


## 0.Setup
### Dependencies

 - torchvision~=0.16.0
 - h5py~=3.10.0
 - numpy~=1.26.2
 - pillow~=10.0.1
 - tensorboardx~=2.6.2.2
 - joblib~=1.2.0
 - scipy~=1.11.4
 - scikit-learn~=1.2.2
 - faiss~=1.7.4·
 - mkl~=2023.1.0
The above are the packages on which the project depends. Of course, you can also directly configure the environment using the following code.
```python
pip install -r requirements.txt
```
### Data
This repository utilizes three open-source datasets: the BEDOI dataset[link]([WHUHaoZhan/BeDOI: This is the implementation of paper “Benchmarks for Determining Overlapping Images with Photogrammetric Information” (github.com)](https://github.com/WHUHaoZhan/BeDOI)) for training, the GL3D dataset[link]([lzx551402/GL3D: GL3D (Geometric Learning with 3D Reconstruction): a large-scale database created for 3D reconstruction and geometry-related learning problems (github.com)](https://github.com/lzx551402/GL3D)) for testing the speed of vocabulary tree retrieval, and the LOIP dataset[link]([xwangSGG/LOIP: This is the implementation of paper "Learning overlapping image pairs for SfM via CNN fine-tuning with photogrammetric geometry information" (github.com)](https://github.com/xwangSGG/LOIP?tab=readme-ov-file)) already employed for Structure-from-Motion (SfM).
## 1.Train the model
It is necessary to first run `/LOIP/ModelTrain.py` with the correct settings. After which a model can be trained using:
```python
python ModelTrain.py
```
The commandline args, the tensorboard data, and the model state will all be saved to `runs/`, which subsequently can be used for testing or validating. In the `dataset.py` file, it mainly includes loading images, preprocessing images, and data augmentation.

## 2.Hierarchical Vocabulary Tree

This section primarily involves the training and retrieval of the hierarchical vocabulary tree. The `/Voc/Train_Voc.py` is the code for training the vocabulary tree. In the code, you can adjust the shape of the vocabulary tree, such as setting K for the number of nodes per layer and L for the depth of the tree, to alter its structure. Simply run the code to observe the changes：
```
python Train_Voc.py
```
Additionally, we provide the code `Retrieval_Voc.py` for conducting retrieval using the vocabulary tree.
```
python Retrieval_Voc.py
```

## 3.Calculate
  
In this section, we offer code for Exhausive matching to find the TOP-N matches for comparison, scripts for generating overlapping image pairs and calculating Mean Average Precision (MAP).
Based on the extracted features, we can employ exhaustive matching to determine the Top-N for comparison. The code is as follows:
```
python Retrieval_Exh.py
```
  
In addition, you can use 'Cal_Map.py' to calculate map@[N].
```
python Cal_Map.py
```
Finally, you can run `GetImagePairs.py`. This step generates a file of overlapping image pairs based on the TOP-N matches. We save the file as a default txt format, which can be fed into Colmap to expedite the image matching process.
```
python GetImagePairs.py
```




## Citation
Any question can be sent to [xwang@sgg.whu.edu.cn](mailto:xwang@sgg.whu.edu.cn).


