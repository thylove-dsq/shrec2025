# ProteinProject

#### 介绍
蛋白质分类项目，是根据蛋白质的3d模型，转成2d的图像进行分类。现有的模型对二维图像提取特征有着较成熟的表现，对于3d模型最直观的想法，是把3d模型转为2维图像，再进行特征提取，分类。

#### 软件架构
软件架构说明


#### 安装教程

1. Ubuntu 24.04.2 LTS 、Python 3.9.21
2. pip install -r requirements.txt

#### 使用说明

1.  将[shrec2025](https://shrec2025.drugdesign.fr/)提供的数据集（**Training_Set**: [Downloads VTK files](https://shrec2025.drugdesign.fr/files/train_set_vtk.tar.gz) — [Download CSV file](https://shrec2025.drugdesign.fr/files/train_set.csv)和**Test_Set**: [Download VTK files](https://shrec2025.drugdesign.fr/files/test_set_vtk.tar.gz) — [Download CSV file](https://shrec2025.drugdesign.fr/files/test_set.csv)）放在`data`目录下
2.  运行`deal_data.py`，生成蛋白质的图像，具体过程：先是将`.tar.gz`解压，再将`.vtk`文件转换成`.obj`文件、，最后利用`.obj`文件生成蛋白质的二维图像。
3.  运行`main.py`，在`data`目录生成`category_test_set.csv`文件，此文件是对`test_set.csv`文件进行分类后的文件。具体过程：加载数据集，按照`3:1:1`分成训练集、测试集、和验证集，训练、评估，分类、对未分类的测试集数据进行分类，在`data`目录下生成`category_test_set.csv`文件。

