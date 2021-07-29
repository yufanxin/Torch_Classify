# 深度学习在图像分类中的应用

## 前言

起因：因为我看github上面很多其他计算机视觉任务的集成，都写得很好了，但是分类这块，一直没找到我想要的那种清楚点的，每次要用的时候都很烦，索性自己整理了一个符合自己需求的。以后也会陆续添加模型。

* 本教程是对本人本科生期间的研究内容进行整理总结，总结的同时也希望能够帮助更多的小伙伴。后期如果有学习到新的知识也会与大家一起分享。
* 本教程使用Pytorch进行网络的搭建与训练。
* 本教程参考的链接附在最后。感谢大家的支持。

## 目前

#### 2021. 7.29

* 更新了tools中创建数据的工具

#### 2021. 7.28

* 增加了ResMlp-Mixer VoVNet se-resnet SqueezeNet MnasNet模型
* 优化了logs文件夹，记录每次exp的config，添加requirements.txt并纠正环境配置
* 修复了warmup_epoch=0的BUG
* 更新了tools中转换权重、计算模型参数、模型FPS、模型吞吐量的工具
* 更新了权重加载方式和权重链接

#### 2021.7.25

- first commit


# 支持模型

```
#  --------------------------------------------------------------------------------------
# |model_prefix    |model_suffix                                                         |
# |--------------------------------------------------------------------------------------|
# |vgg             |11 13 16 19 bn11 bn13 bn16 bn19                                      |
# |--------------------------------------------------------------------------------------|
# |resnet          |18 34 50 101 152                                                     |
# |--------------------------------------------------------------------------------------|
# |resnext         |50-32x4d 101-32x8d                                                   |
# |--------------------------------------------------------------------------------------|
# |regnetx         |200mf 400mf 600mf 800mf 1.6gf 3.2gf 4.0gf 6.4gf 8.0gf 12gf 16gf 32gf |
# |--------------------------------------------------------------------------------------|
# |regnety         |200mf 400mf 600mf 800mf 1.6gf 3.2gf 4.0gf 6.4gf 8.0gf 12gf 16gf 32gf |
# |--------------------------------------------------------------------------------------|
# |mobilenetv2     |0.25, 0.5, 0.75, 1.0, 1.25, 1.5                                      |
# |--------------------------------------------------------------------------------------|
# |mobilenetv3     |small large                                                          |
# |--------------------------------------------------------------------------------------|
# |ghostnet        |0.5 1.0 1.3                                                          |
# |--------------------------------------------------------------------------------------|
# |efficientnetv1  |b0 b1 b2 b3 b4 b5 b6 b7                                              |
# |--------------------------------------------------------------------------------------|
# |efficientnetv2  |small medium large                                                   |
# |--------------------------------------------------------------------------------------|
# |shufflenetv2    |0.5 1.0 1.5 2.0                                                      |
# |--------------------------------------------------------------------------------------|
# |densenet        |121 161 169 201                                                      |
# |--------------------------------------------------------------------------------------|
# |xception        |299                                                                  |
# |--------------------------------------------------------------------------------------|
# |vit             |base-patch16 base-patch32 large-patch16 large-patch32 huge-patch14   |
#  --------------------------------------------------------------------------------------
# |resmlp-mixer    |12 24 36 B24                                                         |
#  --------------------------------------------------------------------------------------
# |vovnet          |27slim 39 57                                                         |
#  --------------------------------------------------------------------------------------
# |se-resnet       |18 34 50 101 152                                                     |
#  --------------------------------------------------------------------------------------
# |squeezenet      |1.0 1.1                                                              |
#  --------------------------------------------------------------------------------------
# |mnasnet         |0.5 0.75 1.0 1.3                                                     |
#  --------------------------------------------------------------------------------------
```

# 训练准备

## 数据格式

```
# -data
#    -train
#       -class_0
#          -1.jpg
#       -class_1
#       -...
#    -val
#       -class_0
#       -class_1
#       -...
```

## 环境配置

* Anaconda3
* python 3.8
* pycharm (IDE， 建议使用)
* pytorch 1.8.1
* apex 0.1.0
* VS2019
* Cuda10.2

### 配置cl.exe

C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.29.30037\bin\Hostx86\x64

注意是vs2019  中间版本号可能不同 但是一定要Hostx86/x64的cl.exe

把cl.exe路径添加到系统环境变量并移至最上层         cuda10.2的path高于10.1.10.0

在cmd中输入

	set Path=C
	cl
显示cl的版本号就无问题了。如19.14.29.30037

在我的mmsegmentation环境安装教程中有对cl.exe配置的片段。https://www.bilibili.com/video/BV1NB4y1N7Qo

### 安装apex

注意该步骤在Anaconda的**PowerSheel**中进行
```
cd apex-master
pip install -r requirements.txt
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" .
```
注意语句最后的点也要复制

在我的yolox环境安装教程中有对cl.exe配置的片段。https://www.bilibili.com/video/BV1Gf4y157u8

# 训练

在config.py中修改你想要的模型配置，注意，我的代码中，**每个模型有2部分组成**，分别是model_prefix和model_suffix。

为了方便大家，我写了关于参数用途的注释。

例如

```
model_prefix='shufflenetv2'
model_suffix='0.5'
```

配置好之后运行train.py

# 可视化相关指标

训练完成之后在你的log_dir中查看训练过程。

<img src=".\logs\mobilenetv3_small\exp0\P-R-F1-per-class.jpg" width="700"/>
<img src=".\logs\mobilenetv3_small\exp0\P-R-F1.jpg" width="700"/>
<img src=".\logs\mobilenetv3_small\exp0\data_distribution.jpg" width="700" />

# 预测

* 我只写了单张图片的预测，但是你可以在我的基础上很灵活的更改成适合你项目需求的预测代码。

* 同样的，在config.py中修改load_from，predict_img_path，注意这里img_path不再有效，因为img_path只针对训练。

# 模型权重

* 我对torch的官方权重或者是原论文作者放出的权重进行了收集，所以应该不存在模型出错的问题，如果有，请及时通知我。并且在每一个model.py中注明了权重链接。但注意，因为本人能力有限，仍有部分模型的权重搜集不到，如果你有相关的权重链接，请通知我！

* 请注意，densenet需要运行tools/convert_weight_densenet.py进行转换权重。当然你也可以直接下载我转换后的权重。

* 在utils/general.py中有加载权重的方式：取模型和权重的公共的键值对进行加载。因为这样可以基于你魔改之后的网络加载。

  * densenet 链接：https://pan.baidu.com/s/1Lo-xSbR_9VpvfDJrXO31Fg  提取码：BDAS 
  * efficientnetv1 链接：https://pan.baidu.com/s/1KDWtBzVn78C6xZrOBC9lkg  提取码：BDAS 
  * efficientnetv2 链接：https://pan.baidu.com/s/1sLYyrRAeHd7XNo7l9pHQhg  提取码：BDAS 
  * mobilenetv2 链接：https://pan.baidu.com/s/1YA5qgVN-PICEZXwhlwzOig  提取码：BDAS 
  * mobilenetv3 链接：https://pan.baidu.com/s/1fijzykqVz5cvv8w_OZgR4w  提取码：BDAS 
  * regnet 链接：https://pan.baidu.com/s/1WDfgFcmudzMOOceViRY1BQ  提取码：BDAS 
  * ghostnet 链接：https://pan.baidu.com/s/1QByMVopEXgMsnNEIln9q5g  提取码：BDAS 
  * resne(x)t 链接：https://pan.baidu.com/s/14rd042Ra6DhscNHqtEk38A  提取码：BDAS 
  * shufflenetv2 链接：https://pan.baidu.com/s/1hMxjwOE02aDXL7nM6smFRw  提取码：BDAS 
  * vgg 链接：https://pan.baidu.com/s/1HkY--EQJauwp-Cq1bKyd9w  提取码：BDAS 
  * xception 链接：https://pan.baidu.com/s/19uo_dLENcuL1oFznLvl24A  提取码：BDAS 
  * vit 链接：https://pan.baidu.com/s/15OTxNPDWkgdiuqZWSQ7sUw  提取码：BDAS 
  * resmlp-mixer 链接：https://pan.baidu.com/s/1_qHcRdy65JatyHrHx5fReQ  提取码：BDAS 
  * vovnet 链接：https://pan.baidu.com/s/1vaY-WOJdLwONiB_HYcpvpA  提取码：BDAS 
  * se-resnet 链接：https://pan.baidu.com/s/1Zs4gs7MR3pm25Vilqo3yyA  提取码：BDAS 
  * squeezenet 链接：https://pan.baidu.com/s/10NDqSJH4sgDZroGriTU3uQ  提取码：BDAS 
  * mnasnet 链接：https://pan.baidu.com/s/1Vs3c-qU0IyQCs7BUpQ5HaA  提取码：BDAS 


# 参考

1. https://github.com/pytorch/vision/tree/master/torchvision/models

2. https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/tree/master/pytorch_classification

3. https://github.com/rwightman/pytorch-image-models/tree/master/timm/models


# 联系方式

1. QQ：2267330597
2. E-mail：201902098@stu.sicau.edu.cn
