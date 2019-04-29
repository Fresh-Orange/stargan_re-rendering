## 简介
[StarGan](https://github.com/yunjey/stargan)的修改版本。
毕业论文[《基于对抗生成网络的人脸光照变换》](https://github.com/Fresh-Orange/stargan_re-rendering/blob/master/%E8%B5%96%E8%B4%A4%E5%9F%8E_%E6%AF%95%E4%B8%9A%E8%AE%BA%E6%96%87.pdf)的源代码。
基于StarGan加入了L1损失和身份损失。
其中L1损失的实现是通过修改代码中的loader，身份损失的实现是通过把[facenet-pytorch](https://github.com/liorshk/facenet_pytorch)在CMU-MULTI-PIE数据集上训练的模型加入StarGan的身份损失计算中。

## 文件(夹)作用
`main.py`: 程序入口，控制使用的GPU，生成一个data_loader

`solver.py`: 模型训练与测试的代码，对损失（loss）的修改在该文件中，如何处理测试集的代码也在其中。

`model.py`: 生成器与判别器的网络实现，以及facenet的网络实现
 
`preprocess.py`: 将文件夹下的图片整合成pth文件，这样做是为了方便实现L1损失

`myloader.py`: 为了实现L1损失而重新设计的loader，能很方便的得到某个人不同光照的图片

`1nn_classfier`: 用于做*人脸识别率*实验，使用的是原始图像的k-nn算法，其中k为1

`命令.txt`: 运行不同情况的命令行

`BaselFaceModel`: 3D渲染生成人脸模型，用以生成Basel数据集，
首先先要下载[预训练模型](https://pan.baidu.com/s/1y2ucTFLSmd-pBFf1AioqlQ)[提取码: j7ap]，
将其放在BaselFaceModel/PublicMM1文件夹下，然后使用matlab运行 `script_gen_random_head.m`

`facenet_pytorch`: facenet的pytorch版本（为什么不用官方的版本？ 答：因为我做不到将tensorflow模型变成可供pytorch调用）。
运行其中的`train_triplet.py`即可训练

`illumation_trsnasition`: 论文中用到的对比算法（ITI和NPL-QI），均是传统方法，要求输入的图像是灰度图

`illumation_trsnasition / NPL_QI`: 
首先训练TrainSubspace，这里用到的是全部类数据。例如：
```matlab
TrainSubspace([1:211], [0,1,4,7,10,13], '/media/data2/laixc/AI_DATA/type_changed/',6)
```
然后训练TrainStandardcoordinate，**注意**这里传入的OrderList只需要你希望模型预测的那个类。例如：
```matlab
TrainStandardcoordinate([1:211], [7], '/media/data2/laixc/AI_DATA/type_changed/')
```

`illumation_trsnasition / Illumination_Transition_Image`: 全部逻辑都集中在`Illumination_Transition_Image.m`中了，需要改变预测的类别，只需要改变代码中ObjectIllumination的值即可

## 注意事项与其他
`128与256`: 训练128x128模型时，将model中的FaceModel中的fc层输入改为512x4x4，而如果是训练512x512，则改成2048x4x4

`图像质量评估函数`: 使用sewar包，在`solver.py`中使用了

`Basel`: 运行`script_gen_random_head.m`前，请先手动创建所需的文件夹，文件夹名见代码内

`facenet_pytorch`: [原版本](https://github.com/liorshk/facenet_pytorch)的代码存在一些问题，该仓库中的版本已修复。
因此不要直接用原版本，而是用本仓库的版本