## 简介
[StarGan](https://github.com/yunjey/stargan)的修改版本。
毕业论文[《基于对抗生成网络的人脸光照变换》](https://github.com/Fresh-Orange/stargan_re-rendering/blob/master/%E8%B5%96%E8%B4%A4%E5%9F%8E_%E6%AF%95%E4%B8%9A%E8%AE%BA%E6%96%87.pdf)的源代码。
基于StarGan加入了L1损失和身份损失。
其中L1损失的实现是通过修改代码中的loader，身份损失的实现是通过把[facenet-pytorch](https://github.com/liorshk/facenet_pytorch)在CMU-MULTI-PIE数据集上训练的模型加入StarGan的身份损失计算中。

## 文件作用
`main.py`: 程序入口，控制使用的GPU，生成一个data_loader

`solver.py`: 模型训练与测试的代码，对损失（loss）的修改在该文件中，如何处理测试集的代码也在其中。

`model.py`: 生成器与判别器的网络实现

`preprocess.py`: 将文件夹下的图片整合成pth文件，这样做是为了方便实现L1损失

`myloader.py`: 为了实现L1损失而重新设计的loader，能很方便的得到某个人不同光照的图片