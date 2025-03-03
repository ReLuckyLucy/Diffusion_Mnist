
<div align="center">
 <img alt="logo" height="200px" src="img\logo.png">
</div>

<h1 align="center">基于MNIST数据集，从零构建diffusion扩散模型</h1>

<p align="center">
    <img alt="GitHub" src="https://img.shields.io/github/license/ReLuckyLucy/diffusion_mnist">
    <img alt="GitHub top language" src="https://img.shields.io/github/languages/top/ReLuckyLucy/diffusion_mnist">
    <img alt="GitHub release (latest by date including pre-releases)" src="https://img.shields.io/github/v/release/ReLuckyLucy/diffusion_mnist?include_prereleases">
    <img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/ReLuckyLucy/diffusion_mnist">
</p>
<p align="center">
    <img alt="GitHub code size in bytes" src="https://img.shields.io/github/languages/code-size/ReLuckyLucy/diffusion_mnist">
    <img alt="GitHub repo size" src="https://img.shields.io/github/repo-size/ReLuckyLucy/diffusion_mnist">
    <img alt="GitHub All Releases" src="https://img.shields.io/github/downloads/ReLuckyLucy/diffusion_mnist/total">
    <img alt="GitHub stars" src="https://img.shields.io/github/stars/ReLuckyLucy/diffusion_mnist?style=social">
</p>

本项目以MNIST数据集为例，从零构建扩散模型，具体会涉及到如下知识点：



- 退化过程（向数据中添加噪声）
- 构建一个简单的UNet模型
- 训练扩散模型
- 采样过程分析

## 🔥开始
```
#克隆仓库
git clone https://github.com/ReLuckyLucy/diffusion_mnist
cd diffusion_mnist

#创建虚拟环境
conda create -n diffusion_mnist python==3.9 -y
conda activate diffusion_mnist

# 通过pip下载依赖
pip install -r requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple/
```
## 📜数据集
 MNIST数据集是一个小数据集，存储的是0-9手写数字字体，每张图片都28X28的灰度图片，每个像素的取值范围是[0,1]，

## 🔛开始训练
```
python main.py
```

## 🪦结果
展示其训练过程的loss曲线如下图所示：

<div align="center">
 <img alt="loss" " src="img\losscurve.png">
</div>

结果将保存在/output文件夹内
