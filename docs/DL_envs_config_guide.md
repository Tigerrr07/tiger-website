---
sidebar_position: 4
slug: 深度学习环境配置指南
title: 深度学习环境配置指南
tags: [miniconda, pytorch, pytorch geometric]
---

## 1. Python包管理器 Miniconda
 [Miniconda — conda documentation](https://docs.conda.io/en/latest/miniconda.html)

**安装方式**

以linux为例，
``` bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh *.sh
```

**换源帮助**

[anaconda | 镜像站使用帮助 | 清华大学开源软件镜像站 | Tsinghua Open Source Mirror](https://mirrors.tuna.tsinghua.edu.cn/help/anaconda/) 

[pypi | 镜像站使用帮助 | 清华大学开源软件镜像站 | Tsinghua Open Source Mirror](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/)

**查看设置的镜像**
``` bash
conda config --show-sources
pip config list
```

**虚拟环境**

不同的任务可能需要不同的Python package，为了防止冲突，创建虚拟环境分隔开。下面创建名字为pyg的虚拟环境，

``` bash
conda create -n pyg python=3.8   # 创建
conda activate pyg  # 激活
```

**其他命令**
* 退出虚拟环境
  ``` bash
  conda deactivate
  ```
* 删除虚拟环境
  ``` bash
  conda remove -n pyg -–all
  ```
* 查看所有虚拟环境
  ``` bash
  conda env list
  ```

## 2. 深度学习框架 Pytorch
[Start Locally | PyTorch](https://pytorch.org/get-started/locally/)

**前提条件**

根据操作系统和有无显卡，
* 无显卡，安装CPU版本
* 有显卡，查看显卡驱动版本，任何低于该版本的cuda版本都可以安装
  ```
  nvidia-smi
  ```
  使用该命令，会显示Driver Version和Cuda Version.

**安装方式**

* Pytorch在使用cuda时，只会用到其中的部分功能，故安装时可选择直接安装相应的cudatoolkit，例如安装1.12.1版本的Pytorch

  ```bash
  conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
  ```
* 但当服务器已经安装了cuda，且满足你使用的版本，则没必要在安装Pytorch时同时安装cudatoolkit，这样做会占用一部分内存，则把指定的cudatoolkit取消即可

    ```
    conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 -c pytorch
    ```
    这时需要修改环境变量来切换cuda版本，可编辑`.bashrc`文件进行修改，例如指定cuda版本为11.1
    ```
    PATH=/usr/local/cuda-11.1/bin:$PATH
    LD_LIBRARY_PATH=/usr/local/cuda-11.1/lib64:$LD_LIBRARY_PATH
    ```
## 3. 图神经网络框架 Pytorch-Geometric
[PyG (PyTorch Geometric)](https://pytorch-geometric.readthedocs.io/en/latest/index.html)

**安装方式**

``` bash
conda install pyg -c pyg
```