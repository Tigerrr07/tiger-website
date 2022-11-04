---
slug: 拉普拉斯矩阵
title: 拉普拉斯矩阵
authors: slorber
tags: [拉普拉斯矩阵, 拉普拉斯特征映射]
---

> 默认向量都是列向量。求和符号可以简写。

## 拉普拉斯矩阵

### 定义

无向图$G=(V,E)$，$A \in \mathbb{R}^{n \times n}$为邻接矩阵，其元素
$$
a_{ij}=\begin{cases}
1 & \mathrm{if}\ (v_i,v_j) \in E \\
0 & \mathrm{else}
\end{cases}
$$

$N(i)$为结点$v_i$的邻居，$D \in R^{n \times n}$为度矩阵，对角矩阵，其元素
$$
d_{ii}= \sum_{j=1}^n a_{ij}= \sum _{j \in N(i)} a_{ij}
$$

定义拉普拉斯矩阵(Laplacian matrix) $L=D-A$，其元素
$$
l_{ij}=
\begin{cases}
d_i & \mathrm{if}\ i=j \\
-1 & \mathrm{if}\ (v_i,v_j) \in E  \\
0 & \mathrm{otherwise}
\end{cases}
$$

正则化表达形式(symmetric normalized laplacian) $L_{\mathrm{sym}}=D^{-1/2}LD^{-1/2}$，其元素
$$
l_{\mathrm{sym}}(i,j)=
\begin{cases}
1 & \mathrm{if}\ i=j \\
\frac{-1}{\sqrt{d_i d_j}}  & \mathrm{if}\ (v_i,v_j) \in E \\
0 & \mathrm{otherwise}
\end{cases}
$$

定义向量$\boldsymbol{x}=[x_1,x_2,···,x_n]^T$，可认为是图信号。则
$$
\begin{aligned}
L\boldsymbol{x}=(D-A)\boldsymbol{x}&=[···, d_ix_i-\sum_{j=1}^{n}a_{ij}x_j,···]^T
\\
&= [···,\sum _{j=1}^{n} a_{ij} x_i - \sum _{j=1}^{n} a_{ij} x_j,···]^T \\
&= [···, \sum _{j=1}^{n}a_{ij}(x_i-x_j),···]^T
\end{aligned}
$$

分量 $\sum _{j=1}^{n}a_{ij}(x_i-x_j)$ 可写成 $\sum _{j\in N(i)}(x_i-x_j)$，由此可知，拉普拉斯矩阵是反映图信号局部平滑度的算子。

接着我们定义二次型，
$$
\begin{aligned}
\boldsymbol{x}^TL\boldsymbol{x}&=\sum_{i=1}^{n} x_i \sum _{j=1}^{n}a_{ij}(x_i-x_j) \\
&= \sum_{i=1}^{n}\sum_{j=1}^{n} a_{ij}(x_i^2-x_ix_j)
\end{aligned}
$$
调换$i,j$符号，求和顺序保持不变，我们得到
$$
\boldsymbol{x}^TL\boldsymbol{x}=\sum_{i=1}^{n}\sum_{j=1}^n a_{ij}(x_i^2-x_ix_j)=\sum_{i=1}^{n}\sum_{j=1}^na_{ij}(x_j^2-x_ix_j)
$$
将等式左右两边相加，于是
$$
\begin{aligned}
\boldsymbol{x}^TL\boldsymbol{x} &= \frac{1}{2}\sum_{i=1}^{n}\sum_{j=1}^n a_{ij}(x_i^2-2x_ix_j+x_j^2) \\
&= \frac{1}{2}\sum_{i=1}^{n}\sum_{j=1}^n a_{ij}(x_i-x_j)^2
\end{aligned}
$$

由公式可以看出，二次型 $\boldsymbol{x}^TL\boldsymbol{x}$ 能刻画图信号的总体平滑度，成为总变差。
### 来源

拉普拉斯矩阵的定义来源于拉普拉斯算子，$n$维欧式空间中的一个二阶微分算子：$\Delta f=\sum_{i=1}^n \frac{\partial ^2 f}{\partial x_i^2}$。将该算子退化到离散二维图像空间就是边缘检测算子：
$$
\begin{aligned}
\Delta f(x,y) &=
\frac{\partial ^2 f(x,y)}{\partial x^2} + 
\frac{\partial ^2 f(x,y)}{\partial y^2}\\
&= [(f(x+1,y)-f(x,y))-(f(x,y)-f(x-1,y))]\\
&+ [(f(x,y+1)-f(x,y))-(f(x,y)-f(x,y-1))]\\
&= [f(x+1,y)+f(x-1,y)+f(x,y+1)+f(x,y-1)] -4f(x,y)
\end{aligned}
$$
图像处理中通常被当作模板的形式：
$$
\begin{bmatrix} 0 & 1 & 0\\ 1 & -4 & 1 \\0 & 1 & 0 \end{bmatrix}
$$
拉普拉斯算子用来描述中心像素与局部上、下、左、右四邻居像素的总的差异，这种性质经常也被用来当作图像上的边缘检测算子。



## Laplacian Eigenmaps 

假设图 $G=(V,E)$ 中有 $n$ 个节点，嵌入维度为 $d$，可得到如下 $n \times d$ 矩阵 $Y$，

$$
\begin{bmatrix}
y_1^{(1)} & y_1^{(2)} & \cdots & y_1^{(d)} \\
y_2^{(1)} & y_2^{(2)} & \cdots & y_2^{(d)} \\
\vdots & \vdots & \ddots& \vdots \\
y_n^{(1)} & y_n^{(2)} & \cdots & y_n^{(d)}
\end{bmatrix}
$$

$n$ 维行向量 $\boldsymbol{y}_k=[y_k^{(1)},y_k^{(2)}, ..., y_k^{(d)}]$，可表示一个节点的Embedding。

拉普拉斯特征映射（Laplacian Eigenmaps）用于图嵌入中，在图中邻接的节点，在嵌入空间中距离应该尽可能的近。可将其作为一阶相似性的定义，因此可以定义如下的loss function:

$$
\mathcal{L}_{1st}=\sum_{i=1}^{n} \sum_{j=1}^{n} a_{ij} ||\boldsymbol{y_i}-\boldsymbol{y_j}||_2^2
$$



$n$ 维列向量 $\boldsymbol{y}^{(k)} = [y_1^{(k)}, y_2^{(k)}, ···,y_n^{(k)}]^T$，可指一组图信号。因此可以得到，
$$
\begin{aligned}
\mathcal{L}_{1st}=\sum_{i=1}^{n} \sum_{j=1}^{n} a_{ij} ||\boldsymbol{y_i}-\boldsymbol{y_j}||_2^2&=
\sum_{i=1}^{n} \sum_{j=1}^{n} a_{ij} \sum_{k=1}^d (y_{i}^{(k)}-y_{j}^{(k)})^2  \\
&= \sum_{k=1}^d \sum_{i=1}^{n} \sum_{j=1}^{n} a_{ij} (y_{i}^{(k)}-y_{j}^{(k)})^2  \\
&= 2\sum_{k=1}^d \boldsymbol{y}^{(k)T} L \boldsymbol{y}^{(k)} \\
&= 2tr(Y^TLY)
\end{aligned}
$$

### 二次型表达式的另一种推导

对任意$n$维列向量 $\boldsymbol{y}=[y_1, y_2, ..., y_n]^T$，展开可得到，
$$
\begin{aligned}
\boldsymbol{y}^TL\boldsymbol{y}&= 
\boldsymbol{y}^T D \boldsymbol{y} - \boldsymbol{y}^T A \boldsymbol{y}\\
&= \sum_{i=1}^{n} d_iy_i^2- \sum_{i=1}^{n} \sum_{j=1}^{n}  a_{ij} y_iy_j\\
&= \frac{1}{2} (\sum_{i=1}^{n} d_iy_i^2  -2\sum_{i=1}^{n} \sum_{j=1}^{n} a_{ij}y_iy_j + \sum_{j=1}^{n} d_jy_j^2) \\
&=\frac{1}{2} (\sum_{i=1}^{n} \sum_{j=1}^{n}  a_{ij}y_i^2  -2\sum_{i=1}^{n} \sum_{j=1}^{n} a_{ij}y_iy_j + \sum_{j=1}^{n} \sum_{i=1}^{n}  a_{ji}y_j^2) \\
&=\frac{1}{2} (\sum_{i=1}^{n} \sum_{j=1}^{n}  a_{ij}y_i^2  -2\sum_{i=1}^{n} \sum_{j=1}^{n} a_{ij}y_iy_j + \sum_{i=1}^{n} \sum_{j=1}^{n}  a_{ij}y_j^2) \\
&= \frac{1}{2}\sum_{i=1}^{n} \sum_{j=1}^{n} a_{ij}(y_i-y_j)^2
\end{aligned}
$$





