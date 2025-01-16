---
title: AFGRL阅读笔记
date: 2024-08-02
updated: 2024-08-02
tags: [图神经网络]
categories: [论文笔记] 
---

## Abstract
提出了一种新颖的无增强图自监督学习框架，节点粒度的工作（个人感觉已经不像是对比学习了，不需要负例，只需要最小化正例之间的距离）

## Introduction
在对比学习方法中，需要对数据进行增强来构造正负例，但是 image 和 graph 在数据特征上具有很大的不同：虽然增强在图像上得到了很好的定义，但它在图上的行为可能是任意的。例如，就图像而言，即使随机裁剪和旋转它们，或者扭曲它们的颜色，它们的底层语义也几乎没有改变。另一方面，当我们扰动（删除或添加）图的边/节点及其特征时，我们无法确定增强图是否与原始图正相关，更糟糕的是，由于图很难可视化，因此我们也没办法直观的感受到增强的有效性。例如，在分子图中，从阿司匹林的苯环上去掉一个碳原子会破坏芳香系统并产生烯烃链。此外，扰乱阿司匹林的连接可能会引入一种性质完全不同的分子，即五元内酯，这些微小改动造成了巨大的变化。

然后作者还进行了实验对比，说明了先前的工作有效性与否，很大程度上取决于使用的数据增强方式

## contribution
不同于之前的对比学习方法，本文提出的方法`不构造`view，而是从图本身去找 view，那么如何去确保自己找的 view 是一个好的 view 呢，本文提出了考虑了两种找 view 的方式结合起来，这样找到的 view 就更可能是好 view。

## Impletation
![](https://raw.githubusercontent.com/zy-Pioneer/BlogImage/main/img/2025/01/0584cdf7.png)

如上图所示，首先对一张图输入到两个 GCN 中，分别得到两个节点的嵌入H<sup>ξ</sup> 和 H<sup>θ</sup> ，这两者可以分别看做是初步找到的 view 和原始 view（分别记作 VE1 和 VE），接下来我们是要针对这个初步找到的 view 进行优化，让它更好。

### KNN
第一步我们对 VE 中的每个节点，去计算 VE1 中每个节点和该节点的距离：

![](https://raw.githubusercontent.com/zy-Pioneer/BlogImage/main/img/2025/01/b0ef9ecf.png)

然后找到 K 个最近的节点，作为第一步的结果（KNN，K nearest neighbor），假设对于节点 V<sub>i</sub> ，KNN 的结果为 B<sub>i</sub> 

### Local Structural Information
首先作者做了实验拥挤，发现方法 1 的结果，随着 K 的增大，其中和目标节点共享同一标签的比率逐渐降低

![](https://raw.githubusercontent.com/zy-Pioneer/BlogImage/main/img/2025/01/20f4f5b0.png)

说明 1 的方法得到的结果存在一定的误差，为了缓解这一误差。作者考虑到了这一假设：

> 对于节点 vi，其相邻节点 Ni 倾向于与查询节点 vi 共享相同的标签，即平滑假设（Zhu、Ghahramani 和 Lafferty 2003）。
>

作者结合了 Local Structural Information，也就是说将目标节点的V<sub>i</sub> 的邻居节点集合 N<sub>i</sub> 和上一步得到的B<sub>i</sub> 作交集

![](https://raw.githubusercontent.com/zy-Pioneer/BlogImage/main/img/2025/01/a6f5721d.png)

### Global Semantics
在之前的步骤中，我们的筛选找到了和目标节点相邻的、具有相似语义的节点，但是也存在这样一种情况，即和目标节点不相邻、具有语义相似的节点（在文中举例是，在引文网络中，两个学者的研究方向一致，但是两人的文章并没有引用关系）。

作者认为这种特征可以根据聚类来捕获。（这里是我的一个疑惑的地方，为什么聚类可以捕获这种特征）

具体做法是，针对之前得到的 H<sup>ξ</sup> 进行 Kmeans 聚类，得到很多个簇，假设我们的目标节点 ![](https://raw.githubusercontent.com/zy-Pioneer/BlogImage/main/img/2025/01/d21e93a2.png)，那么和 v<sub>i </sub>属于同一个簇的所有节点我们记作 C<sub>i</sub> ，这里同样我们取交集

![](https://raw.githubusercontent.com/zy-Pioneer/BlogImage/main/img/2025/01/fdab93fb.png)

最终，我们通过上述两部分，得到了筛选过后的 VE1:

![](https://raw.githubusercontent.com/zy-Pioneer/BlogImage/main/img/2025/01/8b57fb69.png)

最终构造的正例：

![](https://raw.githubusercontent.com/zy-Pioneer/BlogImage/main/img/2025/01/7b3d5c13.png)



## Overall Loss function
![](https://raw.githubusercontent.com/zy-Pioneer/BlogImage/main/img/2025/01/43e0f731.png)

我们的目标函数旨在最小化查询节点 v<sub>i</sub> 与其正例 P<sub>i</sub> 之间的余弦距离，这里的 

![](https://raw.githubusercontent.com/zy-Pioneer/BlogImage/main/img/2025/01/6f883bb9.png)

