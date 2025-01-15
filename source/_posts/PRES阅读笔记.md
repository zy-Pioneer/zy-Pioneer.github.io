---
title: PRES阅读笔记
date: 2025-01-16 01:51:46
updated: 2025-01-16 01:51:46
tags: [图神经网络]
categories: [论文笔记] 
---

## 摘要
基于内存的动态图神经网络 (MDGNN)十分有效，但是存在一个比较致命的问题：

**在一个时间窗口内的事件，其中的时间依赖关系无法捕捉**



## 背景知识（TGN的memory机制解释）
之前实现的MAGIC，是在静态图上进行处理，一次性加载完所有的数据。而动态图的特点在于动态，图的信息是不断扩展动态变化的，图并不是一次全部加载完全，而是每次只加载一个时间窗口中的事件，也就是边。

那么如何去捕获每个窗口之间的信息呢，在TGN中，它通过在每个时间窗口中计算节点的状态（这个状态由当前窗口中的事件以及上一时刻的状态决定），这个状态就称作memory，这个memory就代表了这个节点的历史信息，这样就能够捕获每个节点从出生到当前状态的所有信息。

下面详细解释一下节点memory的更新过程：

![图1](https://raw.githubusercontent.com/zy-Pioneer/BlogImage/main/img/2025/01/92c8f93e.png)

![图2](https://raw.githubusercontent.com/zy-Pioneer/BlogImage/main/img/2025/01/2d88de91.png)

如图2所示，整个TGN的核心就在这个TGNMemory，它维护了三个结构，分别是memory、msg_s_store（源节点的msg存储）和msg_d_store（目标节点的msg存储）。先计算msg：对于batch中的src和dst分别计算一次msg（视为无向图），每次msg计算如下：

```python
raw_msg = torch.cat(raw_msg, dim=0)
t_rel = t - self.last_update[src]
t_enc = self.time_enc(t_rel.to(raw_msg.dtype))

msg = msg_module(self.memory[src], self.memory[dst], raw_msg, t_enc)
```

这里的raw_msg就是msg_s_store或msg_d_store，取决于是计算src的msg还是dst的msg



计算完msg之后，在当前时间窗口中，对每个节点聚合和它交互的边，这个聚合操作可以自定义，在TGN中它的实现是选择最新的那条边，其他边就扔掉了（这也就是PERS这篇论文中想解决的问题，处理一个时间窗口中的时间顺序关系），将选择的那条边对应的msg和当前状态的memory一起送进GRU（一种循环神经网络，可以捕获msg和memory的序列特征）中，来更新memory。



## Introduction
### 现有工作存在的问题
1. MDGNN无法维持同一批次（时间窗口）内的数据点之间的时间依赖性，因此常常选择较小的时间窗口。
2. 较小的时间窗口无法有效利用计算资源，因此解决批量大小瓶颈对于计算资源的利用至关重要。

![](https://raw.githubusercontent.com/zy-Pioneer/BlogImage/main/img/2025/01/f616b3f1.png)

### 本文完成的工作
1. 进行了数学分析，证明了大时间窗口比小时间窗口更好（全文基本上都是数学分析，公式占了一半的篇幅，实在是看不太明白）
2. 设计了一种新颖的框架来改善小时间窗口的问题
3. 进行了实验，证明了本文提出的工作效果好



## PERS METHOD
### 方法总览
![](https://raw.githubusercontent.com/zy-Pioneer/BlogImage/main/img/2025/01/02ce2897.png)

![](https://raw.githubusercontent.com/zy-Pioneer/BlogImage/main/img/2025/01/4076c791.png)

整篇文章的目的是对上式的右半边进行优化，右半边的值越小越好，其中最具影响力的两个因素是μ和σ<sub>max</sub>，因此本文就是为了去增大μ并减小σ<sub>max</sub>，提出了两个方向的改进措施<sup></sup>。

+ ITERATIVE PREDICTION-CORRECTION SCHEME（优化第二项）
+ MEMORY COHERENCE SMOOTHING（优化第一项）

伪代码如下：

![](https://raw.githubusercontent.com/zy-Pioneer/BlogImage/main/img/2025/01/bb9aa082.png)

### ITERATIVE PREDICTION-CORRECTION SCHEME（迭代预测校正）
#### 概念定义：pending events
![](https://raw.githubusercontent.com/zy-Pioneer/BlogImage/main/img/2025/01/8ad26fbb.png)



#### 背景知识：Gaussian Mixture Model
 GMM 假设数据点由多个高斯分布混合生成。每个高斯分布称为一个成分（component）。GMM 的概率密度函数是这些成分的加权和：  

![](https://raw.githubusercontent.com/zy-Pioneer/BlogImage/main/img/2025/01/20f8e7aa.png)

因此GMM的参数包括：

+ 每个成分的均值向量 μ<sub>i</sub> 
+  每个成分的协方差矩阵 Σ<sub>i</sub> 
+  每个成分的权重 π<sub>i</sub> 

接着使用 期望最大化（EM）算法来更新参数，其主要包含两个主要步骤：

+ **E 步（期望步）**：计算每个数据点属于每个成分的概率，即责任度（responsibility）。
+ **M 步（最大化步）**：根据 E 步计算的责任度，重新估计参数。

![](https://raw.githubusercontent.com/zy-Pioneer/BlogImage/main/img/2025/01/e8cddf5d.png)

#### 纠错机制
设置GMM的成分数为2，由于该论文中的场景是进行链接预测，因此它会进行正负采样，所以这里设置成分数为2，不太懂这个是怎么训练的，但是最终，这个GMM的作用是预测δ<sub>si</sub> 的分布，通过这个分布可以每次去生成一个预测的节点状态：

![](https://raw.githubusercontent.com/zy-Pioneer/BlogImage/main/img/2025/01/774320c1.png)

之后为了拟合真实的考虑了时间的节点状态，作者使用参数γ去中和：

![](https://raw.githubusercontent.com/zy-Pioneer/BlogImage/main/img/2025/01/e30f87e2.png)

这时得到的值就是进行纠错之后的节点状态，那么为什么是上面两个公式这样的形式呢，作者在附录中进行了推导（作者说明了这里的符号和论文其他部分不一致），证明了：

![](https://raw.githubusercontent.com/zy-Pioneer/BlogImage/main/img/2025/01/6eeb6afa.png)

![](https://raw.githubusercontent.com/zy-Pioneer/BlogImage/main/img/2025/01/3f208881.png)

也就是表明本文方法计算出的状态相对于未改进之前，更接近真实值，优化了方差的大小（不懂那个方差是啥，但它说是就是吧）

### MEMORY COHERENCE SMOOTHING
![](https://raw.githubusercontent.com/zy-Pioneer/BlogImage/main/img/2025/01/14c6d51b.png)

这个内存一致性怎么计算的看不懂，也不太能看懂是个啥，看它的描述，上半部分应该是考虑了pending events时（考虑了时间依赖性）去更新节点状态计算的梯度和未考虑pending events去更新节点状态计算的梯度进行点积。也就是说，这个一致性在直观上来讲，它度量了理想的梯度下降方向和实际的梯度下降方向的一致程度。当它是正值的时候，表示方向一致（可能参数都是增加或者较小？），负值就表示方向相反。



