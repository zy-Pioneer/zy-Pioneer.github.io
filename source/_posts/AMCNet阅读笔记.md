---
title: AMCNet阅读笔记
date: 2024-08-15
updated: 2024-08-15
tags: [动态图神经网络]
categories: [论文笔记]
---

## Abstract
动态链接预测（在动态图中，预测两个点在时间t是否存在边）关注两个信息：结构信息和时间信息。现有工作要么只考虑其中一个因素，要么两个因素都单独考虑而没有将它们联系起来，忽略了两者之间的关联性。

因此本文提出了一种方法，**多尺度注意力协同进化网络（Attentional Multi-scale Co-evolving Network）**，使用多层次注意力机制的序列到序列的模型（**sequence-to-sequence**）来学习不同层次之间的动态变化的内在关系。

## Introduction
![](https://raw.githubusercontent.com/zy-Pioneer/BlogImage/main/img/2025/01/9a2f5847.png)

本文中将图中的结构分为三个层次：

+ 微观层次（Microscopic structure）：图中的节点和边
+ 中观层次（Mesoscopic structure）：图中的子图或者社区
+ 宏观层次（Macroscopic structure）：整张图

本文认为网络是由个体及其相互联系构成的，因此宏观时间动态自然取决于每个个体如何选择与他人建立联系的微观时间动态。另一方面，研究表明，人类行为受到社会关系的动态影响，例如政治取向、音乐品味，甚至人们如何选择新朋友。（中观影响微观，微观影响中观和宏观）

### contribution
+ 设计了多层次表示学习模块（通过GNN学习到节点表示，再通过节点表示来设计pooling表示中观和宏观），为了表示中观，提出了一个`motifs`的概念
+ 提出了一个多层次协同进化模型来学习每个层次的动态特征
+ 为了了解不同结构尺度的时间动态之间的内在相关性，通过一种新颖的基于注意力的分层模型，利用较高尺度的表示来指导较低尺度表示的学习过程。

> motifs简介
>
> motifs是一种模式，如下图所示：
>
> ![](https://raw.githubusercontent.com/zy-Pioneer/BlogImage/main/img/2025/01/d203283f.png)
>
> 它表示一种子图结构，如何去判定一个子图结构是否为motif呢？有两个指标：频率和重要性。
>
> 频率指的是它在图中出现的频率，重要性通过随机构造一张随机图，判断真实图种motif出现的频率和随机图出现的频率的差异来判断重要性，差异越大，越重要
>
> 这个motif的发现有现成的工作来做，本文中并没有涉及构造motif的代码，它直接用了
>

## method
![](https://raw.githubusercontent.com/zy-Pioneer/BlogImage/main/img/2025/01/f7a04ef9.png)

整个架构分为两部分：

1. **multiscale representation learning module**
2. **multi-scale evolving module.**

### Multi-scale Representation Learning
#### Microscopic Representation
用GAT来生成节点表示（就是MAGIC的骨架网络）

#### Mesoscopic Representation
作者一开始尝试，通过对motifs中的节点做平均池化来获得中观表示，但是效果不好，因此不单纯的做平均，加上了一个可学习的参数矩阵来为每个motif作为注意力参数

![](https://raw.githubusercontent.com/zy-Pioneer/BlogImage/main/img/2025/01/a089c176.png)

![](https://raw.githubusercontent.com/zy-Pioneer/BlogImage/main/img/2025/01/faf0efb3.png)

但是很奇怪，根据公式三，这样得到的表示不就是一个值了吗，我以为是每个motif都计算一个值作为中观表示

#### Macroscopic Representation
全局的宏观表示就是将所有节点的特征平均一下：

![](https://raw.githubusercontent.com/zy-Pioneer/BlogImage/main/img/2025/01/7dd9d57b.png)

### Multi-scale Co-evolving module
本文提出了两个观点：

1. 首先，从信息论的角度来看，数据中的噪声量和信息聚合水平通常呈负相关。结构层次越高，信息越少，噪声也越小。因此，较高结构尺度的时间动态比较低结构尺度更可预测。因此，较高尺度的预测有助于纠正较低尺度预测的潜在系统偏差，这对学习模型施加了尺度不变的约束。
2. 其次，不同结构尺度的信息捕获了图的不同特征，从而相互补充。对不同尺度的时间动态进行联合建模使模型能够利用来自不同上下文范围的信息来进行预测。

#### Impletation
**Sequence to Sequence Backbone：**使用序列模型来捕获每个层次的数据的内在结构，设计了三个：

1. **seq-seq-node：**Seq2Seq_Attention(enc, dec, dev)
2. **seq-seq-motif：**Seq2Seq_Attention(enc, dec, dev)
3. **seq-seq-graph：**Seq2Seq(enc, dec, dev)

从graph层次开始，graph指导motif，motif指导node：

![](https://raw.githubusercontent.com/zy-Pioneer/BlogImage/main/img/2025/01/b681fa65.png)



其中**seq2seq:**

```python
class Seq2Seq(torch.nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        assert encoder.hidden_size == decoder.hidden_size, "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers,       "Encoder and decoder must have equal number of layers!"


    def forward(self, x, y, teacher_forcing_ratio = 0.75):
```

其中**Seq2Seq_Attention:**

```python
class Seq2Seq_Attention(torch.nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.attention = nn.Linear(encoder.hidden_size*2, encoder.hidden_size)
        self.softmax = nn.Softmax(dim=1)

        assert encoder.hidden_size == decoder.hidden_size, "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers,       "Encoder and decoder must have equal number of layers!"
        nn.init.xavier_normal_(self.attention.weight)
        nn.init.constant_(self.attention.bias, 0.0)

    def forward(self, x, y,hiddens, teacher_forcing_ratio = 0.75):
```



**encoder（LSTM）：**

```python
class Encoder(torch.nn.Module):
    def __init__(self,
                input_size = 128,
                embedding_size = 128,
                hidden_size = 256,
                n_layers = 2,
                dropout = 0.5):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.linear = nn.Linear(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, n_layers,
                        dropout = dropout)
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_normal_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0.0)
```

**decoder（LSTM）：**

```python
class Decoder(torch.nn.Module):
    def __init__(self,
                output_size = 256,
                embedding_size = 128,
                hidden_size = 256,
                n_layers = 4,
                dropout = 0.5):
        super().__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.embedding = nn.Linear(output_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, n_layers, dropout = dropout)
        self.linear = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

        nn.init.xavier_normal_(self.embedding.weight)
        nn.init.constant_(self.embedding.bias, 0.0)
        nn.init.xavier_normal_(self.linear.weight)
        nn.init.constant_(self.linear.bias, 0.0)
```



由于是联合优化，因此总体Model如下：

```python
class Model(torch.nn.Module):
    def __init__(self, seq2seq_node, seq2seq_motif,seq2seq_graph, device):
        super().__init__()
        self.seq2seq_node = seq2seq_node
        self.seq2seq_motif = seq2seq_motif
        self.seq2seq_graph = seq2seq_graph
        self.device = device

    def forward(self, node_x, motif_x,graph_x,node_y,motif_y,graph_y, teacher_forcing_ratio = 0.75):
        graph_emb,graph_hiddens=seq2seq_graph(graph_x,graph_y,teacher_forcing_ratio)
        motig_emb,motif_hiddens=seq2seq_motif(motif_x,motif_y,graph_hiddens,teacher_forcing_ratio)
        node_emb,_ =seq2seq_node(node_x,node_y,motif_hiddens,teacher_forcing_ratio)
        final_emb=torch.cat((node_emb,motig_emb,graph_emb),axis=2)
        return final_emb
```

## Summary
很有启发性的一篇工作，总体思路就是在AE的基础上，串联多层次AE，来融合不同层次的语义信息（通过在不同层次之间，将上层的输出来生成下层的注意力，进行语义融合）。

文章的writting和画图都很棒，很值得借鉴，就是开源的代码不完全，GAT部分没有给出，无法复现。



