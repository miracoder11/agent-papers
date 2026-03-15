# Attention Is All You Need (Transformer)

## 层 1：电梯演讲

**一句话概括**：Google 团队在 2017 年提出完全基于注意力机制的 Transformer 架构，抛弃了 RNN 和 CNN 的序列建模方式，在机器翻译任务上超越所有已有模型（包括集成模型），训练时间从数周缩短到 12 小时，开启了大语言模型时代。

---

## 层 2：故事摘要

### 背景：2017 年的困境

时间回到 2017 年，序列建模任务（如机器翻译）被 RNN、LSTM 和 GRU 牢牢统治。这些模型有一个致命缺陷：**序列计算无法并行化**。

想象一下这个场景：
- 你要翻译一个 100 词的句子
- RNN 必须按顺序处理：第 1 个词 → 第 2 个词 → ... → 第 100 个词
- 即使你有 100 个 GPU，也只能用 1 个，因为第 2 步必须等第 1 步完成
- 训练一个最先进的翻译模型需要数周时间

研究者们的焦虑：
> "为什么我们必须按顺序处理？人类阅读句子时并不需要一个字一个字地读啊！"

### 核心洞察

Google Brain 团队的 Jakob Uszkoreit 提出了一个大胆的想法：
**"如果完全不要 RNN，只用 attention 机制，会怎样？"**

这个想法在当时看起来非常激进——attention 只是 RNN 的辅助工具，怎么可能完全替代 RNN？

### Transformer 的诞生

Ashish Vaswani 和 Illia Polosukhin 设计并实现了第一个 Transformer 模型。关键突破：

```
┌─────────────────────────────────────┐
│     问题：RNN 无法并行化             │
│          长距离依赖学习困难          │
└─────────────┬───────────────────────┘
              │
┌─────────────▼───────────────────────┐
│     洞察：Attention 可以建立任意    │
│     位置间的直接连接                 │
└─────────────┬───────────────────────┘
              │
┌─────────────▼───────────────────────┐
│    方案：完全抛弃 RNN，只用         │
│    Multi-Head Self-Attention        │
└─────────────┬───────────────────────┘
              │
┌─────────────▼───────────────────────┐
│    结果：训练时间从数周→12 小时       │
│          BLEU 分数超越所有 SOTA      │
└─────────────────────────────────────┘
```

### 实验结果震撼学界

| 模型 | EN-DE BLEU | EN-FR BLEU | 训练时间 |
|------|-----------|-----------|----------|
| GNMT + RL Ensemble | 26.30 | 41.16 | 数周 |
| ConvS2S Ensemble | 26.36 | 41.29 | 数天 |
| **Transformer (big)** | **28.4** | **41.8** | **3.5 天** |

Transformer 不仅质量更高，训练成本只有之前模型的几十分之一。

---

## 层 3：深度精读

### 开场：一个失败的场景

让我们回到 2016 年，Google 的翻译系统 GNMT 刚刚上线。工程师们遇到了一个头疼的问题：

**场景**：翻译这个句子
> "The animal didn't cross the street because **it** was too tired."

问题：代词 "it" 指的是什么？

- LSTM 模型需要从头读到 "it"，然后记住前面的 "animal"
- 如果句子更长，比如 "it" 出现在第 50 个词，而 "animal" 在第 1 个词
- LSTM 的隐状态早就被中间 48 个词的信息"淹没"了

这就是**长距离依赖问题**。RNN 的设计决定了它难以捕捉相距遥远的词之间的关系。

研究者们尝试过各种方法：
- 更大的隐层维度 → 过拟合
- 更多的层数 → 梯度消失
- LSTM/GRU → 有改善，但根本问题未解决

直到...

### 第一章：Attention 机制的前世今生

**2014 年**，Bahdanau 等人在机器翻译中引入了 Attention 机制。

**Attention 的直观理解**：

想象你在阅读这句话：
> "The cat sat on the mat because **it** was comfortable."

当你看到 "it" 时，你的大脑会自动：
1. 向前查找可能的指代对象
2. 发现 "cat" 和 "mat" 都是候选
3. 根据语义（comfortable 通常形容生物）判断 "it" 指的是 "cat"

这个过程就是 attention——**有选择地关注输入的不同部分**。

**旧范式中 Attention 的角色**：
```
┌─────────────┐    ┌─────────────┐
│   Encoder   │───→│   Decoder   │
│   (LSTM)    │    │   (LSTM)    │
└─────────────┘    └─────────────┘
        │                  │
        └────Attention────┘
        (辅助工具)
```

Attention 只是 RNN 的"配角"。

**Transformer 的革命**：
```
┌─────────────────────────────────┐
│         Transformer             │
│  ┌─────────────────────────┐    │
│  │   Encoder (纯 Attention) │    │
│  └─────────────────────────┘    │
│  ┌─────────────────────────┐    │
│  │   Decoder (纯 Attention) │    │
│  └─────────────────────────┘    │
└─────────────────────────────────┘
```

Attention 从配角变成了唯一的主角。

### 第二章：Transformer 架构详解

#### 2.1 整体架构

```
                    ┌─────────────────────────────────────┐
                    │           Transformer               │
                    └─────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
          ┌─────────────────┐                 ┌─────────────────┐
          │    Encoder      │                 │    Decoder      │
          │   Stack × 6     │                 │   Stack × 6     │
          └─────────────────┘                 └─────────────────┘
                    │                                   │
          ┌─────────┴─────────┐           ┌─────────────┴─────────────┐
          ▼                   ▼           ▼             ▼             ▼
   ┌──────────────┐  ┌──────────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
   │ Multi-Head   │  │ Feed-Forward │ │ Masked   │ │ Encoder- │ │ Feed-    │
   │ Self-Attn    │  │ Network      │ │ Self-Attn│ │ Decoder  │ │ Forward  │
   │              │  │              │ │          │ │ Attn     │ │ Network  │
   └──────────────┘  └──────────────┘ └──────────┘ └──────────┘ └──────────┘
```

**Encoder 结构**（6 层相同）：
```
输入嵌入 → [Multi-Head Self-Attention + Residual + LayerNorm]
         → [Feed-Forward Network + Residual + LayerNorm] → 输出
```

**Decoder 结构**（6 层相同）：
```
输出嵌入 → [Masked Multi-Head Self-Attention + Residual + LayerNorm]
         → [Encoder-Decoder Attention + Residual + LayerNorm]
         → [Feed-Forward Network + Residual + LayerNorm] → 输出
```

#### 2.2 为什么是这种设计？

**研究者的思维历程**：

**第一版尝试**：直接用 attention 替换 RNN
```python
# 天真版本
output = attention(query, keys, values)
```
问题：信息流太浅，学不到复杂模式

**第二版尝试**：堆叠多层
```python
# 多层版本
h1 = attention(x, x, x)
h2 = attention(h1, h1, h1)
h3 = attention(h2, h2, h2)
```
问题：梯度不稳定，训练困难

**最终版本**：引入残差连接和 LayerNorm
```python
# Transformer 版本
h1 = LayerNorm(x + MultiHeadAttention(x, x, x))
h2 = LayerNorm(h1 + FeedForward(h1))
```
成功了！训练稳定，效果卓越

### 第三章：核心概念 - 大量实例

#### 3.1 Scaled Dot-Product Attention

**定义**：
```
Attention(Q, K, V) = softmax(QK^T / √dk) V
```

**生活类比 1：图书检索系统**

想象你在图书馆找书：
- **Query (Q)**：你想找的书的主题（如"机器学习"）
- **Keys (K)**：所有书的分类标签
- **Values (V)**：所有书的实际内容

过程：
1. 计算你的查询和每个标签的匹配度（QK^T）
2. 用 softmax 转换成概率分布（越匹配的书权重越高）
3. 加权平均所有书的内容（得到最相关的信息）

**为什么需要缩放因子 1/√dk？**

这是一个关键但容易被忽视的细节。

**问题场景**：
假设 dk = 512，每个维度的值都是均值为 0、方差为 1 的随机变量。
- 点积结果 = 512 个独立变量的和
- 方差 = 512，标准差 ≈ 22.6
- 点积可能达到 ±50 甚至更大

**softmax 的问题**：
```
softmax(x) = e^x / Σ e^xi

当 x = 50 时：e^50 ≈ 5.2 × 10^21
当 x = -50 时：e^-50 ≈ 1.9 × 10^-22
```
softmax 会进入"饱和区"，梯度几乎为 0，模型无法学习！

**解决方案**：
除以 √dk = √512 ≈ 22.6，将点积缩放到合理的范围。

**代码实例**：
```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V):
    """
    Q, K, V: (batch_size, seq_len, d_k)
    """
    d_k = Q.size(-1)

    # 计算注意力分数
    scores = torch.matmul(Q, K.transpose(-2, -1))  # (batch, seq_len, seq_len)

    # 缩放
    scores = scores / (d_k ** 0.5)

    # 应用 softmax
    attention_weights = F.softmax(scores, dim=-1)

    # 加权求和
    output = torch.matmul(attention_weights, V)

    return output, attention_weights
```

#### 3.2 Multi-Head Attention

**为什么需要多头？**

**类比 2：多专家团队**

想象一个医疗诊断团队：
- 专家 A 专注于症状
- 专家 B 专注于病史
- 专家 C 专注于检查结果
- 专家 D 专注于基因信息

每个专家从不同角度分析病人，然后综合所有人的判断。

**Multi-Head Attention 也是如此**：
- Head 1 可能关注语法结构（主谓关系）
- Head 2 可能关注指代关系（代词指什么）
- Head 3 可能关注语义角色（谁对谁做了什么）
- Head 4 可能关注情感色彩

**数学公式**：
```
MultiHead(Q, K, V) = Concat(head₁, head₂, ..., headₕ) W^O

其中 headᵢ = Attention(QWᵢ^Q, KWᵢ^K, VWᵢ^V)
```

**参数维度**：
- Wᵢ^Q ∈ R^(d_model × d_k)
- Wᵢ^K ∈ R^(d_model × d_k)
- Wᵢ^V ∈ R^(d_model × d_v)
- W^O ∈ R^(h·d_v × d_model)

**代码实例**：
```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # 64

        # 线性投影
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)

        # 线性投影并分割头
        q = self.W_q(Q).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.W_k(K).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.W_v(V).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        # (batch, num_heads, seq_len, d_k)

        # 并行计算所有头的 attention
        attn_output, attn_weights = scaled_dot_product_attention(q, k, v, mask)

        # 拼接头
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # 最终线性投影
        output = self.W_o(attn_output)

        return output, attn_weights
```

**实例：不同头关注不同信息**

论文中的可视化展示了 Multi-Head 的强大：

**示例 1：长距离依赖**
```
句子："It is in this spirit that a majority of American governments
      have passed new laws since 2009 making the registration or
      voting process more difficult."

某个 attention head 在 "making" 位置：
making → the registration/voting process (强关联)
making → difficult (强关联)

这个头学会了捕捉 "making...difficult" 这个短语结构。
```

**示例 2：指代消解**
```
句子："The Law will never be perfect, but its application should
      be just - this is what we are missing, in my opinion."

某个 attention head 在 "its" 位置：
its → The Law (强关联，概率 > 0.9)

这个头学会了解决代词指代问题。
```

#### 3.3 Position-wise Feed-Forward Networks

**结构**：
```
FFN(x) = max(0, xW₁ + b₁)W₂ + b₂

其中 W₁ ∈ R^(d_model × d_ff), W₂ ∈ R^(d_ff × d_model)
d_model = 512, d_ff = 2048
```

**类比 3：信息提炼器**

如果把 attention 比作"信息收集器"（从不同位置收集信息），那么 FFN 就是"信息提炼器"：
1. 将收集到的信息投影到更高维空间（512 → 2048）
2. 用 ReLU 进行非线性变换，提取关键特征
3. 投影回原维度（2048 → 512）

**为什么需要 FFN？**

attention 是线性的（加权求和），FFN 提供非线性，增强模型的表达能力。

#### 3.4 Positional Encoding

**问题**：没有 RNN/CNN，如何知道词的位置？

**Transformer 的方案**：
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**直观理解**：

**类比 4：给每个位置一个独特的"指纹"**

每个位置得到一个由不同频率正弦波组成的编码：
- 低频波：粗略定位（前/中/后）
- 高频波：精确定位（具体哪个词）

```
位置 0: [sin(0), cos(0), sin(0.01), cos(0.01), ...]
位置 1: [sin(1), cos(1), sin(1.01), cos(1.01), ...]
位置 2: [sin(2), cos(2), sin(2.01), cos(2.01), ...]
```

**为什么选择正弦函数？**

关键洞察：**相对位置可以通过线性变换表示**。

对于任意固定偏移 k：
```
PE(pos+k) 可以表示为 PE(pos) 的线性函数
```

这意味着模型可以轻松地学会"关注相对位置"，比如"前面第 3 个词"。

**代码实例**：
```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]
```

#### 3.5 三种 Attention 的应用

| 类型 | Query 来源 | Key/Value 来源 | 作用 |
|------|----------|---------------|------|
| Encoder Self-Attention | Encoder 前一层 | Encoder 前一层 | 每个位置关注所有输入位置 |
| Decoder Self-Attention | Decoder 前一层 | Decoder 前一层 | 每个位置关注之前位置（masking） |
| Encoder-Decoder Attention | Decoder 前一层 | Encoder 输出 | Decoder 关注所有输入位置 |

**Decoder Masking 详解**：

为了防止 Decoder 在预测位置 i 时"偷看"未来位置，需要 masking：
```
在 softmax 前，将未来位置的分数设为 -∞

示例（4 个位置）：
    0  1  2  3
0 [ 0 -∞ -∞ -∞ ]  # 位置 0 只能看到 0
1 [ 0  0 -∞ -∞ ]  # 位置 1 能看到 0, 1
2 [ 0  0  0 -∞ ]  # 位置 2 能看到 0, 1, 2
3 [ 0  0  0  0 ]  # 位置 3 能看到 0, 1, 2, 3
```

### 第四章：预期 vs 实际

读到这里，你可能有一些直觉预测。让我们看看实际情况：

#### 4.1 直觉预测 vs Transformer 实现

| 维度 | 你的直觉/预期 | Transformer 实际 | 为什么？ |
|------|--------------|-----------------|----------|
| **注意力头数** | 越多越好 | 8 个头最好，太多反而差 | 头太多会稀释每个头的学习能力 |
| **Key 维度 dk** | 越小越快 | dk 太小 hurts 质量 | 计算 compatibility 需要足够容量 |
| **位置编码** | 必须学习 | 固定正弦函数就够 | 相对位置的线性性质已足够 |
| **层数** | 越深越好 | 6 层平衡效果和效率 | 更深收益递减，训练更困难 |
| **FFN 维度** | 和 d_model 一致 | d_ff = 4×d_model | 更大的中间层增强表达能力 |

#### 4.2 计算复杂度对比

| 层类型 | 每层复杂度 | 顺序操作 | 最大路径长度 |
|--------|----------|----------|-------------|
| Self-Attention | O(n²·d) | O(1) | O(1) |
| RNN | O(n·d²) | O(n) | O(n) |
| CNN | O(k·n·d²) | O(1) | O(log_k(n)) |

**关键洞察**：
- Self-Attention 的**顺序操作是 O(1)**，可以完全并行
- RNN 必须按顺序处理，无法利用并行计算
- 这就是为什么 Transformer 训练快几十倍

### 第五章：反直觉挑战

#### 挑战 1：去掉所有 RNN，只用 Attention？

**预测**：这怎么可能？RNN 是序列建模的基础！

**实际**：不仅可行，而且效果更好！

**为什么**：
- RNN 的"序列建模"能力本质上是为了捕捉依赖关系
- Attention 可以直接连接任意两个位置，依赖关系捕捉能力更强
- RNN 的序列性只是历史遗留，不是必需

#### 挑战 2：正弦位置编码能工作？

**预测**：位置信息这么重要，肯定要学习！

**实际**：固定正弦函数和学习的结果几乎一样！

| 位置编码 | BLEU |
|---------|------|
| 正弦函数 | 25.8 |
| 学习得到 | 25.7 |

**为什么**：
- 正弦函数的关键性质是相对位置可以线性表示
- 这个性质已经足够模型学会位置关系
- 学习的编码并没有额外的优势

#### 挑战 3：Single-Head vs Multi-Head

**预测**：一个头就够了吧？多头只是增加参数量？

**实际**：Single-Head BLEU 下降 0.9！

| 头数 | d_k | BLEU |
|-----|-----|------|
| 1 | 512 | 24.9 |
| 8 | 64 | 25.8 |
| 16 | 32 | 25.8 |
| 32 | 16 | 25.4 |

**为什么**：
- 单头 attention 会对所有位置进行平均
- 多头允许模型同时关注不同子空间的信息
- 就像多专家决策 vs 单专家决策

### 第六章：关键实验细节

#### 6.1 训练设置

**数据**：
- WMT 2014 English-German: 4.5M 句对
- WMT 2014 English-French: 36M 句对
- BPE 词表：37K (DE), 32K (FR)

**硬件**：
- 8 × NVIDIA P100 GPU
- Base 模型：100K steps, 12 小时
- Big 模型：300K steps, 3.5 天

**Optimizer**：
```python
lrate = d_model^(-0.5) × min(step_num^(-0.5), step_num × warmup_steps^(-1.5))
warmup_steps = 4000
```

**学习率曲线**：
```
  lrate
    │     ╱│
    │    ╱ │
    │   ╱  │
    │  ╱   │╲
    │ ╱    │ ╲
    │╱     │  ╲
    └──────┴───┴── step
         4000
```
- 前 4000 步线性增加
- 之后按 step^(-0.5) 衰减

**正则化**：
- Dropout: 0.1
- Label Smoothing: 0.1

#### 6.2 模型变体实验

**Table 3 深度解读**：

**(A) 注意力头数变化**（保持计算量恒定）
```
1 头 (d_k=512)  → 24.9 BLEU
4 头 (d_k=128)  → 25.5 BLEU
8 头 (d_k=64)   → 25.8 BLEU  ✓ 最佳
16 头 (d_k=32)  → 25.8 BLEU
32 头 (d_k=16)  → 25.4 BLEU
```
结论：太多头会稀释每个头的表达能力。

**(B) Key 维度变化**
```
d_k = 16  → 25.1 BLEU
d_k = 32  → 25.4 BLEU
d_k = 64  → 25.8 BLEU  ✓
```
结论：计算 compatibility 需要足够的容量。

**(C) 模型大小变化**
```
d_model = 256   → 24.5 BLEU
d_model = 512   → 25.8 BLEU
d_model = 1024  → 26.0 BLEU
```
结论：越大越好，但收益递减。

**(D) Dropout 变化**
```
dropout = 0.0  → 24.6 BLEU
dropout = 0.1  → 25.8 BLEU  ✓
dropout = 0.2  → 25.5 BLEU
```
结论：Dropout 对防止过拟合非常有效。

### 第七章：与其他方法对比

#### 7.1 机器翻译结果对比

| 模型 | EN-DE BLEU | EN-FR BLEU | 训练成本 (FLOPs) |
|------|-----------|-----------|-----------------|
| ByteNet | 23.75 | - | - |
| Deep-Att + PosUnk | - | 39.2 | 1.0×10²⁰ |
| GNMT + RL | 24.6 | 39.92 | 2.3×10¹⁹ |
| ConvS2S | 25.16 | 40.46 | 9.6×10¹⁸ |
| MoE | 26.03 | 40.56 | 2.0×10¹⁹ |
| **Transformer (base)** | **27.3** | **38.1** | **3.3×10¹⁸** |
| GNMT + RL Ensemble | 26.30 | 41.16 | 1.8×10²⁰ |
| ConvS2S Ensemble | 26.36 | 41.29 | 7.7×10¹⁹ |
| **Transformer (big)** | **28.4** | **41.8** | **2.3×10¹⁹** |

**关键洞察**：
- Transformer (base) 超越了所有单模型和集成模型
- 训练成本只有 GNMT 的 1/7
- Transformer (big) 刷新 SOTA，训练成本只有之前模型的 1/5

#### 7.2 泛化能力：英文句法分析

| Parser | 训练数据 | WSJ 23 F1 |
|--------|---------|----------|
| Vinyals & Kaiser (2014) | WSJ only | 88.3 |
| Petrov et al. (2006) | WSJ only | 90.4 |
| Dyer et al. (2016) RNN | WSJ only | 91.7 |
| **Transformer (4 层)** | WSJ only | **91.3** |
| Dyer et al. (2016) RNN | semi-supervised | 92.1 |
| **Transformer (4 层)** | semi-supervised | **92.7** |

结论：Transformer 可以泛化到其他任务，无需特定调优。

#### 7.3 局限性

**Transformer 的局限**：
1. **自注意力复杂度 O(n²)**：长序列时内存消耗大
2. **没有显式记忆机制**：无法跨样本学习
3. **位置编码外推能力有限**：超过训练长度的序列效果下降

**后续改进方向**：
- **Reformer**：用 LSH attention 降低复杂度到 O(n log n)
- **FlashAttention**：IO 感知的精确 attention，加速并减少内存
- **RoPE**：旋转位置编码，更好的外推能力

### 第八章：如何应用

#### 8.1 适用场景

✅ **适合使用 Transformer**：
- 序列到序列任务（翻译、摘要、问答）
- 需要捕捉长距离依赖
- 有充足训练数据
- 需要快速训练

❌ **不适合使用 Transformer**：
- 超短序列（<10 tokens），RNN 可能更简单
- 资源极度受限的边缘设备
- 严格实时性要求（attention 的 O(n²) 可能太重）

#### 8.2 自己实现 Transformer

**最小可用实现**：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Transformer(nn.Module):
    def __init__(self, num_encoder_layers=6, num_decoder_layers=6,
                 d_model=512, nhead=8, d_ff=2048, dropout=0.1):
        super().__init__()

        self.encoder = EncoderStack(num_encoder_layers, d_model, nhead, d_ff, dropout)
        self.decoder = DecoderStack(num_decoder_layers, d_model, nhead, d_ff, dropout)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        memory = self.encoder(src, src_mask)
        output = self.decoder(tgt, memory, tgt_mask)
        return output

class EncoderStack(nn.Module):
    def __init__(self, num_layers, d_model, nhead, d_ff, dropout):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, nhead, d_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention + residual
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # FFN + residual
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))

        return x
```

### 第九章：延伸思考

#### 苏格拉底式追问

1. **如果去掉残差连接，Transformer 还能训练吗？**
   - 不能。没有残差连接，深层 Transformer 梯度会消失。

2. **为什么 LayerNorm 用在 residual 之后而不是之前？**
   - 论文用 Post-LN，后续研究发现 Pre-LN 更稳定。

3. **Multi-Head 注意力真的学到了不同的"视角"吗？**
   - 论文可视化显示是的，不同头关注不同语法/语义关系。

4. **正弦位置编码为什么能外推到更长序列？**
   - 因为相对位置可以线性表示，模型学到的是相对关系。

5. **如果让你设计一个新的 attention 变体，你会怎么做？**
   - 思考方向：降低 O(n²) 复杂度、增强长距离依赖、引入结构先验...

#### 论文定位图谱

```
                    Attention Is All You Need (2017)
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
   上游基础              并行工作              下游发展
   ────────              ────────              ────────
   Bahdanau Attention   ConvS2S (2017)        BERT (2018)
   (2014)               ByteNet (2016)        GPT 系列 (2018-)
   Luong Attention                           Transformer-XL
   (2015)                                    Reformer (2020)
   Self-Attention                            FlashAttention
   (2015-2016)                               RoPE (2021)
```

---

## 结语

**Attention Is All You Need** 不仅仅是一篇论文，它开启了一个时代。

从 2017 年到 2026 年，Transformer 架构统治了 NLP 领域，并扩展到视觉（ViT）、语音、多模态等各个领域。

回顾这篇论文，最重要的启示是：
> **有时候，打破常规思维，彻底抛弃旧范式，才能带来真正的突破。**

当你下次遇到"一直都是这样做"的问题时，不妨问问自己：
> "如果完全不要 X，只用 Y，会怎样？"

也许下一个 Transformer 级别的突破就在这样的思考中诞生。
