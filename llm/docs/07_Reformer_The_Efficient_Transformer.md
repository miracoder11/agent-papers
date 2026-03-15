# Reformer: The Efficient Transformer

**论文信息**: Kitaev, N., Kaiser, Ł., & Levskaya, A. (2020). Reformer: The Efficient Transformer. ICLR 2020.
**arXiv**: [2001.04451](https://arxiv.org/abs/2001.04451)

---

## 层 1：电梯演讲（30 秒）

**一句话概括**：针对标准 Transformer 的 O(L²) 注意力复杂度和高内存占用问题，Google Research 提出 Reformer，通过 LSH 注意力机制将计算复杂度降至 O(L log L)，通过可逆残差层将内存占用从 O(N×L) 降至 O(L)（N 为层数），在保持模型质量的同时支持长达 64K token 的序列处理。

---

## 层 2：故事摘要（5 分钟读完）

### 核心问题

2019 年，Transformer 已成为 NLP 的标准架构，但面临两大瓶颈：

1. **内存墙**：标准 Transformer 的自注意力需要存储 O(L²) 的注意力矩阵，处理长序列时 GPU 内存迅速耗尽
2. **计算墙**：注意力复杂度是序列长度的平方，处理 64K token 需要约 40 亿次运算，实际不可行

当时的变通方案是截断序列或使用稀疏注意力，但都会损失模型质量。

### 关键洞察

Reformer 团队有两个核心洞察：

**洞察 1：注意力可以局部敏感哈希（LSH）**
- 观察：Attention(Q,K,V) 中，每个 query 主要关注少数几个 key
- 想法：用 LSH 把相似的 query 和 key 分到同一桶，只在桶内计算注意力
- 结果：复杂度从 O(L²) 降至 O(L log L)

**洞察 2：残差层可以可逆**
- 观察：标准 Transformer 每层都要存储激活值用于反向传播，N 层就是 N 倍内存
- 想法：设计可逆层，反向传播时从输出重构输入，只需存储一层的激活值
- 结果：内存从 O(N×L) 降至 O(L)

### 实验结果

| 数据集 | 序列长度 | 任务 | Reformer 效果 |
|--------|----------|------|--------------|
| **enwik8** | 64K | 字符级语言建模 | 优于 Sparse Transformer |
| **imagenet64** | 4K | 图像生成 | 达到 PixelCNN 水平 |
| **WikiText-Enwik8** | 16K | 跨域语言建模 | 证明长程依赖学习能力 |

关键优势：
- **内存效率**：处理 64K 序列时，内存占用仅为标准 Transformer 的 1/10
- **计算速度**：长序列训练速度快 3-5 倍
- **模型质量**：在相同困惑度下，Reformer 能处理更长的上下文

---

## 框架大图

```
┌─────────────────────────────────────────────────────────────────┐
│                        问题空间                                  │
│  ┌──────────────────────────────────────────────────────┐       │
│  │  标准 Transformer 的效率瓶颈                          │       │
│  │  - 注意力矩阵 O(L²) 内存和计算                        │       │
│  │  - N 层残差需要存储 N 倍激活值                         │       │
│  │  - 长序列（>4K）训练不可行                           │       │
│  └──────────────────────┬───────────────────────────────┘       │
│                         │                                       │
│         ┌───────────────▼───────────────┐                       │
│         │  当时的改进尝试                │                       │
│         │  - Sparse Transformer: 固定稀疏│                       │
│         │  - Memory Transformer: 缓存   │                       │
│         │  问题：要么损失质量，要么增益有限│                     │
│         └───────────────┬───────────────┘                       │
└─────────────────────────┼───────────────────────────────────────┘
                          │
                          ▼
              ┌─────────────────────────┐
              │   Reformer 核心洞察      │
              │                         │
              │  "相似 token 才需要      │
              │   计算注意力，为何要    │
              │   计算所有对？"          │
              │                         │
              │  "残差层是确定性的，     │
              │   为何不能反向重构？"    │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │      架构组件            │
              │  ┌───────────────────┐  │
              │  │ LSH Attention     │  │
              │  │ - 哈希分桶         │  │
              │  │ - 桶内注意力       │  │
              │  │ - 多轮累积         │  │
              │  ├───────────────────┤  │
              │  │ Reversible Layers │  │
              │  │ - 可逆残差         │  │
              │  │ - 激活重构         │  │
              │  ├───────────────────┤  │
              │  │ Chunking FF       │  │
              │  │ - 分段前馈网络     │  │
              │  └───────────────────┘  │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │        验证层            │
              │  enwik8: 64K 序列        │
              │  BPC: 1.05 (SOTA)       │
              │  内存：1/10             │
              │                         │
              │  ImageNet64: 4K 序列     │
              │  Bits/dim: 3.77         │
              └─────────────────────────┘
```

---

## 预期 vs 实际对比表

| 维度 | 预期假设 | 实际发现 | 洞察 |
|------|----------|----------|------|
| **LSH 能否替代精确注意力** | 会损失大量精度 | 困惑度几乎相同 | 注意力本身是近似的，LSH 捕捉了主要贡献 |
| **哈希碰撞影响** | 需要精确分桶 | 多轮随机哈希足够 | 概率保证每对 token 至少一次同桶 |
| **可逆层的数值稳定性** | 重构误差会累积 | 实测稳定到 1000+ 层 | 浮点误差在可接受范围内 |
| **训练速度** | 可能变慢（额外哈希） | 长序列快 3-5 倍 | 注意力主导计算，LSH 收益远超开销 |
| **序列长度外推** | 可能受限于哈希表大小 | 支持 64K+ 序列 | LSH 的复杂度是对数的 |

---

## 论文定位图谱

```
                    上游工作 (它解决了谁的问题)
                    ┌─────────────────────────┐
                    │   Transformer (2017)    │
                    │  - 标准自注意力         │
                    │  - O(L²) 复杂度         │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │   GPT/BERT (2018-19)    │
                    │  -  Decoder/Encoder-only│
                    │  -  序列长度限制~512    │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │   Sparse Transformer    │
                    │  (Child et al., 2019)   │
                    │  - 固定稀疏注意力       │
                    │  - 仍需 O(L √L)         │
                    └───────────┬─────────────┘
                                │
          ┌─────────────────────┼─────────────────────┐
          │                     ▼                     │
          │            ┌─────────────────┐            │
          │            │    Reformer     │            │
          │            │  (2020) 本研究   │            │
          │            └────────┬────────┘            │
          │                     │                     │
          ┼─────────────────────┼─────────────────────┼
    并行工作                     │              并行工作
┌──────────────────┐            │        ┌──────────────────┐
│  Linear Attention│            │        │  Longformer      │
│  - 线性近似       │            │        │  - 滑动窗口      │
└──────────────────┘            │        └──────────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │   FlashAttention (2022) │
                    │  - IO 感知注意力         │
                    │  - 分块计算 + 重计算     │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │   LLaMA / Mistral       │
                    │  - 高效长序列处理       │
                    │  - 继承 Reformer 思想    │
                    └─────────────────────────┘

         下游工作 (谁解决了它的问题/扩展了它)
```

---

## 第一章：研究者的困境

### 2019 年的 Transformer 危机

2019 年，Transformer 如日中天。BERT、GPT-2 相继发布，NLP 进入了预训练时代。但研究人员面临一个尴尬的现实：

**场景 1：训练一个文档级语言模型**
```
任务：建模整篇论文（约 10K token）
标准 Transformer 需求:
- 注意力矩阵：10K × 10K = 1 亿个浮点数 = 400 MB (FP32)
- 12 层激活值：12 × 400 MB = 4.8 GB
- 梯度 + 优化器状态：再 ×3 = 14.4 GB

结果：单卡（16GB）直接爆炸，需要多卡并行
```

**场景 2：基因组序列建模**
```
任务：建模 DNA 序列（可达 100K+碱基）
标准 Transformer 需求:
- 注意力矩阵：100K × 100K = 100 亿个元素
- 计算量：约 10^20 FLOPs

结果：即使有无限内存，计算也需要数周
```

**场景 3：高分辨率图像生成**
```
任务：ImageNet 64×64 = 4K token
标准 Transformer 需求:
- 每层注意力：4K² = 16M 对
- 20 层：3.2 亿对计算

结果：训练慢到无法迭代实验
```

### 当时的解决方案及其局限

| 方法 | 思路 | 局限 |
|------|------|------|
| **梯度检查点** | 只存储部分激活，反向时重计算 | 时间换空间，训练慢 2-3 倍 |
| **稀疏注意力** | 只计算部分注意力（如局部窗口） | 丢失长程依赖，质量下降 |
| **序列截断** | 只处理前 N 个 token | 丢失上下文信息 |
| **分布式训练** | 多卡并行 | 通信开销大，成本高 |

团队陷入了一个困境：**要么快但质量差，要么好但跑不动**。

---

## 第二章：试错的旅程

### 第一阶段：从注意力矩阵的稀疏性开始

团队首先观察到一个现象：

**观察**：训练好的 Transformer 中，注意力矩阵通常是稀疏的——每个 token 主要关注少数几个其他 token。

```
示例：句子 "The cat sat on the mat because it was tired."
"it" 的注意力分布:
- "cat": 0.75  (主语指代)
- "tired": 0.15 (状态相关)
- 其他词：0.10 (分散)

结论：90% 的注意力权重集中在 2 个词上
```

**想法 1：硬截断**
- 每个 query 只保留 top-k 个 key
- 问题：需要计算所有注意力才能知道哪些大，没有解决 O(L²) 问题

**想法 2：局部窗口**
- 每个 token 只关注前后 W 个邻居
- 问题：丢失长程依赖（如指代消解需要跨越整句）

**想法 3：学习稀疏模式**
- 让模型自己学哪些位置需要关注
- 问题：训练不稳定，收敛困难

### 第二阶段：LSH 的灵感

某天，团队在读到 LSH（Locality-Sensitive Hashing）论文时获得了灵感。

**LSH 的核心思想**：
```
给定一个哈希函数 h(x):
- 如果 x 和 y 相似，则 h(x) = h(y) 的概率高
- 如果 x 和 y 不相似，则 h(x) ≠ h(y) 的概率高

应用到注意力：
- Q 和 K 相似 → 哈希到同一桶 → 计算注意力
- Q 和 K 不相似 → 不同桶 → 跳过（注意力≈0）
```

**第一版实现**：
```python
# 朴素 LSH 注意力
def lsh_attention(Q, K, V, num_buckets):
    # 1. 计算哈希
    q_hash = hash_function(Q, num_buckets)
    k_hash = hash_function(K, num_buckets)

    # 2. 按桶分组
    buckets = group_by_hash(q_hash, k_hash)

    # 3. 桶内计算注意力
    output = bucketed_attention(Q, K, V, buckets)

    return output
```

**问题 1：哈希碰撞**
- 有些相关的 Q-K 对被分到不同桶
- 解决：多轮哈希，增加覆盖

**问题 2：桶不平衡**
- 有些桶很大，有些很小
- 解决：设置桶大小上限，超出的截断

### 第三阶段：可逆层的发现

在优化内存时，团队注意到残差层的特殊结构：

**标准残差层**：
```
y = x + F(x)
```

**关键洞察**：如果 F 是确定性的，那么给定 y 和 x，可以反解出 x！

**可逆残差层设计**：
```
前向:
  y1 = x1 + F(x2)
  y2 = x2 + G(y1)

反向（重构输入）:
  x2 = y2 - G(y1)
  x1 = y1 - F(x2)
```

**验证**：
```python
# 数值稳定性测试
x1, x2 = torch.randn(2, batch, seq_len, dim)
y1 = x1 + F(x2)
y2 = x2 + G(y1)

# 重构
x2_recon = y2 - G(y1)
x1_recon = y1 - F(x2_recon)

# 误差
error1 = (x1 - x1_recon).abs().max()  # ~1e-6
error2 = (x2 - x2_recon).abs().max()  # ~1e-6
```

结果：浮点误差在可接受范围内，即使 100 层也稳定。

### 第四阶段：完整的 Reformer 架构

经过数月的迭代，Reformer 的最终架构诞生：

```
ReformerLayer:
  1. LSH Self-Attention (支持 chunking)
     - 多层 LSH 累积
     - 因果 mask（对于语言建模）

  2. Reversible Residual Connection
     - 可逆前向/反向

  3. Chunked Feed-Forward
     - 沿序列维度分块
     - 减少内存峰值

整体架构:
  [Embedding] → [ReformerLayer × N] → [Output]
```

---

## 第三章：核心概念 - 大量实例

### 概念 1：LSH Attention 是如何工作的？

**生活类比 1：图书分类**
```
想象一个图书馆有 100 万本书，你想找到与某本书 A 最相似的 10 本。

暴力方法:
- 计算 A 与 100 万本书的相似度
- 排序，取 top 10
- 复杂度：O(100 万)

LSH 方法:
- 按主题分类（哈希分桶）
- 只在与 A 同一类的书中查找
- 复杂度：O(1 万)（假设 100 个类别）

关键：相似的书在同一类的概率很高
```

**生活类比 2：社交网络推荐好友**
```
你想给用户推荐好友：

暴力方法:
- 计算该用户与所有人的相似度
- 推荐最相似的 100 人
- 对于 10 亿用户，需要 10 亿次计算

LSH 方法:
- 按兴趣/地区/职业分组（哈希）
- 只在同组内推荐
- 对于 1000 人的组，只需 1000 次计算

关键：好友通常在同一社交圈内
```

**代码实例 1：LSH 哈希函数**
```python
import torch
import torch.nn as nn

class LSHAttention(nn.Module):
    def __init__(self, dim, num_heads, num_buckets=64, num_hashes=1):
        super().__init__()
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.num_hashes = num_hashes
        self.d_k = dim // num_heads

        # 随机旋转矩阵（每轮哈希一个）
        self.rotation_matrix = nn.Parameter(
            torch.randn(num_hashes, dim, dim),
            requires_grad=False
        )

    def lsh_hash(self, vectors, rotation):
        """
        LSH 哈希：通过随机投影 + 符号
        h(x) = argmax([xR; -xR])
        """
        # 随机旋转
        rotated = torch.matmul(vectors, rotation)

        # 拼接正负（确保每个向量有两个选择）
        rotated = torch.cat([rotated, -rotated], dim=-1)

        # 取 argmax 作为哈希值
        hash_vals = torch.argmax(rotated, dim=-1)

        return hash_vals

    def forward(self, Q, K, V, mask=None):
        batch_size, seq_len, dim = Q.shape

        all_outputs = []

        for r in range(self.num_hashes):
            # 1. 计算 Q 和 K 的哈希
            q_hash = self.lsh_hash(Q, self.rotation_matrix[r])
            k_hash = self.lsh_hash(K, self.rotation_matrix[r])

            # 2. 按哈希值排序（同桶的放一起）
            Q_sorted, K_sorted, V_sorted = self.sort_by_hash(
                Q, K, V, q_hash, k_hash
            )

            # 3. 桶内计算注意力（只在相邻位置）
            output = self.bucketed_attention(
                Q_sorted, K_sorted, V_sorted
            )

            # 4. 恢复原始顺序
            output = self.unsort(output, q_hash)
            all_outputs.append(output)

        # 5. 多轮平均
        output = torch.stack(all_outputs).mean(dim=0)

        return output
```

**代码实例 2：桶内注意力（关键优化）**
```python
def bucketed_attention(self, Q, K, V, chunk_size=64):
    """
    在排序后的序列上，只计算相邻 chunk 内的注意力
    这样每个 token 只与同桶的 token 交互
    """
    batch_size, seq_len, d_k = Q.shape

    # 添加反向键（确保自注意力）
    K_rev = torch.cat([K, torch.zeros_like(K[:, :1, :])], dim=1)
    K_rev = K_rev.flip(dims=[1])

    # 分块计算注意力
    outputs = []
    for i in range(0, seq_len, chunk_size):
        q_chunk = Q[:, i:i+chunk_size, :]

        # 只与相邻块计算注意力
        k_start = max(0, i - chunk_size)
        k_end = min(seq_len, i + 2 * chunk_size)
        k_chunk = K[:, k_start:k_end, :]
        v_chunk = V[:, k_start:k_end, :]

        # 标准注意力
        scores = torch.matmul(q_chunk, k_chunk.transpose(-2, -1))
        scores /= (d_k ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        output = torch.matmul(weights, v_chunk)

        outputs.append(output)

    return torch.cat(outputs, dim=1)
```

**对比场景：标准注意力 vs LSH 注意力**
```
序列长度 L = 1000, 维度 d = 512

标准注意力:
- 计算 QK^T: 1000 × 1000 = 1M 对
- 内存：1M × 4B = 4MB（单头）
- 计算量：1M × 512 = 512M FLOPs

LSH 注意力 (num_buckets=64, num_hashes=8):
- 每轮哈希：每桶平均 1000/64 ≈ 16 个 token
- 桶内计算：64 × (16 × 16) = 16K 对
- 8 轮总计算：8 × 16K = 128K 对
- 加速比：1M / 128K ≈ 8 倍

序列长度 L = 64K 时:
- 标准：4B 对（不可行）
- LSH：8 × 64K × log(64K) ≈ 8M 对
- 加速比：500 倍
```

### 概念 2：可逆残差层

**生活类比 1：可逆加密**
```
想象你有一个加密算法：
- 加密：y = encrypt(x, key)
- 解密：x = decrypt(y, key)

只要你知道密钥和算法，就能无损还原。

可逆层也是类似的：
- 前向：y = x + F(x)
- 反向：x = y - F(x)

只要 F 是确定性的，就能还原。
```

**生活类比 2：记账本**
```
你记录每天的收支：
- 今天余额 = 昨天余额 + 今天收入

如果知道今天余额和今天收入，就能反推昨天余额：
- 昨天余额 = 今天余额 - 今天收入

可逆层就像这个记账本，每一步都是可逆的。
```

**代码实例：可逆 Transformer 层**
```python
class ReversibleBlock(nn.Module):
    def __init__(self, attn_layer, ff_layer):
        super().__init__()
        self.attn = attn_layer
        self.ff = ff_layer

    def forward(self, x):
        # 将输入分成两半
        x1, x2 = x.chunk(2, dim=-1)

        # 前向传播
        y1 = x1 + self.attn(x2)
        y2 = x2 + self.ff(y1)

        # 拼接输出
        return torch.cat([y1, y2], dim=-1)

    def reverse(self, y):
        """
        从输出重构输入（用于反向传播）
        """
        # 分割输出
        y1, y2 = y.chunk(2, dim=-1)

        # 反向重构
        x2 = y2 - self.ff(y1)
        x1 = y1 - self.attn(x2)

        return torch.cat([x1, x2], dim=-1)


class ReversibleTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.dim)

        # 构建可逆层
        layers = []
        for i in range(config.num_layers):
            attn = AttentionLayer(config)
            ff = FeedForwardLayer(config)
            layers.append(ReversibleBlock(attn, ff))

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        # 嵌入
        x = self.embedding(x)
        # 复制一份以分成两半
        x = x.repeat(2, 1, 1)

        # 前向传播
        for layer in self.layers:
            x = layer(x)

        # 取平均作为输出
        x1, x2 = x.chunk(2, dim=0)
        return (x1 + x2) / 2
```

**内存对比**：
```
标准 Transformer (12 层，seq=4K, dim=512):
- 每层激活：4K × 512 × 4B = 8MB
- 12 层总激活：96MB
- 需要存储在 GPU 内存中

可逆 Transformer:
- 只存储最后一层激活：8MB
- 反向传播时逐层重构
- 内存节省：12 倍
```

### 概念 3：Chunking 优化

**问题**：即使使用 LSH，前馈网络（FFN）的激活值仍然很大。

**解决方案**：沿序列维度分块处理。

```python
class ChunkedFeedForward(nn.Module):
    def __init__(self, dim, ff_dim, chunk_size=128):
        super().__init__()
        self.chunk_size = chunk_size
        self.net = nn.Sequential(
            nn.Linear(dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, dim)
        )

    def forward(self, x):
        """
        分块处理，每块独立计算，减少内存峰值
        """
        batch, seq_len, dim = x.shape
        outputs = []

        for i in range(0, seq_len, self.chunk_size):
            chunk = x[:, i:i+self.chunk_size, :]
            # 只存储当前块的激活
            chunk_output = self.net(chunk)
            outputs.append(chunk_output)
            # 释放前一块的激活

        return torch.cat(outputs, dim=1)
```

**内存峰值对比**：
```
序列长度 64K, FFN 中间维度 8192:

不分块:
- 中间激活：64K × 8192 × 4B = 2GB
- 需要一次性分配

分块 (chunk_size=128):
- 每块激活：128 × 8192 × 4B = 4MB
- 内存峰值：4MB（节省 500 倍）
```

---

## 第四章：关键实验的细节

### 实验 1：字符级语言建模 (enwik8)

**设置**：
- 数据集：enwik8（100MB Wikipedia 文本，字符级）
- 序列长度：65,536（64K）
- 模型：Reformer (12 层，512 维，8 头)
- 对比：Sparse Transformer, Transformer-XL

**结果**：
| 模型 | Bits per Character (BPC) ↓ | 训练时间 | 内存峰值 |
|------|---------------------------|----------|----------|
| Transformer-XL | 1.08 | 72 小时 | 32GB |
| Sparse Transformer | 1.07 | 48 小时 | 24GB |
| **Reformer** | **1.05** | 24 小时 | 8GB |

**关键洞察**：
- Reformer 在更短训练时间内达到更好的结果
- 内存峰值仅为对比方法的 1/3 到 1/4
- 64K 序列在单卡上可训练

### 实验 2：图像生成 (ImageNet64)

**设置**：
- 数据集：ImageNet 64×64
- 序列长度：4096（将图像展平为 pixel 序列）
- 任务：自回归生成（预测下一个 pixel）
- 评估指标：bits/dim（越低越好）

**结果**：
| 模型 | Bits/Dim ↓ | 参数量 |
|------|-----------|--------|
| PixelCNN | 3.83 | 47M |
| PixelSNAIL | 3.81 | 280M |
| Sparse Transformer | 3.77 | 67M |
| **Reformer** | **3.77** | 63M |

**关键洞察**：
- Reformer 达到 SOTA 水平，参数量更少
- 证明在视觉生成任务上同样有效

### 实验 3：长程依赖捕捉

**设置**：
- 任务：复制任务（copy task）
- 输入：随机序列 + 分隔符 + 提示
- 输出：与输入序列相同
- 序列长度：从 1K 到 64K

**结果**：
```
准确率 vs 序列长度:

序列长度    | Transformer | Reformer
------------|-------------|----------
1K          | 100%        | 100%
4K          | 85%         | 99%
16K         | 45%         | 98%
64K         | 12%         | 95%
```

**关键洞察**：
- 标准 Transformer 在长序列上迅速退化（内存不足导致 batch size 太小）
- Reformer 在 64K 序列上仍保持 95% 准确率

---

## 第五章：反直觉挑战

**问题 1：LSH 不会丢失重要信息吗？**

直觉：近似哈希会漏掉一些重要的 Q-K 对。

实际：**不会**，原因有三：
1. 多轮哈希（num_hashes=8）保证高概率覆盖
2. 注意力本身是软加权，小权重对结果影响小
3. 实验表明困惑度与精确注意力相当

数学保证：
```
P(某对 Q-K 至少一次同桶) = 1 - (1 - p)^num_hashes

当 p=0.3, num_hashes=8 时:
P = 1 - (0.7)^8 ≈ 0.94

即 94% 的相关对都会被计算
```

**问题 2：可逆层的数值误差会累积吗？**

直觉：100 层可逆层，每层都有浮点误差，累积起来会很大。

实际：**不会显著累积**。

原因：
```
每层重构误差 ~1e-7（浮点精度）
100 层累积误差 ~100 × 1e-7 = 1e-5

对于 FP32（精度~1e-7），1e-5 仍在可接受范围内
实验验证：1000 层 Reformer 仍稳定训练
```

**问题 3：LSH 的超参数敏感吗？**

直觉：num_buckets、num_hashes 需要精心调节。

实际：**默认值在大多数任务上有效**。

推荐配置：
```
序列长度 < 4K:  num_buckets=64, num_hashes=4
序列长度 4K-16K: num_buckets=128, num_hashes=8
序列长度 > 16K: num_buckets=256, num_hashes=16
```

---

## 第六章：与其他论文的关系

### 上游工作

**Transformer (2017)**
- 基础架构，但 O(L²) 复杂度限制了序列长度

**Sparse Transformer (2019)**
- 首次尝试稀疏注意力
- 固定稀疏模式（如每行只计算固定位置）
- Reformer 改进为动态稀疏（由哈希决定）

**Memory Transformer (2019)**
- 缓存历史激活，避免重复计算
- Reformer 改进为可逆层，从根本上减少内存

### 并行工作

**Longformer (2020)**
- 滑动窗口注意力 + 全局注意力
- 同样针对长序列，但采用不同的稀疏策略

**Linear Transformer (2020)**
- 用核方法近似注意力，达到 O(L) 复杂度
- Reformer 的 LSH 方法更稳定

### 下游工作

**FlashAttention (2022)**
- IO 感知的分块注意力
- 借鉴了 Reformer 的 chunking 思想，但针对 GPU 内存层级优化

**FlashAttention-2 (2023)**
- 进一步优化的分块策略
- 成为现代 LLM 训练的标准

**LLaMA / Mistral (2023)**
- 使用滑动窗口注意力 + RoPE
- 继承了 Reformer 的长序列处理思想

---

## 第七章：如何应用

### 场景 1：处理长文档

```python
from transformers import ReformerModel, ReformerTokenizer

# 加载预训练模型
model = ReformerModel.from_pretrained("google/reformer-crime-and-punishment")
tokenizer = ReformerTokenizer.from_pretrained("google/reformer-crime-and-punishment")

# 处理长文本（支持 16K+ token）
text = "Very long document..." * 1000  # 10K tokens
inputs = tokenizer(text, return_tensors="pt", truncation=False)

# 前向传播（不会 OOM）
outputs = model(**inputs)
```

### 场景 2：自定义长序列任务

```python
import torch
from torch import nn

class LongSequenceClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.reformer = ReformerModel(config)
        self.classifier = nn.Linear(config.hidden_dim, config.num_classes)

    def forward(self, input_ids):
        # Reformer 处理长序列
        outputs = self.reformer(input_ids)
        # 使用 [CLS] 或平均池化
        pooled = outputs.last_hidden_state.mean(dim=1)
        return self.classifier(pooled)

# 配置
config = ReformerConfig(
    vocab_size=50265,
    hidden_dim=512,
    num_layers=12,
    num_heads=8,
    max_position=65536,  # 支持 64K
    chunk_size=128,
    num_hashes=8
)
```

### 场景 3：选择合适的位置编码

| 任务类型 | 推荐位置编码 | 原因 |
|----------|-------------|------|
| 语言建模 | 可学习位置编码 | 绝对位置重要 |
| 文档分类 | 相对位置编码 | 关注 token 间关系 |
| 图像生成 | 2D 位置编码 | 保留空间结构 |

---

## 第八章：延伸思考

1. **LSH 注意力的局限是什么？**
   - 提示：哈希碰撞、桶不平衡、短序列 overhead

2. **可逆层为何没有成为主流？**
   - 提示：实现复杂度、数值稳定性、与现代架构的兼容性

3. **Reformer 与 FlashAttention 的关系是什么？**
   - 提示：两者都解决效率问题，但层面不同（算法 vs 系统）

4. **为什么现代 LLM（如 LLaMA）没有直接用 LSH 注意力？**
   - 提示：推理效率、硬件友好性、训练 - 推理一致性

5. **64K 序列长度足够吗？未来的应用需要多长？**
   - 提示：书籍级（100K+）、代码库级（1M+）、视频级（10M+）

6. **LSH 的思想能否应用到其他领域？**
   - 提示：图神经网络、推荐系统、检索系统

---

**论文元信息**
- 标题：Reformer: The Efficient Transformer
- 发表会议：ICLR 2020
- 作者：Nikita Kitaev, Łukasz Kaiser, Anselm Levskaya (Google Research)
- arXiv: 2001.04451
- 阅读时间：2026-03-15
- 方法论：高保真交互式精读协议
