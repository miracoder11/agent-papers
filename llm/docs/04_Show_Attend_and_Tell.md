# Show, Attend and Tell: Neural Image Caption Generation with Visual Attention

## 层 1：电梯演讲

**一句话概括**：Xu 等人在 2015 年提出将 Bahdanau Attention 应用到图像描述生成任务，通过 Encoder (CNN) 提取图像特征，Decoder (LSTM) 生成描述，引入软/硬两种注意力机制让模型学会"看哪里说哪里"，在三个基准数据集上达到 SOTA，开创了视觉注意力在图像描述领域的应用。

---

## 层 2：故事摘要

### 背景：2015 年的图像描述挑战

想象这个场景：你看到一张照片——"一只鸟飞过水面"。你的大脑在瞬间完成了：
1. 识别图像中的物体（鸟、水）
2. 理解它们的关系（鸟在水上方飞行）
3. 用自然语言描述（"A bird flying over water"）

这个看似简单的任务对计算机来说极其困难。

**2014-2015 年的方法困境**：

```
传统方法 A：模板填充
- 检测物体 → 鸟 ✓, 水 ✓
- 填充模板 → "A [物体] 在 [位置]"
- 结果："A bird on water" （不够丰富）

传统方法 B：检索修改
- 找相似图片 → 找到"鸟在树上"的描述
- 修改描述 → 把"树"改成"水"
- 结果："A bird near water" （仍然生硬）
```

**神经网络的突破**：

2014 年，Vinyals 等人提出 "Show and Tell"：
```
图像 → CNN 特征向量 → LSTM → 句子
```

但有一个问题：**整个图像被压缩成一个固定向量**。

```
┌────────────────────────────────────────┐
│  问题：固定向量信息瓶颈                 │
│                                        │
│  图像：复杂的场景，多个物体，关系       │
│         ↓                              │
│  CNN:  4096 维向量                       │
│         ↓                              │
│  LSTM:  试图记住所有信息...             │
│  "鸟在飞...等等，水是什么颜色来着？"    │
└────────────────────────────────────────┘
```

### 核心洞察

Xu 团队问了一个关键问题：

> "人类描述图像时，是一次性看完整个图像，还是逐部分关注？"

答案是：**逐部分关注**（attention）。

当你描述"一只鸟飞过水面"时：
- 说到"鸟"时，你关注鸟的位置
- 说到"飞"时，你关注鸟的翅膀动作
- 说到"水面"时，你关注下方的水体

**灵感来源**：

Bahdanau 等人在机器翻译中引入 Attention 机制（2014 年 9 月），让翻译模型在生成每个目标词时关注源句子的不同位置。

Xu 团队的创新：
> "如果把图像的卷积特征看作'源句子'，把图像描述看作'目标句子'，能否用同样的 attention 机制？"

### 关键突破

**两种注意力机制**：

```
┌─────────────────────────────────────────────────────────┐
│  软注意力 (Soft Attention)                               │
│  - 确定性，可微                                          │
│  - 对所有位置加权平均                                    │
│  - 可以用标准反向传播训练                                │
│  - 类似 Bahdanau Attention                               │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  硬注意力 (Hard Attention)                               │
│  - 随机性，每次只选一个位置                              │
│  - 不可微，需要强化学习 (REINFORCE)                      │
│  - 更像人类的"注视点"机制                                │
│  - 需要变分下界或策略梯度训练                            │
└─────────────────────────────────────────────────────────┘
```

### 实验结果震撼学界

| 数据集 | 方法 | BLEU-4 |
|--------|------|--------|
| Flickr8k | Google NIC (Vinyals 2014) | — |
| Flickr8k | Log Bilinear (Kiros 2014) | 17.7 |
| **Flickr8k** | **Soft Attention** | **21.3** ✓ |
| **Flickr8k** | **Hard Attention** | **20.3** |
| Flickr30k | Google NIC | 17.1 |
| **Flickr30k** | **Soft Attention** | **19.9** ✓ |
| **Flickr30k** | **Hard Attention** | **18.46** |
| MS COCO | BRNN (Karpathy & Li) | 24.6 |
| **MS COCO** | **Soft Attention** | **25.0** ✓ |
| **MS COCO** | **Hard Attention** | **23.04** |

**关键成就**：
- 在三个数据集上都达到 SOTA
- 单模型性能超越之前的集成方法
- METEOR 指标大幅提升（更流畅的描述）

### 可视化：模型真的学会了"看"

最令人兴奋的结果是**注意力可视化**：

```
图像：一只狗在床上，旁边有本书

生成的句子：
"A dog [█▓▓▓░░░░] is laying [░░█▓▓▓░░] on a bed [░░░░█▓▓▓] with a book [░░░░░░█▓]"
         ↑狗的位置      ↑动作位置      ↑床的位置        ↑书的位置
```

模型自动学会了词与图像区域的对应关系，无需任何人工标注！

---

## 层 3：深度精读

### 开场：一个失败的场景

**2014 年末，多伦多大学 Vector 研究所**

研究人员在分析 Vinyals 的 "Show and Tell" 模型时，发现了一个令人沮丧的问题：

```
输入图像：一个穿着西装戴着帽子的男人在滑板上

Show and Tell 输出：
"A man is standing on a beach with a surfboard"
```

问题很明显：
- 模型看到了"男人"
- 模型看到了类似"滑板"的物体（误认为冲浪板）
- 但描述完全错误——没有西装，没有帽子，场景也不对

**根本原因**：

当 LSTM 在第 10 个时间步生成"surfboard"时，它只能依赖：
1. 初始的图像向量（已经被"稀释"）
2. 之前生成的词的历史

它无法**回头去看**图像的特定区域来确认细节。

这正是 Bahdanau 等人在机器翻译中遇到的问题——而他们的解决方案是 Attention。

### 第一章：从翻译到描述——跨领域的洞察

#### 1.1 类比的力量

```
机器翻译：                        图像描述：
─────────────────────────────────────────────────────────
源语言句子   源句子词序列           图像       卷积特征序列
    ↓                              ↓
"Je t'aime"  ← 翻译 ←   "I love you"   "鸟飞水面"  ← 描述 ←  [特征 1, ..., 特征 196]
    ↓                              ↓
目标语言句子  目标词序列           句子       词序列
```

**关键洞察**：

Bahdanau Attention 的核心公式：
```
context_t = Σ α_ti * h_i  (h_i 是编码器隐状态)
```

在图像描述中：
```
context_t = Σ α_ti * a_i  (a_i 是卷积特征向量)
```

形式完全一样！

#### 1.2 为什么卷积特征可以作为"注释向量"？

**传统方法**（Vinyals 2014）：
```
图像 → CNN → 全连接层 → 4096 维向量
                            ↓
                      整个图像的压缩表示
```

**问题**：
- 空间信息丢失
- 无法定位具体物体
- 信息瓶颈严重

**Show, Attend and Tell 的方法**：
```
图像 (224×224) → CNN → 14×14×512 特征图
                             ↓
                      展平为 196×512
                             ↓
              每个 512 维向量对应图像的一个区域
```

**直观理解**：

```
┌────────────────────────────────────────┐
│          14×14 特征图网格                │
│                                        │
│  ┌─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬─┬┐       │
│  │1│2│3│4│5│6│7│8│9│...        │  每  │
│  ├─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼─┼┤  个  │
│  │...         196 个位置        │  格  │
│  │                              │  子 │
│  │每个位置提取 512 维特征         │  对 │
│  │                              │  应 │
│  └──────────────────────────────┘  图 │
                                       像
                                       的
                                       一
                                       个
                                       区
                                       域
```

### 第二章：模型架构详解

#### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│              Show, Attend and Tell 架构                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Encoder (CNN)               Decoder (LSTM + Attention)    │
│  ┌─────────────┐             ┌───────────────────────┐     │
│  │  输入图像    │             │  注意力机制            │     │
│  │  224×224×3  │             │  ┌─────────────────┐  │     │
│  │      ↓      │             │  │ 计算权重α       │  │     │
│  │  VGG-16    │  14×14×512  │  │ context = Σα*a  │  │     │
│  │  (预训练)  │ ──────────→ │  └────────┬────────┘  │     │
│  │      ↓      │  展平 196×512│           ↓         │     │
│  │  特征图    │  {a₁,...,a₁₉₆}│  LSTM 生成词       │     │
│  └─────────────┘             │  y_t = softmax(...) │     │
│                              └───────────────────────┘     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

#### 2.2 Encoder：卷积特征提取

**论文选择**：Oxford VGGNet (19 层)

```python
# 为什么不使用全连接层？
# 全连接层会丢失空间信息

# 卷积层的优势：
conv_layer_output: (batch, 512, 14, 14)
                      ↓
# 重排为：
annotations: (batch, 196, 512)
             L=196 个位置，每个 D=512 维
```

**代码实例**：
```python
import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    def __init__(self, embed_size=512):
        super().__init__()
        # 使用预训练的 VGG-16
        vgg = models.vgg16(pretrained=True)

        # 取第 4 个卷积块（最后一个 max pool 之前）
        # 这样保留更多空间信息
        features = list(vgg.features.children())
        self.conv = nn.Sequential(*features[:24])  # 到 conv4_3

        # 不需要梯度（预训练冻结）
        for param in self.conv.parameters():
            param.requires_grad = False

    def forward(self, images):
        # images: (batch, 3, 224, 224)
        features = self.conv(images)
        # features: (batch, 512, 14, 14)

        # 重排为 (batch, 196, 512)
        features = features.permute(0, 2, 3, 1)
        features = features.contiguous().view(features.size(0), -1, 512)

        return features  # annotations
```

#### 2.3 Decoder：LSTM + Attention

**LSTM 状态更新**（与标准 LSTM 相同）：
```
i_t = σ(W_i * [E(y_{t-1}), h_{t-1}, z_t])
f_t = σ(W_f * [E(y_{t-1}), h_{t-1}, z_t])
o_t = σ(W_o * [E(y_{t-1}), h_{t-1}, z_t])
g_t = tanh(W_g * [E(y_{t-1}), h_{t-1}, z_t])
c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
h_t = o_t ⊙ tanh(c_t)
```

**关键区别**：`z_t` 是 context vector，来自 attention！

#### 2.4 Attention 机制

**步骤 1：计算对齐分数**
```
e_{t,i} = f_att(a_i, h_{t-1})
        = v_a^T * tanh(W_a * h_{t-1} + U_a * a_i)
```

这正是 Bahdanau Attention 的形式！

**步骤 2：softmax 归一化**
```
α_{t,i} = exp(e_{t,i}) / Σ_k exp(e_{t,k})
```

**步骤 3：计算 context**
```
软注意力：z_t = Σ_i α_{t,i} * a_i
硬注意力：z_t = a_{s_t}, 其中 s_t ~ Multinoulli(α_t)
```

**代码实例**：
```python
class Attention(nn.Module):
    def __init__(self, encoder_dim=512, decoder_dim=512, attention_dim=512):
        super().__init__()
        # Bahdanau-style attention
        self.W_a = nn.Linear(decoder_dim, attention_dim)
        self.U_a = nn.Linear(encoder_dim, attention_dim)
        self.v_a = nn.Linear(attention_dim, 1)

    def forward(self, annotations, decoder_hidden, mask=None):
        # annotations: (batch, L, encoder_dim)
        # decoder_hidden: (batch, decoder_dim)

        # 计算对齐分数
        energy = torch.tanh(
            self.W_a(decoder_hidden.unsqueeze(1)) +  # (batch, 1, attention_dim)
            self.U_a(annotations)                     # (batch, L, attention_dim)
        )
        energy = self.v_a(energy).squeeze(-1)         # (batch, L)

        # mask 用于处理 padding（如果需要）
        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-inf'))

        # softmax 得到注意力权重
        attention_weights = torch.softmax(energy, dim=-1)  # (batch, L)

        # 计算 context vector
        context = torch.bmm(
            attention_weights.unsqueeze(1),  # (batch, 1, L)
            annotations                       # (batch, L, D)
        ).squeeze(1)  # (batch, D)

        return context, attention_weights
```

#### 2.5 输出层：Deep Output

**创新点**：使用 deep output layer 整合更多信息

```
p(y_t | a, y_{1:t-1}) ∝ exp(L_o * (E(y_{t-1}) + L_h * h_t + L_z * z_t))
```

**直观理解**：
- `E(y_{t-1})`：上一个词的嵌入
- `h_t`：LSTM 的当前状态（语言历史信息）
- `z_t`：context vector（当前关注的视觉信息）

三者相加后通过线性层映射到词表空间。

**代码**：
```python
class DeepOutputLayer(nn.Module):
    def __init__(self, embed_dim=512, hidden_dim=512, context_dim=512, vocab_size=10000):
        super().__init__()
        self.L_o = nn.Linear(embed_dim + hidden_dim + context_dim, vocab_size)
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim

    def forward(self, word_embed, hidden_state, context):
        # 拼接三种信息
        combined = torch.cat([word_embed, hidden_state, context], dim=-1)
        # 映射到词表空间
        logits = self.L_o(combined)
        return logits  # (batch, vocab_size)
```

#### 2.6 初始化 LSTM 状态

**创新设计**：用 annotation 的平均值初始化

```
c_0 = f_init_c( (1/L) * Σ_i a_i )
h_0 = f_init_h( (1/L) * Σ_i a_i )
```

**为什么有效**：
- 在开始生成之前，模型先"浏览"整个图像
- 得到全局信息作为初始状态
- 比全零初始化更合理

**代码**：
```python
class LSTMInit(nn.Module):
    def __init__(self, encoder_dim=512, decoder_dim=512):
        super().__init__()
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)

    def forward(self, annotations):
        # annotations: (batch, L, D)
        mean_annotation = annotations.mean(dim=1)  # (batch, D)

        c0 = torch.tanh(self.init_c(mean_annotation))
        h0 = torch.tanh(self.init_h(mean_annotation))

        return c0, h0
```

### 第三章：软注意力 vs 硬注意力

#### 3.1 软注意力（Soft Attention）

**定义**：
```
z_t = E_{s_t|a}[z_t] = Σ_i α_{t,i} * a_i
```

**特点**：
- **确定性**：给定输入，输出固定
- **可微**：梯度可以直接反向传播
- **训练简单**：标准 SGD/Adam 即可

**为什么可微**：

因为 `z_t` 是 `α` 的线性函数，而 `α` 通过 softmax 从 `e` 计算，`e` 通过 MLP 从 `h` 和 `a` 计算——整个链路都是可微的！

**训练目标**：最小化负对数似然
```
L_d = -log P(y | a) + λ * Σ_i (1 - Σ_t α_{t,i})²
              ↑
        双重随机正则化
```

**双重随机正则化（Doubly Stochastic Regularization）**：

这是一个巧妙的技巧：
- 自然有：`Σ_i α_{t,i} = 1`（softmax 保证）
- 鼓励：`Σ_t α_{t,i} ≈ 1`（每个位置都被关注过）

**直观理解**：
> "描述完整图像时，应该均匀地关注所有区域，不要遗漏。"

**效果**：
- 定量：BLEU 提升显著
- 定性：描述更丰富，不容易遗漏细节

#### 3.2 硬注意力（Hard Attention）

**定义**：
```
s_{t,i} = 1 如果选择位置 i，否则 0
z_t = a_{s_t} = Σ_i s_{t,i} * a_i

其中 s_t ~ Multinoulli(α_t)
```

**特点**：
- **随机性**：每次采样不同的位置
- **不可微**：采样操作阻断梯度
- **需要强化学习**：REINFORCE 或变分下界

**训练目标**：变分下界
```
L_s = Σ_s P(s | a) * log P(y | s, a) ≤ log P(y | a)

梯度估计：
∂L_s/∂W ≈ (1/N) * Σ_n [∂log P(y|s̃_n,a)/∂W + log P(y|s̃_n,a) * ∂log P(s̃_n|a)/∂W]
```

**方差减少技巧**：

1. **移动平均基线**：
   ```
   b_k = 0.9 * b_{k-1} + 0.1 * log P(y | s̃_k, a)
   ```

2. **熵正则化**：
   ```
   + λ_e * ∂H[s̃_n]/∂W
   ```

3. **期望值混合**：
   ```
   50% 概率用采样，50% 概率用期望值α
   ```

**最终梯度**：
```
∂L_s/∂W ≈ (1/N) * Σ_n [
    ∂log P(y|s̃_n,a)/∂W +
    λ_r * (log P(y|s̃_n,a) - b) * ∂log P(s̃_n|a)/∂W +
    λ_e * ∂H[s̃_n]/∂W
]
```

**代码实例（简化版）**：
```python
def hard_attention_loss(model, annotations, target_words):
    batch_size, L, _ = annotations.shape

    # 采样注意力位置
    attention_probs = model.attention(annotations, decoder_hidden)
    sampled_positions = torch.multinomial(attention_probs, num_samples=1)

    # 获取 context（采样的位置）
    context = torch.gather(
        annotations, 1,
        sampled_positions.unsqueeze(-1).expand(-1, -1, annotations.size(-1))
    ).squeeze(1)

    # 计算 NLL loss
    logits = model.output(word_embed, hidden, context)
    nll_loss = nn.CrossEntropyLoss()(logits, target_words)

    # 计算 REINFORCE loss
    log_probs = torch.log(attention_probs.gather(1, sampled_positions)).squeeze()
    reinforce_loss = (log_probs * (nll_loss.detach() - baseline)).mean()

    # 熵正则化
    entropy = -(attention_probs * torch.log(attention_probs + 1e-8)).sum(dim=-1).mean()

    return nll_loss + lambda_r * reinforce_loss - lambda_e * entropy
```

#### 3.3 两种注意力的对比

| 特性 | 软注意力 | 硬注意力 |
|------|----------|----------|
| 确定性 | 是 | 否 |
| 可微 | 是 | 否 |
| 训练方法 | 反向传播 | REINFORCE/变分下界 |
| 梯度方差 | 低 | 高（需要技巧减少） |
| 更像人类 | 较不像 | 更像（注视点） |
| BLEU 性能 | 略高 | 略低 |
| 可视化 | 模糊权重 | 清晰的点 |

### 第四章：实验细节

#### 4.1 训练配置

```python
# 模型配置
encoder_dim = 512      # VGG conv4_3 输出
decoder_dim = 512      # LSTM 隐层
attention_dim = 512    # 注意力 MLP 隐藏层
embed_dim = 512        # 词嵌入
vocab_size = 10000     # 词表大小

# 训练配置
batch_size = 64
learning_rate = 1e-4   # Adam
num_epochs = 20

# 正则化
dropout = 0.5
doubly_stochastic_lambda = 1.0  # 仅软注意力
entropy_lambda = 0.01           # 仅硬注意力
reinforce_lambda = 1.0          # 仅硬注意力
```

#### 4.2 按长度 Batching

**问题**：不同长度的句子需要不同时间处理

**解决方案**：
```python
def build_length_buckets(captions):
    """按句子长度分组"""
    length_dict = defaultdict(list)
    for img_id, caption in captions:
        length_dict[len(caption.split())].append((img_id, caption))
    return length_dict

def sample_batch(length_dict):
    """采样一个 batch（所有句子长度相同）"""
    lengths = list(length_dict.keys())
    chosen_length = random.choice(lengths)
    batch = random.sample(length_dict[chosen_length], batch_size)
    return batch
```

**效果**：训练速度提升约 2 倍

#### 4.3 评估指标

**BLEU**（Bilingual Evaluation Understudy）：
- BLEU-1 到 BLEU-4（1-gram 到 4-gram 精度）
- 不带 brevity penalty（与之前工作保持一致）

**METEOR**：
- 考虑同义词和词干
- 与人类判断相关性更高

### 第五章：可视化分析

#### 5.1 注意力热力图

**可视化方法**：
```
1. 获取 attention weights: (196,)
2. 重塑为 (14, 14)
3. 上采样 16 倍 → (224, 224)
4. 高斯模糊平滑
5. 叠加在原图上
```

**代码**：
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize_attention(image, attention_weights, word):
    """可视化某个词的注意力"""
    # attention_weights: (196,)
    attn_map = attention_weights.reshape(14, 14)

    # 上采样
    attn_map = cv2.resize(attn_map, (224, 224), interpolation=cv2.INTER_CUBIC)

    # 高斯模糊
    attn_map = cv2.GaussianBlur(attn_map, (23, 23), sigmaX=11, sigmaY=11)

    # 归一化
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())

    # 叠加
    overlay = image * 0.5 + (attn_map[:, :, np.newaxis] * 255 * np.array([1, 0, 0])) * 0.5

    plt.imshow(overlay.astype(np.uint8))
    plt.title(f"Attention for '{word}'")
    plt.axis('off')
    plt.show()
```

#### 5.2 注意力随时间变化

```
句子："A woman is throwing a frisbee in a park"

时间步 1 "A":      [████████░░░░░░░░]  (关注整个场景)
时间步 2 "woman":   [░░░█████░░░░░░░░]  (关注人物)
时间步 3 "throwing":[░░░░░░████░░░░░░]  (关注动作)
时间步 4 "frisbee": [░░░░░░░░░████░░░]  (关注飞盘)
时间步 5 "park":    [████░░░░░░░░░░░░]  (关注背景)
```

### 第六章：错误分析

#### 6.1 典型错误类型

**类型 1：物体识别错误**
```
图像：狗拿着书
生成："A dog is standing on a hardwood floor"
问题：把床误认为地板
```

**类型 2：关系错误**
```
图像：女人拿着甜甜圈
生成："A woman holding a clock in her hand"
问题：甜甜圈看起来像钟
```

**类型 3：细节遗漏**
```
图像：穿西装戴帽子的男人
生成："A man in a suit and a hat" ✓
但注意力没有关注帽子区域
```

#### 6.2 利用注意力诊断错误

**优势**：
- 可以看到模型"看到"了什么
- 定位错误来源（视觉特征提取？attention 分配？语言生成？）

**示例**：
```
生成："A stop sign with a mountain in the background"
注意力显示：
- "stop sign" 正确关注标志 ✓
- "mountain" 关注了背景的山 ✓
- 但实际标志上写的是"STOP"而不是山名

诊断：视觉特征正确，但语言模型过度泛化
```

### 第七章：与其他方法对比

#### 7.1 与 Show and Tell (Vinyals 2014) 对比

```
┌─────────────────────────────────────────────────────────┐
│  Show and Tell (2014)                                   │
│                                                         │
│  图像 → CNN → 全连接 → 4096 维 → LSTM → 句子              │
│                        ↑                                │
│                   信息瓶颈                              │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  Show, Attend and Tell (2015)                           │
│                                                         │
│  图像 → CNN → 14×14×512 → Attention → LSTM → 句子         │
│                        ↑                                │
│                   动态关注不同区域                        │
└─────────────────────────────────────────────────────────┘
```

#### 7.2 与 Bahdanau Attention 对比

| 方面 | Bahdanau (NMT) | Show Attend Tell (Image Caption) |
|------|----------------|----------------------------------|
| Encoder | 双向 RNN | CNN (VGG) |
| Annotations | RNN 隐状态 | 卷积特征 |
| Decoder | RNN | LSTM |
| Attention 形式 | 相同（Bahdanau-style） | 相同 |
| 训练 | 反向传播 | 软：反向传播，硬：REINFORCE |
| 可视化 | 词对齐 | 空间热力图 |

### 第八章：如何应用

#### 8.1 完整实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ShowAttendTell(nn.Module):
    def __init__(self, vocab_size=10000, embed_size=512, hidden_size=512,
                 attention_dim=512, encoder_dim=512, dropout=0.5):
        super().__init__()

        # 组件
        self.encoder = EncoderCNN(encoder_dim)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(encoder_dim, hidden_size, attention_dim)
        self.decoder_lstm = nn.LSTMCell(embed_size + encoder_dim, hidden_size)
        self.init_lstm = LSTMInit(encoder_dim, hidden_size)
        self.output = nn.Linear(hidden_size + encoder_dim + embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, images, captions, teacher_forcing_ratio=0.5):
        batch_size = images.size(0)
        max_len = captions.size(1)
        vocab_size = self.output.out_features

        # 编码
        annotations = self.encoder(images)  # (batch, L, D)

        # 初始化 LSTM
        h, c = self.init_lstm(annotations)

        # 解码
        outputs = []
        attentions = []

        for t in range(max_len):
            # 注意力
            context, alpha = self.attention(annotations, h)
            attentions.append(alpha)

            # 词嵌入
            word_embed = self.embedding(captions[:, t])
            word_embed = self.dropout(word_embed)

            # LSTM 更新
            h, c = self.decoder_lstm(
                torch.cat([word_embed, context], dim=1),
                (h, c)
            )

            # 输出
            output = self.output(
                torch.cat([word_embed, h, context], dim=1)
            )
            outputs.append(output)

        return torch.stack(outputs, dim=1), torch.stack(attentions, dim=1)

    def sample(self, images, max_len=30):
        self.eval()
        with torch.no_grad():
            annotations = self.encoder(images)
            h, c = self.init_lstm(annotations)

            inputs = torch.ones(images.size(0), dtype=torch.long) * 2  # <start>
            captions = []

            for _ in range(max_len):
                word_embed = self.embedding(inputs)
                context, _ = self.attention(annotations, h)
                h, c = self.decoder_lstm(torch.cat([word_embed, context], dim=1), (h, c))
                output = self.output(torch.cat([word_embed, h, context], dim=1))

                _, predicted = output.max(1)
                captions.append(predicted)
                inputs = predicted

                # 检查是否生成<end>
                if (predicted == 3).all():  # 假设 3 是<end>
                    break

            return torch.stack(captions, dim=1)
```

### 第九章：延伸思考

#### 苏格拉底式追问

1. **为什么使用预训练的 CNN 而不是从头训练？**
   - 2015 年数据量有限，从头训练容易过拟合
   - ImageNet 预训练提供通用视觉特征
   - 现代方法（如 Transformer）可以端到端训练

2. **软注意力和硬注意力，哪个更好？**
   - 软：训练稳定，BLEU 略高
   - 硬：更像人类，但训练困难
   - 实际：软注意力成为主流

3. **为什么选择 14×14 的特征图？**
   - 太小：空间分辨率不足
   - 太大：计算开销大，attention 分散
   - 14×14（196 位置）是经验证的平衡点

4. **这个架构可以扩展到其他任务吗？**
   - 视觉问答（VQA）：图像 + 问题 → 答案
   - 视觉对话：多轮图像相关对话
   - 视频描述：3D CNN + Temporal Attention

---

## 结语

**Show, Attend and Tell** 是视觉 - 语言多模态领域的里程碑。它证明了：

> **Attention 机制可以跨领域迁移——从翻译到描述，从文本到图像。**

这篇论文的关键贡献：

1. **首次将 Bahdanau Attention 应用到图像描述**
2. **提出软/硬两种注意力机制**
3. **实现了可解释的视觉对齐**
4. **在三个数据集上达到 SOTA**

**历史定位**：

```
时间线：
─────────────────────────────────────────────────────────→
2014.11      2015.02      2015.09    2016      2017
   │            │            │         │         │
   │            │            │         │         └── Transformer
   │            │            │         └── Show, Attend and Tell (ICML)
   │            │            └── Bahdanau Attention 普及
   │            └── Show, Attend and Tell (arXiv)
   └── Show and Tell (Vinyals)

影响：
- 开创了视觉注意力在图像描述中的应用
- 启发了后续 VQA、视觉对话等任务
- 为 Transformer 在多模态的应用铺路
```

当你看到现代多模态模型（如 CLIP、Flamingo、GPT-4V）时，它们的视觉-语言对齐能力都可以追溯到这篇论文的核心思想：

> **让模型学会"看哪里，说哪里"。**
