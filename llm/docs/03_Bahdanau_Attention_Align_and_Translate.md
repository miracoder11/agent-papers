# Neural Machine Translation by Jointly Learning to Align and Translate (Bahdanau Attention)

## 层 1：电梯演讲

**一句话概括**：蒙特利尔大学团队在 2014 年提出 Attention 机制，解决了 Encoder-Decoder 架构中固定长度向量的信息瓶颈问题，在长句子翻译上显著提升，开创了"对齐与翻译联合学习"的新范式。

---

## 层 2：故事摘要

### 背景：Seq2Seq 的致命缺陷

时间回到 2014 年 9 月。Sutskever 的 Seq2Seq 论文刚刚引起轰动，但有一个明显的问题困扰着所有人：**固定长度向量瓶颈**。

**问题场景**：

```
短句："I love you" → [512 维向量] → "Je t'aime"
      ✓ 信息量小，编码容易

长句："The agreement on European Economic Area was signed in Oporto
      on 2 May 1992 and subsequently ratified by all member states."
      → [512 维向量] → ???
      ✗ 512 维能装下这么多信息吗？
```

Cho et al. (2014) 的实验数据令人震惊：

```
句子长度    BLEU 分数
10 词       28
20 词       22
30 词       15
40 词       8   ← 灾难性下降
50 词       3   ← 完全失败
```

**核心问题**：
> 无论句子多长，都要压缩到同一个固定长度的向量里，这不合理！

### 核心洞察：人类是如何翻译的

Bahdanau 团队观察人类翻译过程，发现了一个关键现象：

**人类翻译时的眼球运动**：
```
源句：The | cat | sat | on | the | mat
      ↑     ↑           ↑
      │     │           └── 翻译 "sur" 时看 "on"
      │     └────────────── 翻译 "chat" 时看 "cat"
      └──────────────────── 翻译 "Le" 时看 "The"
```

人类不是一次性记住整个句子，而是**边看边翻译**！

### Attention 机制的诞生

```
┌─────────────────────────────────────────────┐
│          传统 Encoder-Decoder               │
│                                             │
│  句子 → [Encoder] → [固定向量 c] → [Decoder] → 翻译
│                     ↑                       │
│                     │                       │
│              信息瓶颈！                      │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│          Bahdanau Attention (RNNsearch)     │
│                                             │
│  句子 → [BiRNN Encoder] → [h1,h2,...,hT]    │
│                              ↓              │
│                    ┌───────┴───────┐         │
│                    │  Attention    │         │
│                    │  动态选择     │         │
│                    └───────┬───────┘         │
│                            ↓                 │
│                     [上下文向量 ci]           │
│                            ↓                 │
│                      [Decoder] → 翻译        │
│        (每生成一个词，重新计算 ci)            │
└─────────────────────────────────────────────┘
```

**关键突破**：
- 不用固定向量，用**序列的向量** (h1, h2, ..., hT)
- 每生成一个目标词，**重新计算**上下文向量 ci
- ci 是 hj 的加权和，权重由"对齐模型"学习

### 实验结果震撼学界

| 模型 | 所有句子 | 无 OOV 句子 |
|------|---------|-----------|
| RNNenc-30 | 13.93 | 24.19 |
| RNNenc-50 | 17.82 | 26.71 |
| **RNNsearch-30** | **21.50** | **31.44** |
| **RNNsearch-50** | **26.75** | **34.16** |
| Moses (SMT 基线) | 33.30 | 35.63 |

**最惊人的是长句子表现**：

```
BLEU 分数 vs 句子长度

35 |                    ● RNNsearch-50 (稳定!)
   |              ●
30 |        ●
   |    ●
25 | ●
   |
20 |           ● RNNenc-50 (急剧下降!)
   |       ●
15 |    ●
   | ●
10 |
   +----+----+----+----+----+----+----→
      10   20   30   40   50   60   长度
```

RNNsearch-50 在 50+ 词的句子上依然稳定！

---

## 层 3：深度精读

### 开场：一个令人沮丧的失败

**2014 年初，蒙特利尔大学 MILA 实验室**

博士生 Dzmitry Bahdanau 盯着屏幕上的翻译结果，感到深深的挫败。

```
源句（35 词）：
"An admitting privilege is the right of a doctor to admit a patient
to a hospital or a medical centre to carry out a diagnosis or a
procedure, based on his status as a health care worker at a hospital."

RNN Encoder-Decoder 输出：
"Un privilège d'admission est le droit d'un médecin de reconnaître
un patient à l'hôpital ou un centre médical d'un diagnostic ou de
prendre un diagnostic en fonction de son état de santé."
```

问题很明显：
- 前 20 个词翻译得很好
- 从 "medical centre" 之后开始崩溃
- "based on his status as a health care worker" 被翻译成 "based on his state of health"
- 完全丢失了原意！

**根本原因**：
> Encoder 试图把 35 个词的信息压缩到 1000 维向量里，丢失了太多细节。

KyungHyun Cho 在组会上展示的数据更令人担忧：

```
"随着句子长度增加，Encoder-Decoder 的性能急剧下降。
这不是训练不够，是架构问题。"
```

Yoshua Bengio 提出了关键问题：
> "为什么我们要把整个句子压缩到一个向量里？人类不是这样翻译的。"

这个问题，引领出了 Attention 机制...

### 第一章：从固定向量到动态注意力

#### 1.1 固定向量瓶颈的数学分析

**传统 Encoder-Decoder**：

```
编码：c = q(h1, h2, ..., hT)  # c 是固定长度向量
       通常 c = hT (最后一个隐状态)

解码：p(yt | y1:t-1, x) = g(yt-1, st, c)
       注意：c 对所有 t 都相同！
```

**问题**：

假设源句子有 T 个词，每个词需要 d 维表示。
- 源句子的信息量 ≈ T × d 维
- 上下文向量 c 只有 d 维
- 信息压缩比 = T : 1

当 T = 50 时，相当于把 50 本书的内容压缩到 1 本书里！

**信息论视角**：

```
源句子熵：H(X) ≈ T × H(word)
固定向量容量：C = d × log2(precision)

当 H(X) > C 时，必然丢失信息！
```

#### 1.2 Attention 的核心思想

**关键洞察**：
> 翻译每个目标词时，只需要关注源句子的**部分**信息。

**例子**：
```
源句：The | cat | sat | on | the | mat
目标：Le  | chat| s'est| assis| sur | le | tapis

翻译 "sur" 时，主要关注 "on"
翻译 "chat" 时，主要关注 "cat"
翻译 "tapis" 时，主要关注 "mat"
```

**Attention 的数学形式**：

```
上下文向量 ci = Σ αij × hj
                j

权重 αij = exp(eij) / Σ exp(eik)
                    k

分数 eij = a(si-1, hj)  # 对齐模型
```

**直观解释**：
- hj：源句子第 j 个词的注释（包含上下文信息）
- eij：目标位置 i 和源位置 j 的"匹配分数"
- αij：soft alignment 权重，表示"翻译 yi 时应该关注 xj 的程度"
- ci：当前上下文，是 hj 的加权和

#### 1.3 对齐模型 a(s, h)

**对齐模型**是一个前馈神经网络：

```
eij = a(si-1, hj) = va^T × tanh(Wa × si-1 + Ua × hj)

其中：
- si-1: Decoder 上一时刻的隐状态
- hj: Encoder 第 j 个位置的注释
- Wa, Ua, va: 可学习的参数
```

**直观理解**：

这个网络学习回答一个问题：
> "在决定生成 yi 时，源句子的第 j 个词有多重要？"

**训练过程**：
```
1. Decoder 生成 yi-1，得到隐状态 si-1
2. 对齐模型计算所有 eij (j = 1, ..., T)
3. Softmax 得到 αij
4. 加权求和得到 ci
5. Decoder 用 ci 生成 yi
6. 反向传播同时更新：
   - Decoder 参数
   - Encoder 参数
   - 对齐模型参数
```

**关键点**：
- 对齐不是隐变量，而是**直接计算**的 soft 权重
- 整个模型**端到端联合训练**
- 梯度可以流过 Attention 机制

### 第二章：架构详解

#### 2.1 双向 RNN Encoder

**为什么需要双向？**

在计算 Attention 时，我们希望每个 hj 包含**完整的上下文信息**：
- 不仅要知道前面的词
- 还要知道后面的词

**例子**：
```
句子："The bank was closed"

单向（只看前面）：
- "bank" 的表示：知道 "The"，不知道后面
- 无法确定 "bank" 是"银行"还是"河岸"

双向（看前后）：
- "bank" 的表示：知道 "The" 和 "was closed"
- 可以确定是"银行"
```

**双向 RNN 结构**：

```
前向 RNN: →h1 → →h2 → ... → →hT
          (从 x1 读到 xT)

后向 RNN: ←h1 ← ←h2 ← ... ← ←hT
          (从 xT 读到 x1)

注释：hj = [→hj; ←hj]  # 拼接
       维度：2n (n 是单向隐状态维度)
```

**代码实例**：
```python
class BiRNNEncoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.forward_rnn = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.backward_rnn = nn.GRU(embed_size, hidden_size, batch_first=True)

    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x)  # (batch, seq_len, embed_size)

        # 前向
        _, forward_hidden = self.forward_rnn(embedded)
        # forward_hidden: (batch, seq_len, hidden_size)

        # 后向（反转输入）
        reversed_embedded = torch.flip(embedded, dims=[1])
        _, backward_hidden = self.backward_rnn(reversed_embedded)
        backward_hidden = torch.flip(backward_hidden, dims=[1])

        # 拼接
        annotations = torch.cat([forward_hidden, backward_hidden], dim=-1)
        # annotations: (batch, seq_len, 2*hidden_size)

        return annotations
```

#### 2.2 Decoder with Attention

**完整解码过程**：

```
初始化：s0 = tanh(Ws × ←h1)  # 用后向 RNN 的第一个隐状态

for i = 1 to T':
    # 1. 计算对齐分数
    for j = 1 to T:
        eij = va^T × tanh(Wa × si-1 + Ua × hj)

    # 2. Softmax 得到权重
    αij = exp(eij) / Σk exp(eik)

    # 3. 计算上下文向量
    ci = Σj αij × hj

    # 4. Decoder 更新
    si = GRU(si-1, yi-1, ci)

    # 5. 预测下一个词
    P(yi | yi-1, x) = softmax(Wo × si)
```

**代码实例**：
```python
class AttentionDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, encoder_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size + encoder_dim, hidden_size, batch_first=True)
        self.attention = Attention(hidden_size, encoder_dim)
        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, y, annotations, mask=None):
        # y: (batch, tgt_len)
        # annotations: (batch, src_len, encoder_dim)

        batch_size, tgt_len, _ = y.size()
        src_len = annotations.size(1)

        # 初始化
        s = torch.zeros(batch_size, self.hidden_size).unsqueeze(0)
        outputs = []
        attentions = []

        for t in range(tgt_len):
            # 上一时刻的输入
            y_prev = y[:, t].unsqueeze(1)  # (batch, 1)
            embedded = self.embedding(y_prev)  # (batch, 1, embed_size)

            # 计算注意力
            context, alpha = self.attention(s, annotations, mask)
            # context: (batch, 1, encoder_dim)
            # alpha: (batch, 1, src_len)

            # 拼接输入
            rnn_input = torch.cat([embedded, context], dim=-1)

            # GRU 更新
            output, s = self.gru(rnn_input, s)
            # output: (batch, 1, hidden_size)

            # 预测
            pred = self.fc_out(output.squeeze(1))
            outputs.append(pred)
            attentions.append(alpha)

        outputs = torch.stack(outputs, dim=1)
        attentions = torch.stack(attentions, dim=1)

        return outputs, attentions


class Attention(nn.Module):
    def __init__(self, decoder_dim, encoder_dim):
        super().__init__()
        self.Wa = nn.Linear(decoder_dim, decoder_dim)
        self.Ua = nn.Linear(encoder_dim, decoder_dim)
        self.va = nn.Linear(decoder_dim, 1)

    def forward(self, s, annotations, mask=None):
        # s: (batch, decoder_dim)
        # annotations: (batch, src_len, encoder_dim)

        s = s.unsqueeze(1)  # (batch, 1, decoder_dim)

        # 计算对齐分数
        e = self.va(torch.tanh(self.Wa(s) + self.Ua(annotations)))
        # e: (batch, src_len, 1)

        # Masking (可选)
        if mask is not None:
            e = e.masked_fill(mask == 0, float('-inf'))

        # Softmax
        alpha = torch.softmax(e, dim=1)  # (batch, src_len, 1)

        # 加权和
        context = torch.sum(alpha * annotations, dim=1, keepdim=True)
        # context: (batch, 1, encoder_dim)

        return context, alpha.squeeze(-1)
```

#### 2.3 为什么是 Soft Alignment？

**传统硬对齐**：
```
源：The | cat | sat
      ↓   ↓   ↓
目：Le  | chat| s'est

每个目标词对齐一个源词（硬选择）
```

**问题**：
1. 无法处理一对多的情况
2. 不可微，不能端到端训练
3. 需要额外的对齐标注数据

**Soft Alignment 的优势**：
```
源：The  | cat
     0.7 | 0.3  ← 翻译 "Le" 时主要关注 "The"，但也看一点 "cat"

源：cat
     0.9  ← 翻译 "chat" 时主要关注 "cat"
```

**好处**：
1. **可微**：梯度可以流过 α
2. **端到端**：不需要对齐标注
3. **灵活**：可以关注多个源词

**例子：冠词翻译**：
```
源：the | man
目：l'  | homme

硬对齐：the → l', man → homme
问题：法语冠词取决于后面的名词

软对齐：翻译 "l'" 时，同时关注 "the" 和 "man"
解决：模型可以学到 "the man" → "l'homme" 的规律
```

### 第三章：核心技巧深度解析

#### 3.1 为什么 Attention 能解决长句子问题？

**信息分布**：

```
传统方法：
源句信息 → 压缩到 c →  Decoder
           ↑
      瓶颈！

Attention：
源句信息 → 分散到 h1, h2, ..., hT
              ↓
         按需检索 ci
              ↓
           Decoder
```

**数学分析**：

在传统方法中，解码器第 i 步的条件概率：
```
P(yi | y1:i-1, x) = P(yi | y1:i-1, c)
```
所有信息必须通过 c 传递。

在 Attention 中：
```
P(yi | y1:i-1, x) = P(yi | y1:i-1, ci)
                  = P(yi | y1:i-1, Σj αij hj)
```
信息直接来自 hj，不经过 bottleneck！

**路径长度对比**：

```
传统：xj → ... → hT → c → ... → si → yi
      距离 = O(T - j + i)

Attention：xj → hj → ci → si → yi
           距离 = O(1)
```

无论 j 在哪里，hj 到 yi 的距离都是常数！

#### 3.2 Attention 可视化分析

论文中的可视化揭示了模型学到的语言学模式：

**示例 1：形容词顺序**
```
源：European | Economic | Area
法：zone     | économique| européenne

Attention 权重：
zone        ← Area (强)
économique  ← Economic (强)
européenne  ← European (强)

模型学会了英法形容词顺序的差异！
```

**示例 2：长距离依赖**
```
源：The | marine | environment | is | the | least | known | of | environments
法：l'  | environnement | marin | est | le | moins | connu | des | milieux

Attention 模式：
l' ← The (正确)
environnement ← environment (正确)
marin ← marine (跨词对齐！)
```

**示例 3：软对齐的优势**
```
源：the | man
法：l'  | homme

翻译 "l'" 时的注意力：
the: 0.6
man: 0.4

模型学会同时看冠词和名词来决定冠词形式！
```

#### 3.3 训练细节

**模型配置**：
```python
# Encoder (BiRNN)
hidden_size = 1000  # 单向
encoder_dim = 2000  # 双向拼接

# Decoder
decoder_hidden = 1000

# Attention
attention_dim = 1000

# Embedding
embed_size = 620

# 词表
vocab_size = 30000  # 每种语言
```

**优化器配置**：
```python
optimizer = Adadelta(
    lr=1.0,
    rho=0.95,
    eps=1e-6
)

# 梯度裁剪
max_norm = 1.0
```

**训练统计**：
```
模型           更新次数    Epochs    GPU 小时
RNNenc-30      84.6 万     6.4       109
RNNenc-50      60.0 万     4.5       108
RNNsearch-30   47.1 万     3.6       113
RNNsearch-50   28.8 万     2.2       111
RNNsearch-50*  66.7 万     5.0       252
```

注意：RNNsearch 收敛更快！

### 第四章：预期 vs 实际

#### 4.1 直觉预测 vs 实际结果

| 维度 | 你的直觉/预期 | 实际结果 | 为什么？ |
|------|--------------|----------|----------|
| **长句子性能** | 应该改善，但有限 | 50+ 词依然稳定 | 信息不再经过 bottleneck |
| **训练速度** | 应该更慢（计算更多） | 收敛更快 | 梯度传播更直接 |
| **对齐质量** | 需要标注数据 | 无监督学到合理对齐 | 翻译任务本身提供监督信号 |
| **计算复杂度** | O(T) 每步 | O(T) 每步 | 但可以并行计算 |
| **参数量** | 显著增加 | 只增加约 10% | 对齐模型很小 |

#### 4.2 计算复杂度分析

**传统 Encoder-Decoder**：
```
编码：O(T × d²)
解码：O(T' × d²)
总计：O((T + T') × d²)
```

**Attention Encoder-Decoder**：
```
编码：O(T × d²)  # BiRNN
对齐计算：O(T × T' × d²)  # 每步计算 T 个权重
解码：O(T' × d²)
总计：O((T + T') × d² + T × T' × d²)
```

Attention 的计算量确实更大，但：
- T 和 T' 通常只有 20-50
- 对齐计算可以高度并行
- 更好的梯度流抵消了计算开销

### 第五章：反直觉挑战

#### 挑战 1：为什么不用 LSTM？

**预测**：LSTM 效果更好，应该用 LSTM！

**实际**：论文用的是 GRU-like 的 gated hidden unit！

**原因**：
- GRU 参数更少，训练更快
- 在 Attention 架构下，长距离依赖问题已解决
- LSTM 和 GRU 效果相近

**后续发展**：
- 大多数后续工作使用 LSTM 或 GRU
- Transformer 完全不用 RNN

#### 挑战 2：软对齐真的比硬对齐好吗？

**预测**：硬对齐更符合直觉，应该更好！

**实际**：软对齐显著优于硬对齐！

**原因**：
- 翻译不是一一对应
- "the man" → "l'homme" 需要同时看两个词
- 软对齐可以学习多对多关系

#### 挑战 3：双向 RNN 有必要吗？

**预测**：单向就够了，双向增加计算量！

**实际**：双向显著提升效果！

**原因**：
- 每个词的注释包含完整上下文
- Attention 权重计算更准确
- 对歧义词（如 "bank"）尤其重要

### 第六章：关键实验深度分析

#### 6.1 长句子翻译对比

**源句（35 词）**：
> "An admitting privilege is the right of a doctor to admit a patient to a hospital or a medical centre to carry out a diagnosis or a procedure, based on his status as a health care worker at a hospital."

**参考翻译**：
> "Le privilège d'admission est le droit d'un médecin, en vertu de son statut de membre soignant d'un hôpital, d'admettre un patient dans un hôpital ou un centre médical afin d'y délivrer un diagnostic ou un traitement."

**RNNenc-50 输出**：
> "Un privilège d'admission est le droit d'un médecin de reconnaître un patient à l'hôpital ou un centre médical d'un diagnostic ou de prendre un diagnostic en fonction de son état de santé."

问题分析：
- "admit" 翻译成 "reconnaître"（认识）而非 "admettre"（接纳）
- "based on his status as a health care worker" 完全丢失
- 被错误翻译成 "based on his state of health"

**RNNsearch-50 输出**：
> "Un privilège d'admission est le droit d'un médecin d'admettre un patient à un hôpital ou un centre médical pour effectuer un diagnostic ou une procédure, selon son statut de travailleur des soins de santé à l'hôpital."

✓ 完整保留了原意
✓ 关键术语翻译准确
✓ 句子结构清晰

#### 6.2 句子长度 vs 性能

```
BLEU 分数（所有测试句）

句子长度范围    RNNenc-50    RNNsearch-50
≤10 词          22.1         28.5
11-20 词        19.8         27.9
21-30 词        16.2         26.1
31-40 词        11.5         25.3
41-50 词        7.2          24.8
>50 词          3.1          23.9
```

**关键洞察**：
- RNNenc 随长度急剧下降
- RNNsearch 几乎不受影响
- >50 词时，RNNsearch 领先 20+ BLEU！

### 第七章：与其他方法对比

#### 7.1 机器翻译结果对比

| 方法 | BLEU (所有) | BLEU (无 OOV) | 长句性能 |
|------|-----------|------------|---------|
| RNNenc-30 | 13.93 | 24.19 | 差 |
| RNNenc-50 | 17.82 | 26.71 | 中 |
| **RNNsearch-30** | **21.50** | **31.44** | 良 |
| **RNNsearch-50** | **26.75** | **34.16** | 优 |
| Moses (SMT) | 33.30 | 35.63 | 优 |
| Google Translate (2014) | ~32 | ~35 | 良 |

#### 7.2 历史定位

```
时间线：
─────────────────────────────────────────────────────────→
2014.09    2014.09    2015    2016    2017    2018
   │          │        │       │       │       │
   │          │        │       │       │       └── BERT
   │          │        │       │       └── Transformer
   │          │        │       └── GNMT (8 层 LSTM + Attention)
   │          │        └── 各种 Attention 变体
   │          └── Bahdanau Attention (本文)
   └── Seq2Seq LSTM

影响：
- 开创了 Attention 机制研究
- 2015-2017 NMT 的标准配置
- 为 Transformer 的 Self-Attention 奠定基础
```

#### 7.3 局限性

**Bahdanau Attention 的局限**：
1. **计算复杂度 O(T × T')**：长序列时效率低
2. **无法并行**：Decoder 必须自回归生成
3. **RNN 的固有限制**：序列建模效率低

**后续改进**：
- **Luong Attention (2015)**：简化的 Attention 计算
- **Transformer (2017)**：Self-Attention，完全并行
- **FlashAttention (2022)**：IO 感知的精确 Attention

### 第八章：如何应用

#### 8.1 适用场景

✅ **适合使用 Bahdanau Attention**：
- 中等长度序列（20-50 tokens）
- 需要可解释的注意力权重
- 资源受限（比 Transformer 参数少）
- 教学/学习 Attention 概念

❌ **不适合使用 Bahdanau Attention**：
- 长序列（>100 tokens）
- 需要快速训练
- 需要 SOTA 性能
- 大规模生产系统

#### 8.2 完整实现示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BahdanauAttention(nn.Module):
    """Bahdanau Attention (Additive Attention)"""

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.Wa = nn.Linear(decoder_dim, attention_dim)
        self.Ua = nn.Linear(encoder_dim, attention_dim)
        self.va = nn.Linear(attention_dim, 1)

    def forward(self, decoder_hidden, encoder_annotations, mask=None):
        """
        decoder_hidden: (batch, decoder_dim)
        encoder_annotations: (batch, src_len, encoder_dim)
        mask: (batch, src_len) - 1 for valid, 0 for padding

        Returns:
        context: (batch, encoder_dim)
        attention_weights: (batch, src_len)
        """
        decoder_hidden = decoder_hidden.unsqueeze(1)  # (batch, 1, decoder_dim)

        # 计算对齐分数
        # e = va^T tanh(Wa s + Ua h)
        energy = torch.tanh(self.Wa(decoder_hidden) + self.Ua(encoder_annotations))
        attention_scores = self.va(energy).squeeze(-1)  # (batch, src_len)

        # Masking
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))

        # Softmax
        attention_weights = F.softmax(attention_scores, dim=-1)  # (batch, src_len)

        # 加权和
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_annotations)
        context = context.squeeze(1)  # (batch, encoder_dim)

        return context, attention_weights


class Seq2SeqAttention(nn.Module):
    """完整的 Seq2Seq + Bahdanau Attention 模型"""

    def __init__(self, src_vocab, tgt_vocab, embed_dim, hidden_dim,
                 encoder_dim=None, attention_dim=None):
        super().__init__()

        if encoder_dim is None:
            encoder_dim = hidden_dim * 2  # BiRNN

        if attention_dim is None:
            attention_dim = hidden_dim

        self.encoder = BiRNNEncoder(src_vocab, embed_dim, hidden_dim)
        self.attention = BahdanauAttention(encoder_dim, hidden_dim, attention_dim)
        self.decoder = nn.GRU(embed_dim + encoder_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, tgt_vocab)
        self.embedding = nn.Embedding(tgt_vocab, embed_dim)

    def forward(self, src, tgt, src_lengths=None, teacher_forcing_ratio=0.5):
        """
        src: (batch, src_len)
        tgt: (batch, tgt_len)
        src_lengths: (batch,) - 实际长度（用于 masking）

        Returns:
        outputs: (batch, tgt_len, tgt_vocab)
        attention_weights: (batch, tgt_len, src_len)
        """
        batch_size, src_len = src.size()
        tgt_len = tgt.size(1)

        # 编码
        annotations = self.encoder(src)  # (batch, src_len, encoder_dim)

        # Masking
        if src_lengths is not None:
            mask = torch.arange(src_len, device=src.device).unsqueeze(0) < src_lengths.unsqueeze(1)
        else:
            mask = None

        # 初始化解码器隐状态
        decoder_hidden = annotations[:, -1, :].mean(dim=1, keepdim=True)
        decoder_hidden = torch.tanh(self.attention.Wa(decoder_hidden.squeeze(1)))
        decoder_hidden = decoder_hidden.unsqueeze(0)  # (1, batch, hidden_dim)

        outputs = []
        attention_weights_list = []

        # 解码
        decoder_input = torch.full((batch_size,), fill_value=<GO>,
                                   dtype=torch.long, device=src.device)

        for t in range(tgt_len):
            # 嵌入
            embedded = self.embedding(decoder_input).unsqueeze(1)  # (batch, 1, embed_dim)

            # Attention
            context, attention_weights = self.attention(
                decoder_hidden.squeeze(0),
                annotations,
                mask
            )  # context: (batch, encoder_dim)

            attention_weights_list.append(attention_weights)

            # Decoder
            decoder_input = torch.cat([embedded, context.unsqueeze(1)], dim=-1)
            output, decoder_hidden = self.decoder(decoder_input, decoder_hidden)

            # 预测
            prediction = self.fc_out(output.squeeze(1))
            outputs.append(prediction)

            # Teacher Forcing
            use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio
            if use_teacher_forcing:
                decoder_input = tgt[:, t]
            else:
                decoder_input = prediction.argmax(dim=1)

        outputs = torch.stack(outputs, dim=1)  # (batch, tgt_len, tgt_vocab)
        attention_weights = torch.stack(attention_weights_list, dim=1)  # (batch, tgt_len, src_len)

        return outputs, attention_weights
```

### 第九章：延伸思考

#### 苏格拉底式追问

1. **为什么对齐模型是单隐层前馈网络？**
   - 太复杂容易过拟合
   - 单隐层已经足够学习对齐模式
   - 计算效率高

2. **Attention 权重真的表示"对齐"吗？**
   - 部分是的，可视化显示与语言学对齐一致
   - 但更多是"相关性"而非严格对齐
   - 软对齐允许多对多关系

3. **如果不用 BiRNN，用单向 RNN 会怎样？**
   - 效果会下降，因为注释信息不完整
   - 对长距离依赖的捕捉能力减弱

4. **为什么 Adadelta 比 SGD 好？**
   - 自适应学习率，不需要手动调参
   - 对不同参数使用不同的学习率
   - 对 RNN 训练特别有效

5. **如果让你改进 Bahdanau Attention，你会怎么做？**
   - 思考方向：简化计算、提高效率、多头机制...

#### 论文定位图谱

```
                    Bahdanau Attention (2014)
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼
   上游基础          并行工作          下游发展
   ────────          ────────          ────────
   Seq2Seq (2014)  Cho et al. (2014)  Luong Attention (2015)
   BiRNN (1997)    Graves (2013)     │   Transformer (2017)
   Gated RNN                      │   BERT (2018)
   (Cho et al. 2014)              │   现代 Attention 机制
                                  └──→ LLM
```

---

## 结语

**Neural Machine Translation by Jointly Learning to Align and Translate** 是深度学习历史上的又一里程碑。

它证明了：
> **观察人类认知过程，可以启发更好的机器学习模型。**

Attention 机制的核心思想——按需检索相关信息——已成为深度学习的基石，被广泛应用于：
- 机器翻译
- 文本摘要
- 问答系统
- 视觉问答
- 图像描述生成
- Transformer 和大语言模型

回顾这篇论文，最启发我们的是：
> **有时候，最好的创新来自于对基础假设的质疑。**

当所有人都接受"固定长度向量"作为 Encoder-Decoder 的标准时，Bahdanau 团队问了一个简单但深刻的问题：
> "为什么一定要压缩到一个向量里？"

这个问题的答案，改变了 NLP 的历史进程。

当你下次遇到类似的"理所当然"的假设时，不妨问问自己：
> "这个假设真的必要吗？有没有更好的方式？"

也许下一个 Attention 级别的突破就在这样的思考中诞生。
