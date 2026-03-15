# Improving Language Understanding by Generative Pre-Training (GPT-1)

## 层 1：电梯演讲

**一句话概括**：OpenAI 团队在 2018 年提出使用 Transformer Decoder 进行无监督预训练 + 有监督微调的两阶段方法，在 12 个 NLP 任务中的 9 个达到 SOTA，开创了"预训练 + 微调"范式，是 GPT 系列的开山之作。

---

## 层 2：故事摘要

### 背景：2018 年的 NLP 困境

时间回到 2018 年，NLP 领域面临一个根本性问题：

**标注数据稀缺**：
```
任务                训练数据量
─────────────────────────────────
情感分类 (SST-2)    67,349 样本
文本蕴含 (MNLI)     393,000 样本
问答 (SQuAD)        100,000+ 样本
─────────────────────────────────
但很多任务只有几千样本！

语言学家接受度 (CoLA)  8,551 样本
修辞问答 (RTE)         2,490 样本
```

**人类的优势**：
```
人类学习 NLP 任务：
1. 从小到大阅读海量文本（无监督）
2. 学会语言的基本规律
3. 面对新任务时，只需少量示例就能理解

机器学习的困境：
1. 只能从标注数据学习
2. 每个任务需要重新训练
3. 小数据集容易过拟合
```

**当时的解决方案**：

**方案 1：词嵌入**（Word2Vec、GloVe）
```
优点：
- 从大规模无监督文本学习
- 捕获词汇语义

局限：
- 只学习词级别表示
- 无法捕获句子/篇章语义
- 上下文无关（"bank"只有一种表示）
```

**方案 2：任务特定架构**
```
每个任务设计专门的神经网络：
- 文本蕴含：注意力 + BiLSTM
- 问答：记忆网络
- 情感分类：CNN/递归网络

问题：
- 无法共享知识
- 每个任务需要大量标注数据
- 无法利用无监督数据
```

### 核心洞察

OpenAI 团队提出了一个简单而强大的想法：

**两阶段训练范式**：
```
┌─────────────────────────────────────────────────────────┐
│  阶段 1：无监督预训练                                    │
│  ─────────────────────────────────────────────────────   │
│  数据：BooksCorpus (7000 本书，约 1B 词)                   │
│  目标：语言建模（预测下一个词）                          │
│  模型：12 层 Transformer Decoder                          │
│  结果：学会语言理解和世界知识                            │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│  阶段 2：有监督微调                                      │
│  ─────────────────────────────────────────────────────   │
│  数据：任务特定标注数据                                 │
│  目标：任务特定损失（分类、问答等）                      │
│  模型：预训练模型 + 线性分类层                          │
│  结果：在目标任务上达到 SOTA                             │
└─────────────────────────────────────────────────────────┘
```

**关键创新**：

1. **使用 Transformer 而不是 LSTM**
   - 更长的依赖建模能力
   - 更强的表达力

2. **任务特定的输入转换**
   - 将结构化输入转换为连续 token 序列
   - 最小化模型架构修改

3. **辅助语言建模目标**
   - 微调时保留 LM 损失作为正则化
   - 提升泛化能力

### 实验结果

**在 12 个任务中的 9 个达到 SOTA**：

| 任务 | 数据集 | GPT-1 | 之前 SOTA | 提升 |
|------|--------|-------|-----------|------|
| 文本蕴含 | MNLI | 82.1 | 80.6 | **+1.5** ✓ |
| 文本蕴含 | SNLI | 89.9 | 89.3 | **+0.6** ✓ |
| 文本蕴含 | SciTail | 88.3 | 83.3 | **+5.0** ✓ |
| 问答 | RACE | 62.9 | 60.2 | **+2.7** ✓ |
| 常识推理 | Story Cloze | 86.5 | 77.6 | **+8.9** ✓ |
| 语义相似度 | STS-B | 82.0 | 81.0 | **+1.0** ✓ |
| 语义相似度 | QQP | 70.3 | 66.1 | **+4.2** ✓ |
| 语法接受度 | CoLA | 45.4 | 35.0 | **+10.4** ✓ |
| 情感分类 | SST-2 | 91.3 | 91.6 | -0.3 |
| 复写检测 | MRPC | 82.3 | 83.5 | -1.2 |
| 文本蕴含 | RTE | 56.0 | 61.7 | -5.7 |
| GLUE 综合 | - | 72.8 | 68.9 | **+3.9** ✓ |

**关键成就**：
- 9/12 任务超越 SOTA
- 超越很多专门设计的任务特定架构
- 小数据集（如 RTE）仍有差距

### 消融实验

**预训练的重要性**：
```
有预训练：平均分数 74.7
无预训练：平均分数 59.9
差距：-14.8 分！

结论：预训练带来巨大的性能提升
```

**Transformer vs LSTM**：
```
Transformer：平均分数 74.7
LSTM：平均分数 69.1
差距：-5.6 分

结论：Transformer 架构对迁移学习至关重要
```

**辅助 LM 目标**：
```
有辅助 LM：平均分数 74.7
无辅助 LM：平均分数 75.0
影响：很小，某些任务有提升

结论：辅助目标对大数据集有帮助
```

---

## 层 3：深度精读

### 开场：一个失败的尝试

**2017 年，OpenAI 办公室**

研究人员正在尝试用 LSTM 做半监督学习：

```
任务：情感分类（SST-2）

方法 1：Word2Vec + LSTM
- 用 Word2Vec 初始化词向量
- 在 SST-2 上微调 LSTM
- 结果：87% 准确率

方法 2：语言模型预训练 + LSTM
- 在大规模文本上预训练 LSTM 语言模型
- 在 SST-2 上微调
- 结果：88% 准确率

提升：只有 1%！

问题：LSTM 无法捕获长距离依赖
预训练学到的知识有限
```

**关键洞察**：
> "也许问题不在于预训练本身，而在于模型架构？"

2017 年 6 月，Transformer 论文发表。OpenAI 团队看到了机会：

```
Transformer 的优势：
- 自注意力捕获长距离依赖
- 更强的表达能力
- 在机器翻译上超越 RNN

假设：
如果用 Transformer 做语言模型预训练，
然后微调到下游任务，会怎样？
```

### 第一章：模型架构

#### 1.1 整体设计

GPT-1 使用**Decoder-only Transformer**架构：

```
┌─────────────────────────────────────────────────────────┐
│              GPT-1 架构                                 │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  输入 Token                                             │
│      │                                                  │
│      ▼                                                  │
│  ┌─────────────┐                                        │
│  │ Token 嵌入   │ (768 维)                               │
│  │ + 位置嵌入   │ (768 维，可学习)                       │
│  └─────────────┘                                        │
│      │                                                  │
│      ▼                                                  │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Transformer Decoder Layer × 12                 │   │
│  │  ┌─────────────────────────────────────────┐    │   │
│  │  │ Masked Multi-Head Self-Attention       │    │   │
│  │  │ - 12 个头，每头 64 维                      │    │   │
│  │  │ - 因果掩码（只能看到过去）               │    │   │
│  │  └─────────────────────────────────────────┘    │   │
│  │  ┌─────────────────────────────────────────┐    │   │
│  │  │ LayerNorm                               │    │   │
│  │  └─────────────────────────────────────────┘    │   │
│  │  ┌─────────────────────────────────────────┐    │   │
│  │  │ Position-wise Feed-Forward              │    │   │
│  │  │ - 3072 维中间层                           │    │   │
│  │  │ - GELU 激活                              │    │   │
│  │  └─────────────────────────────────────────┘    │   │
│  │  ┌─────────────────────────────────────────┐    │   │
│  │  │ LayerNorm                               │    │   │
│  │  └─────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────┘   │
│      │                                                  │
│      ▼                                                  │
│  输出 logits (用于预测下一个词或分类)                   │
│                                                         │
└─────────────────────────────────────────────────────────┘

参数量：约 117M
```

**为什么选择 Decoder-only**？

```
Encoder-Decoder（Transformer 原始）：
- 适合序列到序列任务（翻译）
- 对于理解任务过于复杂

Encoder-only（BERT，2018 年末）：
- 双向注意力
- 需要特殊训练目标（Masked LM）

Decoder-only（GPT-1）：
- 因果注意力（从左到右）
- 天然适合语言建模
- 可以无缝迁移到下游任务
```

#### 1.2 代码实现

```python
import torch
import torch.nn as nn
import math

class GPT1Config:
    def __init__(self):
        self.vocab_size = 40478  # BPE 词表
        self.hidden_dim = 768
        self.num_layers = 12
        self.num_heads = 12
        self.ff_dim = 3072  # 4×hidden
        self.max_position = 512
        self.dropout = 0.1

class GPT1Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # 嵌入层
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)
        self.position_embedding = nn.Embedding(config.max_position, config.hidden_dim)

        # Transformer 层
        self.layers = nn.ModuleList([
            GPT1Block(config) for _ in range(config.num_layers)
        ])

        # 层归一化
        self.ln_f = nn.LayerNorm(config.hidden_dim)

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, attention_mask=None):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # 位置编码（可学习）
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        positions = self.position_embedding(positions)

        # Token 嵌入
        x = self.token_embedding(input_ids) + positions

        # 因果掩码
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=device), diagonal=1
        ).bool()

        # Transformer 层
        for layer in self.layers:
            x = layer(x, causal_mask)

        # 最终层归一化
        x = self.ln_f(x)

        return x  # (batch, seq_len, hidden_dim)

class GPT1Block(nn.Module):
    """单个 Transformer 层"""
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.hidden_dim)
        self.attn = nn.MultiheadAttention(
            config.hidden_dim, config.num_heads,
            dropout=config.dropout, batch_first=True
        )
        self.ln_2 = nn.LayerNorm(config.hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_dim, config.ff_dim),
            nn.GELU(),
            nn.Linear(config.ff_dim, config.hidden_dim),
            nn.Dropout(config.dropout)
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, mask):
        # 自注意力 + 残差
        attn_output, _ = self.attn(
            self.ln_1(x), self.ln_1(x), self.ln_1(x),
            attn_mask=mask
        )
        x = x + self.dropout(attn_output)

        # MLP + 残差
        mlp_output = self.mlp(self.ln_2(x))
        x = x + mlp_output

        return x
```

### 第二章：两阶段训练

#### 2.1 无监督预训练

**目标函数**：标准语言建模
```
L1(U) = Σ_i log P(u_i | u_{i-k}, ..., u_{i-1}; Θ)

其中：
- U = {u_1, ..., u_n} 是无标注语料
- k = 512（上下文窗口大小）
- Θ 是模型参数
```

**直观理解**：
```
输入："The cat sat on the ___"
目标：预测下一个词"mat"

模型学习：
- 语法结构
- 语义连贯性
- 世界知识（猫坐在什么上面）
- 长距离依赖（如果前面提到过"mat"）
```

**训练数据：BooksCorpus**
```
- 7000 本未出版书籍
- 约 1B 词
- 多种类型：冒险、奇幻、浪漫等
- 关键：包含长篇幅连续文本

为什么 BooksCorpus 重要？
- 相比 1B Word Benchmark（句子级别打乱）
- BooksCorpus 保留篇章结构
- 模型学会长距离依赖
```

**训练配置**：
```python
# 预训练配置
config = {
    'optimizer': 'Adam',
    'learning_rate': 2.5e-4,
    'betas': (0.9, 0.95),
    'weight_decay': 0.01,
    'batch_size': 64,
    'sequence_length': 512,
    'epochs': 100,
    'warmup_steps': 2000,
    'lr_schedule': 'linear_warmup + cosine_decay'
}

# 学习率调度
# 前 2000 步：从 0 线性增加到 2.5e-4
# 之后：余弦退火到 0
```

**代码实现**：
```python
def pretrain(model, dataloader, config):
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        betas=config['betas'],
        weight_decay=config['weight_decay']
    )

    # 学习率调度
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['warmup_steps'],
        num_training_steps=len(dataloader) * config['epochs']
    )

    model.train()
    for epoch in range(config['epochs']):
        for batch in dataloader:
            input_ids = batch  # (batch, seq_len)
            labels = input_ids[:, 1:]  # 目标：下一个词

            # 前向传播
            hidden = model(input_ids[:, :-1])
            logits = hidden @ model.token_embedding.weight.T
            # logits: (batch, seq_len-1, vocab_size)

            # 计算损失
            loss = nn.CrossEntropyLoss()(
                logits.view(-1, logits.size(-1)),
                labels.reshape(-1)
            )

            # 反向传播
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
```

#### 2.2 有监督微调

**目标函数**：
```
L2(C) = Σ_{(x,y)} log P(y | x_1, ..., x_m)

其中：
- C 是标注数据集
- x = {x_1, ..., x_m} 是输入序列
- y 是标签
```

**辅助语言建模目标**：
```
L3(C) = L2(C) + λ × L1(C)

其中 λ = 0.5

作用：
- 正则化，防止过拟合
- 保持语言建模能力
- 加速收敛
```

**模型修改**：
```
预训练模型：
输入 → Transformer → 输出分布 (vocab_size)

微调模型：
输入 → Transformer → 线性层 → 输出分布 (num_labels)
                              ↑
                        随机初始化
```

**代码实现**：
```python
class GPT1ForSequenceClassification(nn.Module):
    def __init__(self, config, num_labels):
        super().__init__()
        self.transformer = GPT1Model(config)
        self.classifier = nn.Linear(config.hidden_dim, num_labels)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids, labels=None):
        # Transformer 编码
        hidden = self.transformer(input_ids)

        # 使用最后一个 token 的表示
        last_hidden = hidden[:, -1, :]  # (batch, hidden_dim)
        last_hidden = self.dropout(last_hidden)

        # 分类
        logits = self.classifier(last_hidden)

        # 计算损失（如果提供 labels）
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

            # 辅助语言建模损失（可选）
            # lm_logits = hidden @ self.transformer.token_embedding.weight.T
            # lm_loss = compute_lm_loss(lm_logits, input_ids)
            # loss = loss + 0.5 * lm_loss

        return logits, loss
```

### 第三章：任务特定的输入转换

GPT-1 的关键创新之一是将不同任务的输入转换为统一的 token 序列。

#### 3.1 文本分类

**最简单**：直接使用
```
输入："This movie is great!"
模型：[CLS] This movie is great! [SEP]
输出：使用最后一个 token 的表示
```

#### 3.2 文本蕴含

**任务**：判断前提（premise）是否蕴含假设（hypothesis）

```
输入转换：
Premise: The cat is sleeping.
Hypothesis: The animal is resting.

模型输入：
[START] The cat is sleeping. [DELIMITER] The animal is resting. [END]

处理：
- 用 $ 作为分隔符
- 添加 [START] 和 [END] token
```

**代码**：
```python
def prepare_entailment_input(premise, hypothesis):
    tokens = []
    tokens.append(START_TOKEN)
    tokens.extend(tokenize(premise))
    tokens.append(DELIMITER_TOKEN)  # $
    tokens.extend(tokenize(hypothesis))
    tokens.append(END_TOKEN)
    return tokens
```

#### 3.3 语义相似度

**任务**：判断两个句子是否语义等价

**关键洞察**：两个句子没有内在顺序
```
句子 A: The cat is on the mat.
句子 B: A cat sits on a mat.

P(A, B) 应该等于 P(B, A)
```

**解决方案**：处理两个顺序
```
输入 1: [START] A [DELIMITER] B [END]
输入 2: [START] B [DELIMITER] A [END]

输出：两个表示相加
h_final = h(A,B) + h(B,A)
```

**代码**：
```python
def prepare_similarity_input(sentence1, sentence2):
    input1 = [START] + tokenize(sentence1) + [DELIMITER] + tokenize(sentence2) + [END]
    input2 = [START] + tokenize(sentence2) + [DELIMITER] + tokenize(sentence1) + [END]
    return input1, input2

# 模型处理
hidden1 = model(input1)
hidden2 = model(input2)
final_hidden = hidden1[:, -1, :] + hidden2[:, -1, :]
logits = classifier(final_hidden)
```

#### 3.4 问答与常识推理

**任务**：给定文档、问题和答案选项，选择正确答案

```
输入：
Document: John went to the park. He played basketball.
Question: What did John do at the park?
Options:
A) He played soccer.
B) He played basketball.
C) He went home.

输入转换：
对于每个选项 ak:
[START] Document [DELIMITER] Question [DELIMITER] ak [END]

处理：
- 对每个选项独立处理
- softmax 归一化得到答案分布
```

**代码**：
```python
def prepare_qa_input(document, question, answers):
    inputs = []
    for answer in answers:
        tokens = [START]
        tokens.extend(tokenize(document))
        tokens.append(DELIMITER)
        tokens.extend(tokenize(question))
        tokens.append(DELIMITER)
        tokens.extend(tokenize(answer))
        tokens.append(END)
        inputs.append(tokens)
    return inputs

# 模型处理
logits_list = []
for input_ids in inputs:
    hidden = model(input_ids)
    # 使用序列表示计算分数
    score = classifier(hidden[:, -1, :])
    logits_list.append(score)

# softmax 归一化
probabilities = softmax(torch.stack(logits_list))
answer = argmax(probabilities)
```

### 第四章：实验分析

#### 4.1 迁移层数的影响

**问题**：预训练的多少层应该迁移到下游任务？

```
实验：在 MNLI 和 RACE 上测试

迁移层数    MNLI 准确率    RACE 准确率
─────────────────────────────────────
0 (仅嵌入)    73.5          54.2
3 层         76.8          56.1
6 层         79.2          58.3
9 层         81.0          60.5
12 层 (全部)  82.1          62.9

结论：每一层都包含有用的信息
      全部迁移效果最好
```

#### 4.2 Zero-Shot 行为

**问题**：预训练模型本身学到了任务能力吗？

**实验方法**：设计启发式方法，无需微调直接测试

```
CoLA（语法接受度）：
- 计算句子的平均 token 对数概率
- 阈值判断是否可接受

SST-2（情感分类）：
- 在句子后追加"very"
- 看模型预测"positive"还是"negative"概率更高

RACE（问答）：
- 计算每个答案的对数概率
- 选择概率最高的答案

结果：
- Zero-shot 性能稳定且随训练提升
- 证明预训练确实学到了任务相关知识
- Transformer 比 LSTM 的 zero-shot 更稳定
```

#### 4.3 消融实验

**表 5 深度解读**：

| 模型变体 | 平均分数 | CoLA | SST-2 | MRPC | STS-B | QQP | MNLI | QNLI | RTE |
|----------|----------|------|-------|------|-------|-----|------|------|-----|
| **完整模型** | **74.7** | 45.4 | 91.3 | 82.3 | 82.0 | 70.3 | 81.8 | 88.1 | 56.0 |
| 无辅助 LM | 75.0 | 47.9 | 92.0 | 84.9 | 83.2 | 69.8 | 81.1 | 86.9 | 54.4 |
| 无预训练 | 59.9 | 18.9 | 84.0 | 79.4 | 30.9 | 65.5 | 75.7 | 71.2 | 53.8 |
| LSTM 替代 | 69.1 | 30.3 | 90.5 | 83.2 | 71.8 | 68.1 | 73.7 | 81.1 | 54.6 |

**关键发现**：

1. **无预训练**：下降 14.8 分
   - 证明预训练的巨大价值
   - 在所有任务上都有负面影响

2. **无辅助 LM**：上升 0.3 分
   - 辅助目标不是必需的
   - 对大数据集有帮助，小数据集可能干扰

3. **LSTM 替代**：下降 5.6 分
   - Transformer 架构的优势
   - 特别是在需要长距离依赖的任务上

### 第五章：与同时期工作对比

#### 5.1 与 ELMo 对比

**ELMo**（Peters et al., 2018）：
```
架构：2 层双向 LSTM
训练：语言建模（前向 + 后向）
使用：词级别表示，作为特征输入下游模型

优点：
- 上下文相关的词表示
- 在多个任务上提升

局限：
- LSTM 表达能力有限
- 只用作特征提取
- 需要任务特定的模型设计
```

**GPT-1**：
```
架构：12 层 Transformer Decoder
训练：语言建模（单向）
使用：端到端微调

优点：
- Transformer 更强表达力
- 端到端学习
- 最小化任务特定修改

结果：
- GPT-1 在大多数任务上超越 ELMo
- 特别是需要长距离依赖的任务
```

#### 5.2 与 BERT 对比（2018 年末）

**BERT 的优势**：
```
- 双向注意力
- Masked LM 目标
- 在 NLU 任务上表现更好

局限：
- 不能直接用于生成任务
- 预训练更复杂
```

**GPT-1 的优势**：
```
- 单向，适合生成
- 训练更简单
- 为后续 GPT 系列奠定基础
```

### 第六章：历史定位

```
时间线：
─────────────────────────────────────────────────────────→
2017.06   2018.06   2018.10   2019.02   2020.06   2026
   │         │         │         │         │         │
   │         │         │         │         │         └── 现在
   │         │         │         │         └── GPT-3
   │         │         │         └── GPT-2
   │         │         └── BERT
   │         │
   │         └── GPT-1 (本文)
   │
   └── Transformer

影响：
- 开创了"预训练 + 微调"范式
- 证明 Transformer 在语言理解上的优势
- 为 GPT-2/3 和现代 LLM 奠定基础
```

### 第七章：完整代码示例

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

class GPT1FullModel(nn.Module):
    """完整的 GPT-1 模型，支持预训练和微调"""

    def __init__(self, vocab_size=40478, hidden_dim=768, num_layers=12,
                 num_heads=12, max_position=512, dropout=0.1):
        super().__init__()

        self.config = {
            'vocab_size': vocab_size,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'num_heads': num_heads,
            'max_position': max_position,
            'dropout': dropout
        }

        # 嵌入层
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_position, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Transformer 层
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim*4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True  # Pre-LN
            )
            for _ in range(num_layers)
        ])

        # 层归一化
        self.ln_f = nn.LayerNorm(hidden_dim)

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, attention_mask=None, causal_mask=True):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # 位置编码
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        positions = self.position_embedding(positions)

        # Token 嵌入 + 位置编码
        x = self.token_embedding(input_ids) + positions
        x = self.dropout(x)

        # 因果掩码
        if causal_mask:
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=device), diagonal=1
            ).bool()
            mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        else:
            mask = None

        # Transformer 层
        for layer in self.layers:
            x = layer(x, mask)

        # 最终层归一化
        x = self.ln_f(x)

        return x

    def pretrain_step(self, input_ids, labels=None):
        """预训练步骤：语言建模"""
        hidden = self.forward(input_ids)

        # 语言模型头（权重共享）
        logits = hidden @ self.token_embedding.weight.T

        # 计算损失
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = nn.CrossEntropyLoss()(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            return loss

        return logits

    def finetune_classify(self, input_ids, labels=None):
        """微调步骤：序列分类"""
        hidden = self.forward(input_ids)

        # 使用最后一个 token 的表示
        last_hidden = hidden[:, -1, :]

        # 分类头（需要在外部添加）
        return last_hidden


class SequenceClassificationHead(nn.Module):
    """用于微调的分类头"""
    def __init__(self, hidden_dim=768, num_labels=2, dropout=0.1):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels)
        )

    def forward(self, hidden, labels=None):
        logits = self.classifier(hidden)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return logits, loss


# 使用示例
def train_pipeline():
    # 1. 预训练
    model = GPT1FullModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=2.5e-4)

    # 预训练循环
    for batch in pretrain_dataloader:
        input_ids = batch
        loss = model.pretrain_step(input_ids, input_ids)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # 2. 微调
    classifier = SequenceClassificationHead(num_labels=2)
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(classifier.parameters()),
        lr=6.25e-5
    )

    for batch in finetune_dataloader:
        input_ids, labels = batch
        hidden = model.finetune_classify(input_ids)
        logits, loss = classifier(hidden, labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```

---

## 结语

**Improving Language Understanding by Generative Pre-Training** 是 NLP 历史上的里程碑。它开创了：

> **"预训练 + 微调"范式**

GPT-1 的核心贡献：

1. **首次使用 Transformer Decoder 进行预训练**
   - 证明了 Transformer 在语言理解上的优势
   - 为后续 GPT 系列奠定基础

2. **两阶段训练方法**
   - 无监督预训练学习通用表示
   - 有监督微调适应特定任务

3. **任务无关的输入转换**
   - 将不同任务统一为 token 序列
   - 最小化模型架构修改

4. **在 9/12 任务上达到 SOTA**
   - 超越很多任务特定架构
   - 证明了通用模型的可能性

**历史意义**：

GPT-1 虽然没有像后来的 GPT-3 那样引起轰动，但它确立的方向——大规模预训练、Decoder-only 架构、端到端微调——成为了现代 LLM 的标准。

当你今天使用任何大语言模型时，它们的核心训练范式都可以追溯到这篇论文：

> **先通过语言建模学习通用能力，再通过微调或提示适应具体任务**。

这，就是 GPT-1 的遗产。
