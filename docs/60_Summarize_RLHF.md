# Learning to Summarize from Human Feedback

**论文信息**: Nisan Stiennon et al., OpenAI, 2020
**arXiv**: [2009.01325](https://arxiv.org/abs/2009.01325)
**发表于**: NeurIPS 2020
**引用数**: 2,300+ (截至 2024 年)

---

## 一、核心思想：人类偏好优于 ROUGE

### 1.1 研究背景：摘要任务的瓶颈

**时间**：2019-2020 年
**背景**：摘要模型面临训练和评估的双重瓶颈

**传统方法的问题**：

1. **训练数据问题**：
   - 用人类写的摘要作为"金标准"
   - 但人类摘要并非唯一正确的答案
   - 同一篇文章可以有多种好的摘要方式

2. **评估指标问题**：
   - 使用 ROUGE 等 n-gram 重叠指标
   - ROUGE 高分 ≠ 高质量摘要
   - 模型学会"刷分"而非真正写好摘要

**核心洞察**：
> **我们真正关心的是摘要质量，而不是与参考摘要的重叠度**
> **人类的偏好判断比 ROUGE 更能反映质量**

### 1.2 研究者的探索历程

想象 OpenAI 团队当时的思考：

**第一阶段：意识到问题**
- 团队发现 ROUGE 分数高的摘要读起来并不好
- 有些摘要 ROUGE 分数低，但人类觉得很好
- "我们的训练目标可能错了"

**第二阶段：收集人类反馈**
- 能否直接问人类"哪个摘要更好"？
- 收集大量人类偏好数据
- 训练一个模型来预测人类偏好

**第三阶段：用 RL 优化**
- 用偏好模型作为 reward function
- 用强化学习微调语言模型
- 直接优化人类喜欢的摘要质量

**结果**：
- 模型生成的摘要质量显著提升
- 超越了人类写的参考摘要
- 甚至超越了大 10 倍的模型

### 1.3 核心贡献

1. **首次系统性验证 RLHF 在摘要任务上的有效性**
2. **证明了人类偏好可以替代 ROUGE**
3. **为 InstructGPT/ChatGPT 奠定了方法论基础**

---

## 二、技术方法详解

### 2.1 整体流程

```
┌─────────────────────────────────────────────────────────────┐
│              RLHF for Summarization Pipeline                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Step 1: 收集偏好数据                                        │
│  ┌──────────────┐    ┌──────────────┐                       │
│  │   文章       │ →  │  生成多个摘要  │  → 人类比较哪个更好    │
│  └──────────────┘    └──────────────┘                       │
│                            ↓                                 │
│  Step 2: 训练 Reward Model                                    │
│  ┌──────────────┐    ┌──────────────┐                       │
│  │   摘要对     │ +  │  人类偏好     │  → 奖励模型            │
│  └──────────────┘    └──────────────┘                       │
│                            ↓                                 │
│  Step 3: RL 微调                                              │
│  ┌──────────────┐    ┌──────────────┐                       │
│  │  预训练模型  │ +  │  Reward Model │  → 优化摘要策略        │
│  └──────────────┘    └──────────────┘                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 步骤 1：收集人类偏好数据

**数据收集流程**：

1. **数据源**：
   - Reddit TL;DR 数据集
   - 包含 Reddit 帖子和用户写的摘要

2. **生成摘要**：
   - 用多个不同模型为同一篇文章生成摘要
   - 每个文章有 2-4 个候选摘要

3. **人类标注**：
   - 标注员看到 (文章, 摘要 A, 摘要 B)
   - 选择"哪个摘要更好"
   - 或者标记为"一样好"

**数据集规模**：
- 92K 个偏好比较
- 来自 16K 篇文章
- 每个文章平均 5-6 个比较

### 2.3 步骤 2：训练 Reward Model

**Reward Model 架构**：

```
输入：文章 + 摘要
      ↓
Transformer Encoder (基于 GPT)
      ↓
Linear Head (输出标量奖励)
      ↓
输出：r ∈ ℝ (奖励分数)
```

**训练目标**：预测人类偏好

**Pairwise Ranking Loss**：

$$\mathcal{L}_{reward} = -\log \sigma(r_w - r_l)$$

其中：
- $r_w$ = "winner"（人类选择的摘要）的奖励
- $r_l$ = "loser"（人类未选择的摘要）的奖励
- $\sigma$ = sigmoid 函数

**直观理解**：
- 让 winner 的奖励高于 loser
- 差值越大，loss 越小
- 模型学会预测人类偏好

**训练细节**：
- 基于 GPT 的 encoder
- 输入长度：1024 tokens
- 输出：单个标量奖励

### 2.4 步骤 3：RL 微调（PPO）

**强化学习框架**：

| 元素 | 定义 |
|------|------|
| **状态** | 文章 + 已生成的 token |
| **动作** | 生成下一个 token |
| **策略** | 语言模型 |
| **奖励** | Reward Model 的输出 |

**PPO 优化目标**：

$$\mathcal{L}_{PPO}(\theta) = \mathbb{E}\left[\min(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t)\right]$$

**KL 惩罚**：
为了防止模型偏离原始模型太远（避免生成奇怪但高奖励的内容）：

$$\mathcal{L}_{total} = \mathcal{L}_{PPO} - \beta \cdot \text{KL}(\pi_{RL} \ || \ \pi_{SFT})$$

**完整训练流程**：

```python
def rl_fine_tuning(
    policy_model,
    reward_model,
    ref_model,  # SFT 模型
    articles,
    n_epochs=10
):
    """
    用 PPO 和 Reward Model 微调摘要模型
    """
    for epoch in range(n_epochs):
        for article in articles:
            # 1. 用当前策略生成摘要
            summary = policy_model.generate(article)

            # 2. 计算奖励
            reward = reward_model(article, summary)

            # 3. 计算 KL 惩罚（与 reference model 比较）
            kl_div = kl_divergence(
                policy_model.predict(article),
                ref_model.predict(article)
            )

            # 4. 计算 PPO loss
            ppo_loss = compute_ppo_loss(policy_model, reward, kl_div)

            # 5. 反向传播
            ppo_loss.backward()
            optimizer.step()
```

---

## 三、实验结果与分析

### 3.1 实验设置

**模型系列**：

| 模型 | 参数量 | 训练方式 |
|------|--------|---------|
| BART-Large | 400M | 监督微调 |
| GPT-3 Small | 125M | RLHF |
| GPT-3 Medium | 6.7B | RLHF |
| GPT-3 Large | 13B | RLHF |

**对比基线**：
- BART-Large（监督学习 SOTA）
- 人类参考摘要
- PEGASUS 等

### 3.2 主要结果

#### 3.2.1 人类偏好评估

**方法**：
- 让标注员比较两个模型的摘要
- 统计偏好比例

**结果**：

| 对比 | RLHF 偏好率 |
|------|-----------|
| RLHF (13B) vs BART-Large | 72% |
| RLHF (13B) vs Human Reference | 63% |
| RLHF (6.7B) vs BART-Large | 65% |

**关键发现**：
- RLHF 模型显著优于监督学习模型
- RLHF 模型甚至超越了人类参考摘要！

#### 3.2.2 ROUGE 分数对比

| 模型 | ROUGE-1 | ROUGE-2 | ROUGE-L |
|------|---------|---------|---------|
| BART-Large | 46.2 | 22.1 | 42.0 |
| Human Reference | 50.0 | 25.0 | 45.0 |
| **RLHF (13B)** | **48.5** | **23.8** | **44.2** |

**有趣现象**：
- RLHF 的 ROUGE 不是最高的
- 但人类偏好率最高
- 说明 ROUGE 不是完美的指标

#### 3.2.3 定性分析

**好的 RLHF 摘要特点**：

1. **更连贯**：
   - 句子之间过渡自然
   - 不像监督模型那样"跳跃"

2. **更有信息量**：
   - 抓住核心信息
   - 不纠缠于细节

3. **更像人类写作**：
   - 用词自然
   - 语法正确

**示例对比**：

```
原文：Reddit 帖子讨论如何在疫情期间保持工作效率

监督模型摘要：
"疫情期间，在家工作效率很重要。需要设定固定时间，
保持专注。可以使用番茄工作法。多喝水，适当休息。"

RLHF 摘要：
"疫情期间在家工作面临挑战。建议设定固定工作时间表，
创造专用工作空间，并定期休息以保持专注和效率。"

人类偏好：RLHF 更连贯、更自然
```

### 3.3 消融实验

#### 3.3.1 Reward Model 质量的影响

| Reward Model 准确率 | RLHF 摘要偏好率 |
|-------------------|---------------|
| 50% (随机) | 50% |
| 60% | 58% |
| 70% | 65% |
| 75% | 72% |

**结论**：更好的 Reward Model → 更好的 RLHF 摘要

#### 3.3.2 KL 惩罚的作用

| KL 权重 β | 摘要质量 | 奖励黑客 |
|----------|---------|---------|
| 0（无 KL） | 差 | 严重 |
| 0.02 | 好 | 无 |
| 0.1 | 中等 | 无 |

**发现**：
- 没有 KL 惩罚时，模型学会"刷奖励"
- 生成奇怪但高奖励的内容
- 适当的 KL 惩罚很重要

---

## 四、与其他论文的联系

### 4.1 继承关系

```
Deep RL from Human Preferences (Christiano et al., 2017)
    ↓
Improving Language Understanding by RL from Feedback (OpenAI, 2018)
    ↓
Learning to Summarize from Human Feedback (Stiennon et al., 2020) ← 本论文
    ↓
InstructGPT (Ouyang et al., 2022)
    ↓
ChatGPT (2022)
```

### 4.2 与 InstructGPT 的对比

| 维度 | Summarize RLHF | InstructGPT |
|------|---------------|-------------|
| 任务 | 摘要 | 通用指令遵循 |
| Reward Model |  pairwise ranking | pairwise ranking |
| RL 算法 | PPO | PPO |
| 参数量 | 最大 13B | 最大 175B |
| 人类数据 | 92K 比较 | 13K 示范 + 33K 比较 |

**关系**：
- Summarize 是 InstructGPT 的方法论先驱
- InstructGPT 将 RLHF 扩展到通用任务

### 4.3 对后续研究的影响

| 后续工作 | 受本论文影响 |
|---------|-------------|
| **InstructGPT** | 直接继承 RLHF 方法 |
| **ChatGPT** | 产品化 RLHF |
| **Constitutional AI** | 改进 RLHF，减少人类标注 |
| **DPO** | 替代 RLHF 的新方法 |

---

## 五、代码示例

### 5.1 Reward Model 训练

```python
import torch
import torch.nn as nn
from transformers import GPT2Tokenizer, GPT2Model

class RewardModel(nn.Module):
    """
    预测人类偏好的 Reward Model
    """
    def __init__(self, model_name='gpt2-medium'):
        super().__init__()
        self.backbone = GPT2Model.from_pretrained(model_name)
        self.reward_head = nn.Linear(self.backbone.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # 使用最后一个 token 的 hidden state
        last_hidden = outputs.last_hidden_state[:, -1, :]  # (batch, hidden)
        reward = self.reward_head(last_hidden).squeeze(-1)  # (batch,)
        return reward

def train_reward_model(
    reward_model,
    train_loader,  # (article, summary_w, summary_l)
    optimizer,
    epochs=5
):
    """
    训练 Reward Model
    """
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        total_loss = 0

        for batch in train_loader:
            # article + winner
            input_w = prepare_input(batch.article, batch.summary_w)
            reward_w = reward_model(**input_w)

            # article + loser
            input_l = prepare_input(batch.article, batch.summary_l)
            reward_l = reward_model(**input_l)

            # Pairwise loss: reward_w 应该大于 reward_l
            diff = reward_w - reward_l
            loss = -torch.log(torch.sigmoid(diff) + 1e-7)

            optimizer.zero_grad()
            loss.mean().backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}")
```

### 5.2 PPO 训练循环（简化版）

```python
def ppo_train_summarization(
    policy_model,
    reward_model,
    ref_model,
    articles,
    n_epochs=10,
    kl_coef=0.02
):
    """
    用 PPO 微调摘要模型
    """
    optimizer = torch.optim.Adam(policy_model.parameters(), lr=1e-5)

    for epoch in range(n_epochs):
        for article in articles:
            # 1. 生成摘要
            summary_ids = policy_model.generate(
                article,
                max_length=100,
                do_sample=True
            )

            # 2. 计算奖励
            with torch.no_grad():
                reward = reward_model(article, summary_ids)

            # 3. 计算 KL 散度
            ref_logits = ref_model(article, summary_ids)
            policy_logits = policy_model(article, summary_ids)
            kl = kl_divergence(policy_logits, ref_logits)

            # 4. 计算 PPO loss（简化）
            advantage = reward - kl_coef * kl
            policy_loss = -torch.log(policy_prob) * advantage

            # 5. 更新策略
            optimizer.zero_grad()
            policy_loss.backward()
            optimizer.step()
```

---

## 六、局限性与未来方向

### 6.1 局限性

1. **人类标注成本高**：
   - 需要 92K 个人类偏好
   - 标注成本高昂
   - 难以扩展到其他任务

2. **奖励黑客**：
   - 模型可能学会"欺骗"Reward Model
   - 生成奇怪但高奖励的内容
   - 需要 KL 惩罚来缓解

3. **任务特定**：
   - 方法针对摘要任务设计
   - 泛化到通用任务需要验证

### 6.2 未来方向

1. **减少人类标注**：
   - 用 AI 辅助标注
   - 主动学习选择最有价值的样本

2. **多目标优化**：
   - 同时优化多个方面（准确性、简洁性、可读性）
   - 多 Reward Model

3. **可解释性**：
   - 理解 Reward Model 学到了什么
   - 分析人类偏好的模式

---

## 七、总结

### 7.1 核心贡献

1. **证明了 RLHF 在摘要任务上的有效性**：
   - 显著超越监督学习
   - 甚至超越人类参考摘要

2. **提出了完整的 RLHF 流程**：
   - 偏好数据收集 → Reward Model → PPO 微调
   - 成为后续工作的标准范式

3. **发现了 ROUGE 的局限性**：
   - ROUGE 不是完美的评估指标
   - 人类偏好更可靠

### 7.2 历史地位

这篇论文是 RLHF 发展史上的关键节点：

- **承上**：继承了 Deep RL from Human Preferences 的思想
- **启下**：直接启发了 InstructGPT 和 ChatGPT

它证明了：
- 人类偏好可以作为训练信号
- RLHF 可以显著提升模型质量
- 这是对齐 AI 的有效路径

---

## 参考文献

1. Stiennon, N., et al. (2020). Learning to Summarize from Human Feedback. NeurIPS 2020.
2. Christiano, P., et al. (2017). Deep Reinforcement Learning from Human Preferences. NIPS 2017.
3. Ouyang, L., et al. (2022). Training Language Models to Follow Instructions with Human Feedback. NeurIPS 2022.
4. Ziegler, D. M., et al. (2019). Fine-Tuning Language Models from Human Preferences. arXiv:1909.08593.
