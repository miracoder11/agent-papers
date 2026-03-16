# Training Language Models to Follow Instructions with Human Feedback (InstructGPT)

**论文信息**: Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C. L., Mishkin, P., ... & Lowe, R. (2022). Training Language Models to Follow Instructions with Human Feedback. arXiv:2203.02155.

**机构**: OpenAI

**arXiv**: [2203.02155](https://arxiv.org/abs/2203.02155)

---

## 层 1：电梯演讲（30 秒）

**一句话概括**：OpenAI 团队提出 InstructGPT，通过人类反馈强化学习（RLHF）微调 GPT-3，在只需 1.3B 参数的情况下就超越了 175B 的 GPT-3，显著提升了真实性、减少了毒性输出，证明了"对齐税"（alignment tax）的存在但可通过混合预训练目标来最小化。

---

## 层 2：故事摘要（5 分钟读完）

### 核心问题

2022 年初，大型语言模型面临一个根本性问题：**模型变大并不 inherently 意味着更好地遵循用户意图。**

GPT-3 虽然强大，但存在三个关键问题：
1. **编造事实**（hallucination）：生成不真实的信息
2. **有毒/偏见输出**：生成有害或带有偏见的文本
3. **不遵循指令**：不按照用户的实际意图行事

**根本原因**：语言建模的目标（预测下一个 token）与"有帮助且安全地遵循用户指令"的目标不一致（misaligned）。

### 核心洞察

OpenAI 团队的解决方案基于一个简单但强大的想法：**使用人类反馈作为奖励信号来微调模型。**

**关键创新**：
- **三步训练流程**：监督微调（SFT）→ 奖励模型训练（RM）→ 强化学习优化（PPO）
- **40 名标注员**：基于筛选测试表现雇佣的专业团队
- **API 真实数据**：使用 OpenAI API 用户提交的实际 prompt，而非仅实验室数据

### 研究框架图

```
┌─────────────────────────────────────────────────────────────────┐
│                    问题空间                                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  GPT-3 的对齐问题                                        │   │
│  │  - 编造事实 (hallucination)                             │   │
│  │  - 有毒/偏见输出                                         │   │
│  │  - 不遵循用户指令                                        │   │
│  │  - 目标函数不一致 (next token ≠ follow intent)          │   │
│  └──────────────────────┬──────────────────────────────────┘   │
│                         │                                       │
│                         ▼                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  核心研究问题                                            │   │
│  │  "如何通过人类反馈让语言模型更好地对齐用户意图？"        │   │
│  └──────────────────────┬──────────────────────────────────┘   │
└─────────────────────────┼───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    解决方案：RLHF 三步流程                       │
│                                                                 │
│  Step 1: 监督微调 (SFT)                                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  - 收集人类标注员撰写的示范数据 (13k prompts)            │   │
│  │  - 在 OpenAI API 真实 prompt 上微调 GPT-3                │   │
│  │  - 输出：SFT 模型                                         │   │
│  └────────────────────┬────────────────────────────────────┘   │
│                       │                                         │
│                       ▼                                         │
│  Step 2: 奖励模型训练 (RM)                                      │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  - 收集模型输出的对比排名数据 (33k prompts)              │   │
│  │  - 训练奖励模型预测人类偏好                              │   │
│  │  - 输出：Reward Model (RM)                               │   │
│  └────────────────────┬────────────────────────────────────┘   │
│                       │                                         │
│                       ▼                                         │
│  Step 3: PPO 强化学习优化                                       │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  - 使用 RM 输出作为标量奖励                               │   │
│  │  - 用 PPO 算法优化 SFT 模型                               │   │
│  │  - 可混合预训练数据 (PPO-ptx)减少性能回归                │   │
│  │  - 输出：InstructGPT 模型                                │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    实验结果                                      │
│                                                                 │
│  ✓ 人类偏好：1.3B InstructGPT > 175B GPT-3                     │
│  ✓ 真实性：TruthfulQA 提升 2 倍                                 │
│  ✓ 毒性：减少 25% 有毒输出                                       │
│  ✓ 对齐税：某些 NLP 任务性能回归 (SQuAD, DROP 等)                │
│  ✓ PPO-ptx：混合预训练数据可最小化性能回归                      │
└─────────────────────────────────────────────────────────────────┘
```

### 关键结果

| 指标 | GPT-3 175B | InstructGPT 1.3B | InstructGPT 175B |
|------|-----------|-----------------|-----------------|
| 人类偏好胜率 | - | >50% (vs 175B GPT-3) | 85±3% (vs 175B GPT-3) |
| TruthfulQA | baseline | ~2x 提升 | ~2x 提升 |
| 毒性输出 | baseline | -25% | -25% |
| 幻觉率 | 41% | 21% | 21% |

---

## 层 3：深度精读

### 1. 研究动机与背景

#### 1.1 对齐问题的本质

**语言建模目标 vs 用户意图目标的差异**：

```
语言建模目标：预测网页上的下一个 token
  ↓
从互联网数据中学习，包含大量噪声、偏见、错误信息

用户意图目标：有帮助、诚实、无害地遵循指令
  ↓
需要对齐人类的价值观和期望
```

**Askell et al. (2021) 的三 H 标准**：
- **Helpful（有帮助）**：帮助用户完成任务
- **Honest（诚实）**：不编造信息、不误导用户
- **Harmless（无害）**：不对人、环境造成身心或社会伤害

#### 1.2 现有方法的局限性

| 方法 | 优点 | 缺点 |
|------|------|------|
| 扩大模型规模 | 能力提升 | 不 inherently 改善对齐 |
| 提示工程 | 无需训练 | 需要专业知识，不稳定 |
| 数据过滤 | 减少有害内容 | 性能下降，可能影响代表性群体 |
| 控制 token | 可 steering | 需要额外模型 |

### 2. 方法论详解

#### 2.1 数据收集

**Prompt 来源**：
1. **API 用户提交**（主要来源）
   - 来自 OpenAI API Playground
   - 使用早期 InstructGPT 模型（仅 SFT 训练）
   - 用户知情同意

2. **标注员撰写**（bootstrap 用）
   - Plain：任意任务，确保多样性
   - Few-shot：指令 + 多个查询/响应对
   - User-based：基于 API waitlist 用例

**数据集规模**：
| 数据集 | Prompt 数量 | 来源 | 用途 |
|--------|------------|------|------|
| SFT | ~13k | API + 标注员 | 监督微调 |
| RM | ~33k | API + 标注员 | 奖励模型训练 |
| PPO | ~31k | 仅 API | RLHF 优化 |

**Use-case 分布**：
| 类别 | 占比 |
|------|------|
| Generation | 45.6% |
| Open QA | 12.4% |
| Brainstorming | 11.2% |
| Chat | 8.4% |
| Rewrite | 6.6% |
| Summarization | 4.2% |
| Classification | 3.5% |
| Closed QA | 2.6% |
| Extract | 1.9% |

#### 2.2 三步训练流程

**Step 1: 监督微调 (Supervised Fine-Tuning, SFT)**

```python
# 伪代码
sft_model = GPT3_pretrained
sft_dataset = human_demonstrations  # 13k prompt-response pairs

# 标准交叉熵损失
loss = CrossEntropyLoss(sft_model(prompt), human_response)
sft_model.train(loss)
```

**关键细节**：
- 标注员：40 名承包商，基于筛选测试表现
- 标注指导：考虑显式意图（遵循指令）和隐式意图（真实性、无害性）
- 跳过任务不明确的情况

**Step 2: 奖励模型训练 (Reward Model, RM)**

```python
# 伪代码
rm = GPT3_pretrained(with_value_head)
comparison_data = [(prompt, output_A, output_B, preferred) for ...]  # 33k

# Bradley-Terry 模型
P(prefer A over B) = exp(r_A) / (exp(r_A) + exp(r_B))

loss = -log(P(prefer chosen over rejected))
rm.train(loss)
```

**关键细节**：
- 每个 prompt 收集 4-9 个模型输出的排名
- 使用所有成对比较训练
- RM 预测人类偏好概率

**Step 3: PPO 强化学习优化**

```python
# 伪代码
policy = sft_model
reward_model = rm  # frozen

for batch in ppo_dataset:  # 31k prompts
    outputs = policy.generate(batch)
    rewards = reward_model(outputs)

    # PPO 目标函数
    ratio = exp(log_policy(outputs) - log_old_policy(outputs))
    clipped_ratio = clip(ratio, 1-ε, 1+ε)
    ppo_loss = -min(ratio * advantages, clipped_ratio * advantages)

    # KL 惩罚（防止偏离 SFT 太远）
    kl_div = kl_divergence(policy, sft_model)
    total_loss = ppo_loss + β * kl_div

    policy.update(total_loss)
```

**PPO-ptx 变体**：
```python
# 混合预训练数据减少性能回归
pretraining_loss = CrossEntropyLoss(policy(pretrain_text), pretrain_text)
total_loss = ppo_loss + λ * pretraining_loss
```

#### 2.3 训练细节

| 超参数 | 值 |
|--------|-----|
| 标注员数量 | 40 |
| SFT 数据 | ~13k prompts |
| RM 数据 | ~33k prompts |
| PPO 数据 | ~31k prompts |
| 模型架构 | GPT-3 (1.3B, 6B, 175B) |
| RL 算法 | PPO (Schulman et al., 2017) |
| KL 惩罚系数 |  tuned per model size |

### 3. 实验结果

#### 3.1 人类偏好评估

**API Prompt 分布评估**：

![Figure 1](https://i.imgur.com/placeholder.png)

**关键发现**：
- **1.3B InstructGPT > 175B GPT-3**：尽管参数少 100 倍
- **175B InstructGPT 胜率 85±3%** vs 175B GPT-3
- **71±4% 胜率** vs few-shot 175B GPT-3

**评估设置**：
- 测试集：held-out 客户 prompt（不在训练数据中）
- 按用户 ID 划分 train/val/test
- 标注员评估输出质量

#### 3.2 真实性改进

**TruthfulQA Benchmark**：

| 模型 | 真实且信息丰富 (%) |
|------|-------------------|
| GPT-3 175B | baseline |
| InstructGPT | ~2x improvement |

**封闭领域任务幻觉率**：
| 任务类型 | GPT-3 | InstructGPT |
|----------|-------|-------------|
| 摘要/封闭 QA | 41% | 21% |

#### 3.3 毒性与偏见

**RealToxicityPrompts 评估**：
- InstructGPT 生成**25% 更少**有毒输出（当被要求尊重时）

**偏见评估（无显著改进）**：
| 数据集 | 结果 |
|--------|------|
| Winogender | 无显著改进 |
| CrowSPairs | 无显著改进 |

#### 3.4 对齐税 (Alignment Tax)

**性能回归的 NLP 数据集**：
| 数据集 | 任务类型 | 回归程度 |
|--------|----------|----------|
| SQuAD | 阅读理解 | 明显 |
| DROP | 推理 QA | 明显 |
| HellaSwag | 常识推理 | 明显 |
| WMT15 FR-EN | 翻译 | 明显 |

**PPO-ptx 解决方案**：
```
混合预训练数据 → 最小化性能回归 → 保持人类偏好分数
```

#### 3.5 泛化能力

**分布外指令泛化**：
| 任务类型 | GPT-3 | InstructGPT |
|----------|-------|-------------|
| 代码摘要 | 需要精心 prompt | 能直接遵循指令 |
| 代码问答 | 需要精心 prompt | 能直接遵循指令 |
| 多语言指令 | 通常不遵循 | 有时能遵循 |

**关键洞察**：InstructGPT 泛化了"遵循指令"的概念，即使在直接监督信号很少的任务上也保持对齐。

### 4. 局限性与讨论

#### 4.1 当前局限

1. **仍会犯简单错误**：
   - 不遵循指令
   - 编造事实
   - 对简单问题给出冗长的回避答案
   - 无法检测错误前提

2. **对齐到特定群体**：
   - 主要对齐标注员和研究者的偏好
   - 不代表更广泛的"人类价值观"

3. **偏见改进有限**：
   - Winogender 和 CrowSPairs 无显著提升

#### 4.2 公共 NLP 数据集的局限性

**关键发现**：公共 NLP 数据集（FLAN, T0）不能反映 API 使用方式

| 模型 | vs SFT baseline 胜率 |
|------|---------------------|
| InstructGPT | 73.4±2% |
| T0 | 26.8±2% |
| FLAN | 29.8±2% |

**含义**：在公共 NLP 数据集上微调不如在真实用户 prompt 上 RLHF 有效。

#### 4.3 held-out 标注员泛化

**初步实验**：
- held-out 标注员对 InstructGPT 的偏好率 ≈ 训练标注员
- 但需要更多研究 broader 用户群体

### 5. 与相关工作的对比

#### 5.1 与 Constitutional AI 对比

| 维度 | InstructGPT (RLHF) | Constitutional AI (RLAIF) |
|------|-------------------|--------------------------|
| 反馈来源 | 人类标注 | AI 自我批评 |
| 可扩展性 | 受限于人类标注成本 | 更易扩展 |
| 帮助性 - 无害性权衡 | 存在 | 缓解 |
| 透明度 | 有限 | 更好（chain-of-thought） |

#### 5.2 与 DPO 对比

| 维度 | InstructGPT (RLHF) | DPO |
|------|-------------------|-----|
| 训练方式 | 三步流程（SFT→RM→PPO） | 直接优化偏好 |
| 复杂度 | 高（需要 RM 和 RL） | 低（仅需分类损失） |
| 计算成本 | 高 | 低 |

### 6. 影响与启示

#### 6.1 对对齐研究的影响

1. **证明了 RLHF 的有效性**：在广泛任务上显著改善对齐
2. **揭示了对齐税**： alignment 可能以某些任务性能为代价
3. **提供了可扩展方法**：PPO-ptx 最小化性能回归

#### 6.2 实际应用价值

1. **ChatGPT 的基础**：InstructGPT 是 ChatGPT 的前身
2. **行业标准**：RLHF 成为大模型对齐的标准方法
3. **API 产品改进**：直接改善 OpenAI API 用户体验

#### 6.3 开放问题

1. **价值观对齐**：如何对齐到更广泛的人类价值观？
2. ** disagreement 处理**：人类对期望行为存在分歧时如何处理？
3. **更强模型的监督**：如何监督超越人类能力的模型？

### 7. 代码示例

#### 7.1 简化版 RLHF 训练流程

```python
import torch
from transformers import GPT2LMHeadModel
from trl import PPOTrainer, PPOConfig

# Step 1: 加载 SFT 模型
model = GPT2LMHeadModel.from_pretrained("sft-model")

# Step 2: 加载奖励模型
reward_model = AutoModelForSequenceClassification.from_pretrained("reward-model")
reward_model.eval()

# Step 3: PPO 配置
ppo_config = PPOConfig(
    batch_size=32,
    learning_rate=1e-5,
    ppo_epochs=4,
    kl_penalty="kl"
)

ppo_trainer = PPOTrainer(ppo_config, model, ref_model=model)

# Step 4: PPO 训练
for batch in ppo_dataloader:
    # 生成响应
    response_tensors = ppo_trainer.generate(batch["input_ids"])

    # 计算奖励
    rewards = []
    for response in response_tensors:
        reward = reward_model(response).logits
        rewards.append(reward)

    # PPO 更新
    stats = ppo_trainer.step(batch, response_tensors, rewards)
```

### 8. 术语表

| 术语 | 定义 |
|------|------|
| **RLHF** | Reinforcement Learning from Human Feedback，人类反馈强化学习 |
| **SFT** | Supervised Fine-Tuning，监督微调 |
| **RM** | Reward Model，奖励模型 |
| **PPO** | Proximal Policy Optimization，近端策略优化 |
| **PPO-ptx** | PPO with pretraining mix，混合预训练数据的 PPO |
| **Alignment Tax** | 对齐税，对齐过程导致的性能下降 |
| **Hallucination** | 幻觉，模型编造不真实信息 |

---

## 总结

InstructGPT 证明了通过人类反馈强化学习可以显著提升语言模型的对齐程度，其核心贡献在于：

1. **实用有效的三步流程**：SFT → RM → PPO
2. **小模型超越大模型**：1.3B InstructGPT > 175B GPT-3
3. **真实性显著提升**：TruthfulQA 提升 2 倍，幻觉率减半
4. **揭示并对齐税问题**：PPO-ptx 最小化性能回归

尽管存在局限性（仍会犯错、偏见改进有限、对齐到特定群体），InstructGPT 为后续 ChatGPT 和整个行业的 RLHF 实践奠定了基础。

---

**文档生成时间**: 2026-03-15
**论文验证**: ✓ PDF 内容与标题匹配
