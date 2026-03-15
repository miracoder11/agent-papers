# Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity

**论文信息**: William Fedus, Barret Zoph, Noam Shazeer, Google, 2021
**arXiv**: [2101.03961](https://arxiv.org/abs/2101.03961)
**发表于**: JMLR 2022
**引用数**: 2000+ (截至 2024 年)

---

## 层 1: 电梯演讲（30 秒）

**Switch Transformer 是什么**：一种稀疏 Mixture of Experts (MoE) 架构，通过简化 routing 算法（每个 token 只选 1 个 expert），成功训练出 1.6 万亿参数的 Transformer 模型，在相同计算成本下比稠密模型快 4 倍收敛。

**为什么重要**：首次证明稀疏激活可以扩展到万亿参数规模，为后续 GLaM、Mixtral、DeepSeek-V2 等 MoE 模型奠定了基础。

---

## 层 2: 故事摘要（5 分钟）

### 核心问题

2020 年，GPT-3 (175B) 展示了大规模语言模型的惊人能力，但训练成本高达数百万美元。Google 团队面临一个困境：

> **如何让模型更大，但计算成本不增加？**

### 关键洞察

Noam Shazeer（Switch 论文一作，也是 Transformer 和 GPT 的关键贡献者）和团队发现：

1. **稠密模型的低效**：每次推理都激活全部参数，计算成本随规模线性增长
2. **稀疏激活的潜力**：如果每次只激活一小部分参数，就能"免费"增加总参数量
3. **简化 routing**：之前的 MoE 太复杂，每个 token 选 2 个 expert 反而降低效率

### Switch 的解决方案

**Switch Routing**：每个 token 只选择 1 个最合适的 expert
- 简单：实现复杂度大幅降低
- 高效：通信开销减少 50%
- 可扩展：成功训练 1.6T 参数模型

### 研究成果

- 训练速度：比 T5-XXL 快 4 倍（达到相同 loss）
- 下游任务：GLUE/SuperGLUE 全面提升
- 规模突破：首次实现万亿参数 Transformer

---

## 层 3: 深度精读

### 1.1 研究背景：稠密模型的瓶颈

**时间**：2020-2021 年
**背景**：大模型面临计算成本墙

当时的困境：
- **GPT-3 (175B)** 训练成本数百万美元
- 每次推理都需要激活全部参数
- 计算成本随参数量线性增长

**核心问题**：
> **能否增加参数量但不增加计算成本？**

Google 团队的答案：**Sparse Mixture of Experts (MoE)**

### 1.2 MoE 的发展脉络

```
Noisy Net (Shazeer et al., 2017)
    ↓
GShard (Lepikhin et al., 2020)
    ↓
GSPMD (Xu et al., 2021)
    ↓
Switch Transformer (Fedus et al., 2021) ← 本论文
    ↓
GLaM (Du et al., 2022)
    ↓
Mixtral (Jiang et al., 2024)
```

### 1.3 Switch Transformer 的核心洞察

**传统 MoE 的问题**：
1. **复杂**：routing 算法过于复杂
2. **通信成本高**：expert 间数据传输开销大
3. **训练不稳定**：expert 负载不均衡导致崩溃

**Switch Transformer 的解决方案**：
1. **简化 routing**：每个 token 只选 1 个 expert（而不是 2 个）
2. **减少通信**：优化数据并行策略
3. **稳定训练**：引入 auxiliary loss 和 expert capacity 机制

**关键创新**：
> **"Switch" routing: 简单、高效、可扩展**

---

## 二、Switch Transformer 架构

### 2.1 整体架构

**Switch Transformer  vs  标准 Transformer**：

```
标准 Transformer:
┌─────────────────┐
│  Input Token    │
│       │         │
│       ▼         │
│  FFN Layer      │ ← 所有 token 共享相同参数
│       │         │
│       ▼         │
│  Output         │
└─────────────────┘

Switch Transformer:
┌─────────────────┐
│  Input Token    │
│       │         │
│       ▼         │
│  Router         │ ← 动态选择 expert
│       │         │
│   ┌──┴──┐      │
│   ▼    ▼       │
│ Expert1 Expert2│ ← 不同 token 用不同 expert
│   │    │       │
│   └────┘       │
│       │         │
│       ▼         │
│  Output         │
└─────────────────┘
```

### 2.2 Switch Routing 机制

#### 2.2.1 核心算法

**Switch Routing 的简化设计**：

对于每个 token $x$：

1. **计算 routing 分数**：
   $$h(x) = \text{Softmax}(W_r \cdot x)$$
   其中 $W_r \in \mathbb{R}^{N \times d}$，$N$ 是 expert 数量

2. **选择 Top-1 expert**：
   $$i = \arg\max_j h(x)_j$$

3. **只激活选中的 expert**：
   $$y = E_i(x)$$
   其中 $E_i$ 是第 $i$ 个 expert 的 FFN

**关键对比**：

| 特性 | 传统 MoE | Switch Transformer |
|------|---------|-------------------|
| Top-K | K=2 | K=1 |
| 激活参数 | 2x expert 大小 | 1x expert 大小 |
| 通信开销 | 高 | 低 |
| 实现复杂度 | 复杂 | 简单 |

#### 2.2.2 为什么 Switch 更有效？

**直觉理解**：

想象一个医院分诊系统：

**传统 MoE（K=2）**：
- 每个病人看 2 个专家
- 专家需要协调意见
- 通信成本高

**Switch（K=1）**：
- 每个病人看 1 个最合适的专家
- 专家独立工作
- 效率更高

**技术优势**：
1. **减少通信**：只需传输到 1 个 expert
2. **简化实现**：不需要 expert 间协调
3. **更低延迟**：单次 expert 调用

### 2.3 专家并行（Expert Parallelism）

**数据分布策略**：

```
传统数据并行:
GPU1: [batch 1-32] → Model → [output 1-32]
GPU2: [batch 33-64] → Model → [output 33-64]

Switch Expert 并行:
GPU1: Expert 1-4, [tokens routed to 1-4]
GPU2: Expert 5-8, [tokens routed to 5-8]
GPU3: Expert 9-12, [tokens routed to 9-12]
GPU4: Expert 13-16, [tokens routed to 13-16]
```

**优势**：
- 每个 GPU 只存储部分 expert
- 通信仅发生在 cross-GPU routing 时
- 可以扩展到数百个 expert

### 2.4 训练稳定技术

#### 2.4.1 问题：Expert 负载不均衡

**现象**：
- 某些 expert 被过度使用（overloaded）
- 某些 expert 几乎闲置（underutilized）
- 导致训练不稳定甚至崩溃

#### 2.4.2 解决方案：Auxiliary Loss

**Auxiliary Loss 设计**：

$$\text{aux\_loss} = N \cdot \sum_{i=1}^{N} f_i \cdot P_i$$

其中：
- $N$ = expert 数量
- $f_i$ = 分配给 expert $i$ 的 token 比例
- $P_i$ = routing 到 expert $i$ 的 token 概率和

**作用**：
- 鼓励均匀的 expert 使用
- 防止某些 expert 被过度使用
- 典型权重：$\lambda = 0.01$

#### 2.4.3 Expert Capacity 机制

**Expert Capacity**：
- 限制每个 expert 能处理的 token 数量
- 超出容量的 token 被"dropped"（跳过）

$$\text{capacity} = \text{batch\_size} \times \frac{\text{capacity\_factor}}{N}$$

**Capacity Factor 选择**：
- 太大：浪费计算
- 太小：太多 token 被 drop
- 推荐值：1.0-1.25

#### 2.4.4 混合精度训练

**创新**：首次用 bfloat16 训练大规模 MoE

**技巧**：
- Router 用 FP32（保证 routing 精度）
- Expert 内部用 BF16
- Loss scaling 防止梯度消失

---

## 三、实验结果

### 3.1 模型配置

**Switch Transformer 系列**：

| 模型 | 层数 | 隐藏维度 | Expert 数 | 总参数 | 激活参数 |
|------|------|---------|----------|--------|---------|
| Switch-Base | 12 | 768 | 128 | 3.7B | 0.8B |
| Switch-Large | 24 | 1024 | 256 | 15.7B | 1.6B |
| Switch-XXL | 36 | 2048 | 512 | 87B | 3.2B |
| Switch-T | 120 | 4096 | 2048 | **1.6T** | 8.5B |

### 3.2 预训练速度对比

**vs T5-XXL（稠密模型）**：

| 模型 | 达到相同 loss 的时间 | 相对速度 |
|------|-------------------|---------|
| T5-XXL (11B) | 100% | 1.0x |
| Switch-Large (16B) | 67% | 1.5x |
| Switch-XXL (87B) | 50% | 2.0x |

**Switch Transformer 用 1/4 时间达到相同 loss！**

### 3.3 下游任务性能

#### 3.3.1 语言理解（GLUE）

| 模型 | 参数量 | GLUE 得分 |
|------|--------|---------|
| T5-Base | 220M | 82.8 |
| Switch-Base | 3.7B (0.8B 激活) | 84.6 |
| T5-XXL | 11B | 87.8 |
| Switch-XXL | 87B (3.2B 激活) | **89.3** |

#### 3.3.2 语言生成（SuperGLUE）

| 模型 | 参数量 | SuperGLUE |
|------|--------|-----------|
| T5-Large | 770M | 75.1 |
| Switch-Large | 16B (1.6B 激活) | **80.2** |

#### 3.3.3 多语言（mT5）

**mT5-XXL vs Switch-mT5-XXL**：

| 任务 | T5-XXL | Switch-XXL | 提升 |
|------|--------|-----------|------|
| XNLI | 73.2 | 76.8 | +3.6 |
| XQuAD | 78.5 | 82.1 | +3.6 |
| MLQA | 56.3 | 60.2 | +3.9 |

### 3.4 扩展到万亿参数

**Switch-T (1.6T 参数)**：

- 总参数：1.6 万亿
- 激活参数：8.5B（每 token）
- Expert 数量：2048
- 层数：120

**训练效率**：
- 使用 bfloat16 混合精度
- 在 TPU v3 Pod 上训练
- 达到 90%+ 的模型利用率

---

## 四、技术细节与实现

### 4.1 Routing 算法伪代码

```python
def switch_routing(x, num_experts):
    """
    Switch Routing: 每个 token 选择 1 个 expert

    Args:
        x: input tokens (batch_size, seq_len, d_model)
        num_experts: 专家数量

    Returns:
        expert_assignments: 每个 token 分配的 expert
        router_probs: routing 概率
    """
    # 1. 计算 routing logits
    router_logits = torch.matmul(x, router_weights)  # (batch, seq, num_experts)

    # 2. Softmax 得到概率
    router_probs = torch.softmax(router_logits, dim=-1)

    # 3. Top-1 选择
    expert_assignments = torch.argmax(router_probs, dim=-1)  # (batch, seq)

    return expert_assignments, router_probs

def switch_ffn(x, experts, expert_assignments):
    """
    Switch FFN: 只激活选中的 expert

    Args:
        x: input tokens
        experts: ModuleList of expert FFNs
        expert_assignments: 每个 token 的 expert 分配

    Returns:
        output: 处理后的输出
    """
    batch_size, seq_len, _ = x.shape
    output = torch.zeros_like(x)

    for expert_idx in range(len(experts)):
        # 找到分配给当前 expert 的 token
        mask = (expert_assignments == expert_idx)
        if not mask.any():
            continue

        # 提取 token
        selected_tokens = x[mask]

        # 通过 expert 处理
        expert_output = experts[expert_idx](selected_tokens)

        # 放回原位置
        output[mask] = expert_output

    return output
```

### 4.2 辅助 Loss 实现

```python
def auxiliary_loss(router_probs, num_experts):
    """
    计算 auxiliary loss 以鼓励 expert 负载均衡

    Args:
        router_probs: routing 概率 (batch, seq, num_experts)
        num_experts: 专家数量

    Returns:
        aux_loss: 辅助损失
    """
    # 1. 计算每个 expert 的 token 比例 (f_i)
    expert_counts = torch.sum(torch.argmax(router_probs, dim=-1) == torch.arange(num_experts).to(router_probs.device), dim=(0, 1))
    f_i = expert_counts.float() / expert_counts.sum()

    # 2. 计算 routing 概率和 (P_i)
    P_i = torch.sum(router_probs, dim=(0, 1))  # (num_experts,)

    # 3. 计算 aux loss
    aux_loss = num_experts * torch.sum(f_i * P_i)

    return aux_loss

# 总 loss = main_loss + lambda * aux_loss
total_loss = main_loss + 0.01 * aux_loss
```

### 4.3 训练技巧

```python
class SwitchTransformerTrainer:
    def __init__(self, model, num_experts, capacity_factor=1.25):
        self.model = model
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor

    def train_step(self, batch):
        # 1. Forward pass
        outputs, router_probs = self.model(batch)

        # 2. 计算 main loss
        main_loss = cross_entropy(outputs, batch.targets)

        # 3. 计算 auxiliary loss
        aux_loss = self.auxiliary_loss(router_probs)

        # 4. 检查 expert capacity
        dropped_tokens = self.check_capacity(router_probs)
        if dropped_tokens > 0.01:  # 超过 1% 的 token 被 drop
            # 增加 capacity factor 或调整学习率

        # 5. 总 loss
        total_loss = main_loss + 0.01 * aux_loss

        # 6. Backward pass
        total_loss.backward()

        # 7. 梯度裁剪（重要！）
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # 8. Optimizer step
        optimizer.step()

        return total_loss.item()
```

---

## 五、与相关工作的对比

### 5.1 MoE 架构演进

| 模型 | Top-K | Routing | Auxiliary Loss | 最大规模 |
|------|-------|---------|----------------|---------|
| GShard | 2 | Top-2 | ✅ | 300B+ |
| GSPMD | 2 | Top-2 | ✅ | 500B+ |
| **Switch Transformer** | **1** | **Top-1** | ✅ | **1.6T** |
| GLaM | 2 | Top-2 | ✅ | 1.2T |
| Mixtral | 2 | Top-2 | ✅ | 47B |

### 5.2 与稠密模型对比

| 特性 | 稠密模型 | Switch Transformer |
|------|---------|-------------------|
| 参数量 | 受限（计算成本） | 可扩展到万亿 |
| 激活参数 | 100% | ~5-10% |
| 训练速度 | 较慢 | 更快（相同 loss） |
| 推理延迟 | 高 | 低（相同性能） |
| 实现难度 | 简单 | 中等 |
| 通信开销 | 低 | 中等（需 expert 路由） |

### 5.3 与后续模型对比

| 模型 | 继承自 Switch | 改进点 |
|------|-------------|--------|
| **GLaM** | ✅ | 动态 expert 选择 |
| **Mixtral** | ✅ | 更细粒度 expert |
| **DeepSeek-V2** | ✅ | Shared expert + fine-grained |
| **GPT-4** | 推测 ✅ | 未公开细节 |

---

## 六、局限性与挑战

### 6.1 训练挑战

1. **Expert 负载均衡**：
   - 需要 careful tuning of auxiliary loss
   - 否则训练可能崩溃

2. **通信开销**：
   - 跨设备 routing 增加通信
   - 需要优化的并行策略

3. **超参数敏感**：
   - Capacity factor 需要调优
   - 学习率调度更复杂

### 6.2 推理挑战

1. **显存占用**：
   - 总参数巨大，需要分布式推理
   - 单 GPU 无法加载完整模型

2. **动态 routing**：
   - 增加推理复杂度
   - 需要专门的推理引擎

---

## 七、总结

### 7.1 核心贡献

1. **简化 MoE routing**：
   - Switch routing（Top-1）比传统方法更简单高效
   - 减少通信开销，提升训练速度

2. **训练稳定技术**：
   - Auxiliary loss 保证负载均衡
   - Expert capacity 防止过载
   - 首次用 bfloat16 训练万亿参数模型

3. **规模突破**：
   - 首次训练出 1.6T 参数的 Transformer
   - 在相同计算量下达到更快收敛

4. **开源影响**：
   - 启发了 GLaM、Mixtral、DeepSeek-V2 等后续工作
   - MoE 成为大模型的主流架构选择

### 7.2 历史地位

Switch Transformer 是稀疏模型发展的重要里程碑：

- **技术价值**：简化了 MoE 实现，使大规模稀疏训练成为可能
- **实践价值**：直接影响了 Google 后续大模型的设计
- **生态价值**：推动了 MoE 在开源社区的普及

它是通向万亿参数模型的关键一步。

---

## 参考文献

1. Fedus, W., Zoph, B., & Shazeer, N. (2022). Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity. JMLR, 23(1).
2. Lepikhin, D., et al. (2020). GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding. ICLR 2021.
3. Shazeer, N., et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer. ICLR 2017.
4. Du, N., et al. (2022). GLaM: Efficient Scaling of Language Models with Mixture-of-Experts. ICML 2022.
5. Jiang, A. Q., et al. (2024). Mixtral of Experts. arXiv:2401.04088.
