# DeepSeek LLM: Scaling Open-Source Language Models with Longtermism

**论文信息**: DeepSeek-AI, 2024
**arXiv**: [2401.02954](https://arxiv.org/abs/2401.02954)
**发布时间**: 2024 年 1 月

---

## 一、核心思想：长期主义的胜利

### 1.1 研究背景：Scaling Laws 的迷雾

**时间**：2023 年末
**背景**：开源大模型的迷茫期

当时的局面：
- LLaMA 证明了开源模型的可行性
- 但 Scaling Laws 的结论存在分歧
- Kaplan (2020) vs Chinchilla (2022) 的最优策略不同
- 开源社区不知道应该如何扩展模型

**核心问题**：
> **在开源配置下（7B、65B），Scaling Laws 到底是什么？**
> **如何长期可持续地扩展开源模型？**

DeepSeek 团队决定：
1. 系统性地研究 Scaling Laws
2. 基于研究结果训练自己的模型
3. 坚持长期主义，不追求短期突破

### 1.2 长期主义的理念

**什么是长期主义？**

DeepSeek 在论文中明确提出"Longtermism"理念：

1. **不追求短期 SOTA**
   - 不急于发布模型抢headline
   - 专注于基础研究和长期价值

2. **系统性研究 Scaling**
   - 先研究规律，再训练模型
   - 基于数据做决策，而非直觉

3. **开源承诺**
   - 所有模型对社区开放
   - 推动整个生态发展

这种理念在"快节奏"的 AI 界是一股清流。

---

## 二、Scaling Laws 研究发现

### 2.1 实验设计

团队进行了系统的 Scaling 实验：

**模型规模**：
- 从 1B 到 67B 参数
- 覆盖开源社区常用配置

**训练数据**：
- 2T 高质量 tokens
- 多来源、多语言

**评估维度**：
- 训练 loss 曲线
- 下游任务性能
- 计算效率

### 2.2 核心发现

#### 2.2.1 Loss 预测公式

团队提出了改进的 loss 预测公式：

$$L(N, D) = \frac{A}{N^\alpha} + \frac{B}{D^\beta} + E$$

其中：
- $N$ = 参数量
- $D$ = 训练数据量
- $A, B, \alpha, \beta, E$ 是拟合参数

#### 2.2.2 最优数据量

对于 7B 和 67B 配置，团队发现：

| 模型规模 | 最优数据量 |
|---------|-----------|
| 7B | ~1T tokens |
| 67B | ~2T tokens |

这个比例介于 Kaplan 和 Chinchilla 之间。

#### 2.2.3 收敛行为

关键发现：
- 大模型在相同数据密度下收敛速度相似
- 不存在"突然的 phase change"
- Scaling 是平滑的、可预测的

---

## 三、DeepSeek 模型系列

### 3.1 DeepSeek 7B

**配置**：
- 参数量：7B
- 训练数据：1T+ tokens
- 上下文长度：4096

**性能**：
- 超越 LLaMA-7B 在所有基准上
- 接近 LLaMA-13B 的性能

### 3.2 DeepSeek 67B

**配置**：
- 参数量：67B
- 训练数据：2T+ tokens
- 上下文长度：4096

**性能对比**：

| 基准 | LLaMA-65B | DeepSeek 67B | 提升 |
|------|-----------|--------------|------|
| MMLU | 57.8% | 61.2% | +3.4% |
| GSM8K | 40.3% | 48.5% | +8.2% |
| HumanEval | 15.8% | 22.6% | +6.8% |
| MBPP | 31.2% | 39.4% | +8.2% |

在相同规模下，DeepSeek 67B 显著超越 LLaMA-65B。

---

## 四、技术特点

### 4.1 架构设计

DeepSeek 采用现代化的 Transformer 架构：

- **RMSNorm**: 更高效的归一化
- **RoPE**: 旋转位置编码
- **SwiGLU**: 门控激活函数
- **Multi-Query Attention**: 加速推理

### 4.2 训练数据策略

**数据来源**：
- 公开网页数据（经过严格过滤）
- 代码数据（GitHub、StackOverflow）
- 数学和科学数据
- 多语言数据（中英文为主）

**数据处理**：
- 去重和过滤
- 质量评分
- 多样性平衡

### 4.3 分词设计

**词表特点**：
- 100K+ 词表大小
- 支持中英文混合
- 优化的代码 tokenization

---

## 五、与 Scaling Laws 论文的关系

### 5.1 验证与扩展

DeepSeek 的工作验证了：

1. **Chinchilla 的核心结论**
   - 等比例扩展是有效的
   - 现有模型普遍欠训练

2. **但存在细微差异**
   - 最优数据量略低于 Chinchilla 预测
   - 可能与数据质量有关

### 5.2 实践指导

DeepSeek 基于 Scaling Laws 研究：

1. **确定训练目标**
   - 根据计算预算选择模型规模
   - 确定对应的最优数据量

2. **预测最终性能**
   - 用 fitted scaling curve 预测 loss
   - 指导 early stopping 决策

3. **资源分配**
   - 平衡模型规模和数据量
   - 避免欠训练或过训练

---

## 六、总结

### 6.1 核心贡献

1. **系统性 Scaling 研究**
   - 针对开源配置的详细分析
   - 提供实用的训练指导

2. **高质量模型发布**
   - DeepSeek 7B 和 67B
   - 性能超越同规模 LLaMA

3. **长期主义理念**
   - 不追求短期 SOTA
   - 专注于可持续的进步

### 6.2 对开源社区的意义

DeepSeek 展示了：
- 开源模型可以持续改进
- 基于研究而非直觉做决策
- 长期投入带来长期回报

---

## 参考文献

1. DeepSeek-AI. (2024). DeepSeek LLM: Scaling Open-Source Language Models with Longtermism. arXiv:2401.02954.
2. Hoffmann, J., et al. (2022). Training Compute-Optimal Large Language Models (Chinchilla). arXiv:2203.15556.
3. Kaplan, J., et al. (2020). Scaling Laws for Neural Language Models. arXiv:2001.08361.
4. Touvron, H., et al. (2023). LLaMA: Open and Efficient Foundation Language Models. arXiv:2302.13971.
