# 大模型论文阅读清单

本目录收录了约 52 篇大模型相关核心论文，分为三个类别：

## 01_Core_Foundations (核心基础篇) - 18 篇

这些是理解大模型的绝对基础，必须精读：

| 序号 | 论文 | 简介 |
|------|------|------|
| 01 | Attention Is All You Need | Transformer 架构的起源，大模型的基石 |
| 02 | Language Models are Few-Shot Learners (GPT-3) | 展示大规模语言模型的涌现能力 |
| 03 | Sequence to Sequence Learning with Neural Networks | Seq2Seq 模型，机器翻译的基础 |
| 04 | Neural Machine Translation by Jointly Learning to Align and Translate | 引入注意力机制的早期工作 |
| 05 | Adam: A Method for Stochastic Optimization | 最常用的优化算法 |
| 06 | Batch Normalization: Accelerating Deep Network Training | 加速训练的标准组件 |
| 07 | Distilling the Knowledge in a Neural Network | 知识蒸馏开山之作 |
| 08 | Show, Attend and Tell | 视觉注意力机制经典论文 |
| 09 | Deep Residual Learning for Image Recognition (ResNet) | 残差网络，解决深层网络退化 |
| 10 | Very Deep Convolutional Networks (VGG) | VGG 网络结构 |
| 11 | Auto-Encoding Variational Bayes (VAE) | 变分自编码器 |
| 12 | Generative Adversarial Nets (GAN) | 生成对抗网络 |
| 13 | Unsupervised Representation Learning with DCGAN | DCGAN |
| 15 | Long Short-Term Memory (Hochreiter) | LSTM，解决长序列依赖 |
| 16 | Deep Learning (Goodfellow Book) | 深度学习圣经（书/综述） |
| 17 | ImageNet Classification with Deep Convolutional Neural Networks (AlexNet) | 深度学习革命的开端 |
| 18 | Rectified Linear Units Improve Restricted Boltzmann Machines (ReLU) | ReLU 激活函数 |
| 19 | Very Deep Convolutional Networks for Large-Scale Image Recognition | VGG 完整版 |

## 02_Advanced_Alignment (大模型进阶与对齐篇) - 25 篇

这些论文定义了 ChatGPT 及后续模型的技术路线：

| 序号 | 论文 | 简介 |
|------|------|------|
| 01 | InstructGPT | 引入指令微调和 RLHF |
| 02 | Chain-of-Thought Prompting | 思维链，激发推理能力 |
| 03 | Self-Consistency Improves Chain of Thought | 自一致性，提升 CoT 效果 |
| 04 | LoRA | 高效微调技术代表 |
| 05 | Scaling Laws for Neural Language Models | 模型性能随规模扩大规律 |
| 06 | Instruction Tuning with GPT-4 | GPT-4 指令微调 |
| 07 | LLaMA | 开源大模型代表作 |
| 08 | Constitutional AI | 无需人类反馈的自我对齐 |
| 09 | QLoRA | 高效量化微调技术 |
| 10 | FlashAttention | 大幅提升 Attention 效率 |
| 11 | RoFormer | RoPE 位置编码 |
| 12 | Improving Language Understanding by Generative Pre-Training | GPT 预训练 |
| 13 | Learning to Summarize from Human Feedback | 人类反馈摘要 |
| 14 | Reformer | 高效 Transformer |
| 15 | Multitask Prompted Training | 多任务提示训练 |
| 16 | ToolLLM | 让 LLM 掌握 16000+API |
| 17 | Training Compute-Optimal Large Language Models (Chinchilla) | 修正 Scaling Law |
| 18 | Direct Preference Optimization (DPO) | 简化 RLHF 的新范式 |
| 19 | Switch Transformers | 稀疏 MoE 模型架构 |
| 20 | Vision Transformer (ViT) | Transformer 应用于视觉 |
| 21 | CLIP | 图文对比学习里程碑 |
| 22 | Diffusion Models Beat GANs | 扩散模型超越 GAN |
| 23 | High-Resolution Image Synthesis with Latent Diffusion Models | Stable Diffusion 核心 |
| 24 | Toolformer | 让模型学会使用工具 |
| 25 | ReAct | 结合推理与行动的 Agent 框架 |

## 03_Frontier_Exploration (前沿探索与反思篇) - 9 篇

2024-2026 前沿探索与反思：

| 序号 | 论文 | 简介 |
|------|------|------|
| 01 | GPT-4 Technical Report | GPT-4 技术报告 |
| 02 | DeepSeek LLM | 中国开源大模型 |
| 03 | DeepSeek-V2 | MoE 架构创新 |
| 04 | Judging LLM-as-a-Judge | LLM 作为评判者 |
| 05 | Sora | 文生视频模型 |
| 06 | Implicit Chain of Thought Reasoning | 隐式思维链 |
| 07 | Large Language Models are Human-Level Prompt Engineers | 提示工程 |
| 08 | Mechanistic Interpretability of Language Models | 机制可解释性 |
| 09 | World Models | 世界模型早期构想 |

## 阅读建议

1. **初学者**：从核心基础篇开始，按顺序阅读 01-10 篇
2. **进阶学习**：重点阅读大模型进阶篇的 InstructGPT、LoRA、Scaling Laws
3. **前沿追踪**：关注前沿探索篇的最新技术报告

---
*最后更新：2026 年 3 月*
