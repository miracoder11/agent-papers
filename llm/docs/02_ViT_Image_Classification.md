# ViT: 当 Transformer 遇见视觉——纯 Attention 如何征服计算机视觉

## 层 1: 电梯演讲

**一句话概括**：针对 CNN 在计算机视觉中的统治地位，Google Brain 提出纯 Transformer 架构 Vision Transformer (ViT)，将图像切分为 16×16 像素块序列，在大规模数据预训练后，ImageNet 分类准确率超越 ResNet 等 CNN 架构，开创视觉 Transformer 时代。

---

## 层 2: 故事摘要 (5 分钟读完)

**核心问题**：2020 年，计算机视觉被 CNN（尤其是 ResNet）完全统治。Transformer 虽已在 NLP 领域成为标准，但在视觉领域的应用仍然有限——要么与 CNN 结合使用，要么只能替换部分组件。核心问题是：视觉是否需要卷积的归纳偏置（平移不变性、局部性）？

**关键洞察**：2020 年，Alexey Dosovitskiy 和团队提出一个激进问题："如果完全去掉卷积，只用 Transformer，会怎样？" 他们发现，当在大规模数据（JFT-300M，3 亿图像）上预训练时，ViT 的表现超越 CNN，且计算资源需求更少。

**解决方案**：将图像切分为固定大小的 patch（16×16 像素），线性嵌入为向量，加上位置编码，输入标准 Transformer Encoder。使用 [CLS] token 进行分类。

**验证结果**：
- ImageNet 分类：ViT-L/16 (JFT-300M 预训练) 达到 88.55% 准确率，超越 BiT ResNet
- CIFAR-100：99.5% 准确率
- VTAB：在多个任务上达到 SOTA
- 关键发现：大规模训练胜过归纳偏置

---

## 框架大图

```
┌─────────────────────────────────────────────────────────────────┐
│                        问题空间                                  │
│  ┌──────────────────────────────────────────────────────┐       │
│  │  CNN 的统治地位 (2020)                                │       │
│  │  - ResNet/DenseNet 等主导图像分类                    │       │
│  │  - 卷积的归纳偏置：局部性、平移不变性                 │       │
│  │  - Transformer 在视觉中仅作为辅助组件                 │       │
│  └──────────────────────┬───────────────────────────────┘       │
│                         │                                       │
│         ┌───────────────▼───────────────┐                       │
│         │  核心问题                      │                       │
│         │  "视觉是否真的需要卷积？"      │                       │
│         │  归纳偏置 vs 大规模训练        │                       │
│         └───────────────┬───────────────┘                       │
└─────────────────────────┼───────────────────────────────────────┘
                          │
                          ▼
              ┌─────────────────────────┐
              │   ViT 核心方法           │
              │                         │
              │  图像 → Patch 序列        │
              │  ┌───────────────────┐  │
              │  │ 224×224 图像       │  │
              │  │ ↓ 切分             │  │
              │  │ 196 个 16×16 patches│  │
              │  │ ↓ 线性嵌入         │  │
              │  │ 196 个 768 维向量    │  │
              │  │ + [CLS] token      │  │
              │  │ + Position Embed  │  │
              │  │ ↓ Transformer      │  │
              │  │ [CLS] → 分类结果   │  │
              │  └───────────────────┘  │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │        验证层            │
              │  ImageNet: 88.55%      │
              │  CIFAR-100: 99.5%      │
              │  VTAB: 多任务 SOTA      │
              │                         │
              │  关键发现：             │
              │  大数据 > 归纳偏置      │
              └─────────────────────────┘
```

---

## 预期 vs 实际对比表

| 维度 | 预期假设 | 实际发现 | 洞察 |
|------|----------|----------|------|
| **CNN vs Transformer** | CNN 的归纳偏置对视觉至关重要 | 大规模训练下 ViT 超越 CNN | 数据量足够大时，归纳偏置不再是优势 |
| **小数据集表现** | ViT 应该与 CNN 相当 | ViT 在小数据集上显著差于 CNN | ViT 缺乏归纳偏置，需要大数据"弥补" |
| **计算效率** | Transformer 应该更慢 | ViT 训练效率高于同规模 CNN | Attention 并行化更好，无卷积的滑动窗口开销 |
| **Patch 大小** | 越小越好（保留更多信息） | 16×16 是最佳平衡点 | 太小计算量爆炸，太大丢失细节 |

---

## 论文定位图谱

```
                    上游工作 (它解决了谁的问题)
                    ┌─────────────────────────┐
                    │   ImageNet / ResNet     │
                    │  - CNN 主导图像分类      │
                    │  - 卷积归纳偏置          │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │   Transformer (2017)    │
                    │  - NLP 领域的 SOTA       │
                    │  - 纯 Attention 架构     │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │   BERT (2018)           │
                    │  - [CLS] token 用于分类  │
                    │  - 预训练 + 微调范式     │
                    └───────────┬─────────────┘
                                │
          ┌─────────────────────┼─────────────────────┐
          │                     ▼                     │
          │            ┌─────────────────┐            │
          │            │      ViT        │            │
          │            │  (2020) 本研究   │            │
          │            └────────┬────────┘            │
          │                     │                     │
          ┼─────────────────────┼─────────────────────┼
    并行工作                     │              并行工作
┌──────────────────┐            │        ┌──────────────────┐
│  DeiT            │            │        │  BEiT            │
│  知识蒸馏训练 ViT │            │        │  Masked 预训练    │
└──────────────────┘            │        └──────────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │   Swin Transformer      │
                    │  - 层级式 ViT           │
                    │  - 滑动窗口 Attention   │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │   CLIP / ALIGN          │
                    │  - 视觉 - 语言联合预训练  │
                    │  - ViT 作为 vision encoder│
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │   MAE / DINOv2          │
                    │  - 自监督 ViT 预训练     │
                    │  - 更强的表征能力        │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │   Sora / Video ViT      │
                    │  - 时空 Patch           │
                    │  - 视频生成模型          │
                    └─────────────────────────┘

         下游工作 (谁解决了它的问题/扩展了它)
```

---

## 开场：一个"疯狂"的想法

2020 年初，Google Brain 团队的办公室里，Alexey Dosovitskiy 提出了一个在当时看来"疯狂"的想法。

"如果我们，"他在白板上画了一个Transformer 框图，"完全去掉卷积，只用Transformer 处理图像，会怎样？"

会议室里陷入了短暂的沉默。

"可是，"有人问，"CNN 在图像分类上这么成功，ResNet 已经做到 90%+ 的 ImageNet 准确率。Transformer 没有卷积的归纳偏置——平移不变性、局部性——怎么可能工作？"

这确实是一个激进的想法。当时的情况是：
- CNN 已经统治计算机视觉 20 多年（从 LeNet 到 ResNet）
- Transformer 在 NLP 领域大获成功（BERT、GPT-2）
- 但在视觉领域，Transformer 只能作为 CNN 的"配件"——要么替换某些层，要么与卷积结合使用

"我们知道 Transformer 在 NLP 中成功的关键，" Alexey 继续说，"是大规模预训练。如果在足够大的数据集上训练，Transformer 能否自己学习到视觉所需的归纳偏置？"

这个想法听起来简单，但挑战巨大。视觉数据与文本数据有着本质的不同：
- 文本是 1D 序列，图像是 2D 网格
- 文本 token 有明确语义，像素本身没有意义
- 图像的空间结构至关重要

但 Alexey 团队决定尝试。他们的实验结果将彻底改变计算机视觉的面貌。

---

## 第一章：CNN 的统治与 Transformer 的挑战

### CNN 的三大归纳偏置

2020 年的计算机视觉，CNN 是绝对王者。ResNet-50、EfficientNet、DenseNet... 这些模型在 ImageNet 上的准确率已经突破 90%。

CNN 成功的关键在于其内置的**归纳偏置**：

**1. 局部性（Locality）**
卷积核只关注局部区域（如 3×3），这符合图像的本质——相邻像素通常相关。

**2. 平移不变性（Translation Invariance）**
无论物体在图像的哪个位置，卷积核都能检测到。这是通过权重共享实现的。

**3. 层次结构（Hierarchy）**
浅层检测边缘，中层检测纹理，高层检测物体部件——这是 CNN 的天然层次。

这些归纳偏置让 CNN 在**小数据集**上也能工作良好。即使只有 1000 张图像，CNN 也能学到有用的特征。

### Transformer 的困境

Transformer 在 NLP 领域的成功有目共睹：
- BERT 在 11 个 NLP 任务上达到 SOTA
- GPT-3 展现出惊人的few-shot 能力
- 机器翻译、文本生成几乎完全转向 Transformer

但在视觉领域，Transformer 的应用却十分有限：

**尝试 1：CNN + Attention 混合**
```
图像 → CNN 提取特征 → Attention 增强 → 分类
```
问题：仍然依赖 CNN 提取基础特征，Transformer 只是"锦上添花"。

**尝试 2：用 Transformer 替换部分 CNN 层**
```
图像 → Conv 层 → Transformer 层 → Conv 层 → 分类
```
问题：保留了 CNN 的整体结构，Transformer 的作用受限。

**核心问题**：为什么纯 Transformer 在视觉上无法工作？

当时的共识是：**视觉需要归纳偏置**，而 Transformer 没有。

---

## 第二章：试错的旅程

### 第一阶段：如何把图像变成序列？

Transformer 处理的是 1D 序列（如文本），但图像是 2D 网格。如何将图像转换为 Transformer 可接受的格式？

**尝试 1：逐像素展开**
```
224×224 图像 → 50176 个像素 → 序列长度 50176
```
问题：序列太长！Transformer 的 Attention 复杂度是 O(n²)，50000+ 的序列长度无法处理。

**尝试 2：下采样后展开**
```
224×224 → 平均池化到 56×56 → 3136 个像素
```
问题：仍然太长，且丢失信息。

**突破：Patch Embedding**

团队想到一个巧妙的方案：将图像切分为固定大小的 patch。

```
224×224 图像，Patch 大小 16×16
→ (224/16) × (224/16) = 14 × 14 = 196 个 patches
→ 序列长度 196
```

这个想法的灵感来自 NLP 中的 tokenization：
- 文本被切分为 wordpiece token（约 30000 个）
- 图像被切分为 visual patches（196 个）

每个 patch 是一个 16×16×3 的小图像（3 是 RGB 通道）。通过一个线性层，将 patch 展平并投影到固定维度（如 768）：

```python
# Patch Embedding 的数学形式
# patch: 16×16×3 = 768 维
# E: 768×768 投影矩阵
patch_embed = patch_flatten @ E  # 768 维向量
```

### 第二阶段：位置信息与 [CLS] Token

Transformer 没有内在的位置概念（如 CNN 的 2D 网格结构），如何注入位置信息？

团队借鉴了原始 Transformer 的做法：**可学习的位置嵌入**。

```
z_0 = [x_class; x_p^1 E; x_p^2 E; ...; x_p^N E] + E_pos
```

其中：
- `x_class` 是 [CLS] token（可学习的分类标记）
- `x_p^i E` 是第 i 个 patch 的嵌入
- `E_pos` 是位置嵌入（197×768，可学习）

**[CLS] Token 的作用**

[CLS] token 的概念来自 BERT。它的设计直觉是：
- 放在序列开头
- 通过 Attention 机制"收集"整个序列的信息
- 最终用于分类

```
Transformer Encoder 输出 → [CLS] 位置的向量 → MLP Head → 分类结果
```

### 第三阶段：模型变体与训练策略

团队设计了三个 ViT 变体：

| 模型 | 层数 | 隐藏维度 | Attention 头数 | 参数量 |
|------|------|----------|---------------|--------|
| ViT-Base | 12 | 768 | 12 | 86M |
| ViT-Large | 24 | 1024 | 16 | 307M |
| ViT-Huge | 32 | 1280 | 16 | 632M |

训练策略借鉴了 NLP 的成功经验：

**预训练 + 微调范式**
```
1. 在大规模数据集上预训练（JFT-300M 或 ImageNet-21k）
   - 任务：图像分类（21843 类或 1000 类）
   - 优化器：Adam，learning rate warmup + decay

2. 在下游数据集上微调（ImageNet-1k、CIFAR-100 等）
   - 任务：特定数据集的分类
   - 技巧：高分辨率 finetuning，dropout 增加
```

### 第四阶段：关键发现——规模胜过归纳偏置

实验结果出来后，团队发现了一个反直觉的现象：

**小数据 regime（ImageNet 从头训练）**：
```
ResNet-50:    76.5% Top-1
ViT-Base:     70.9% Top-1  (更差！)
```

**大规模预训练（JFT-300M → ImageNet 微调）**：
```
BiT-ResNet50: 87.5% Top-1
ViT-L/16:     88.6% Top-1  (超越！)
```

这个发现揭示了 ViT 的本质：
- **归纳偏置在小数据时有优势**（CNN 的"先验知识"有帮助）
- **但大规模训练可以弥补归纳偏置的缺失**（ViT 从数据中学习）

"这就像，" Alexey 在论文中写道，"人类专家与新手的区别。新手需要明确的规则（归纳偏置），但专家通过大量经验已经内化了这些规则。"

---

## 第三章：关键概念 - 大量实例

### 概念 1：Patch Embedding 如何工作？

**生活类比 1：拼图游戏**
想象你有一个 1000 片的拼图。要理解整幅图，你可以：
1. 一次看一小块（卷积的局部性）
2. 或者把所有小块摆在一起，同时观察（Attention 的全局性）

ViT 选择的是第二种方式——把图像切成小块，然后同时"观察"所有块之间的关系。

**生活类比 2：像素化艺术**
当你把一张照片像素化（如 16×16 的大像素），你仍然能认出图中是什么。这说明：
- 局部细节不是最重要的
- 整体结构和块之间的关系才是关键

ViT 的 patch 就是这种"像素化"——每个 patch 是一个"超级像素"。

**代码实例 1：Patch Embedding 实现**
```python
import torch
import torch.nn as nn
import einops

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, dim=768):
        super().__init__()
        assert img_size % patch_size == 0

        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.dim = dim

        # 卷积实现 patch 切分 + 线性投影
        # kernel_size=patch_size, stride=patch_size 确保不重叠
        self.proj = nn.Conv2d(
            in_channels, dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        """
        x: (batch, channels, height, width)
        returns: (batch, num_patches, dim)
        """
        x = self.proj(x)  # (batch, dim, h/patch, w/patch)
        x = einops.rearrange(x, 'b d h w -> b (h w) d')
        return x

# 示例
batch_size = 4
img_size = 224
patch_size = 16
dim = 768

patch_emb = PatchEmbedding(img_size, patch_size, 3, dim)
x = torch.randn(batch_size, 3, img_size, img_size)
patches = patch_emb(x)
print(f"输入：{x.shape}")       # (4, 3, 224, 224)
print(f"输出：{patches.shape}")  # (4, 196, 768)
print(f"Patch 数量：{196}")     # (224/16)^2 = 196
```

**代码实例 2：完整 ViT 前向传播**
```python
class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=1000,
                 dim=768, depth=12, heads=12, mlp_dim=3072):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2

        # Patch Embedding
        self.patch_embedding = PatchEmbedding(img_size, patch_size, 3, dim)

        # [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        # 位置嵌入
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=mlp_dim
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # 分类头
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        # Patch Embedding
        x = self.patch_embedding(x)  # (B, N, D)

        # 添加 [CLS] token
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, N+1, D)

        # 位置嵌入
        x = x + self.pos_embedding

        # Transformer (需要 (N, B, D) 格式)
        x = x.transpose(0, 1)  # (N+1, B, D)
        x = self.transformer(x)

        # 取 [CLS] token 输出进行分类
        cls_output = x[0]  # (B, D)
        return self.mlp_head(cls_output)

# 示例
vit = VisionTransformer(img_size=224, patch_size=16, num_classes=1000)
x = torch.randn(4, 3, 224, 224)
output = vit(x)
print(f"输出：{output.shape}")  # (4, 1000)
```

### 概念 2：为什么 ViT 需要大规模数据？

**对比场景：CNN vs ViT 在不同数据量下的表现**

| 预训练数据量 | ResNet-50 | ViT-Base | 差距 |
|-------------|-----------|----------|------|
| ImageNet-1k (128 万) | 76.5% | 70.9% | CNN +5.6% |
| ImageNet-21k (1400 万) | 82.1% | 79.8% | CNN +2.3% |
| JFT-300M (3 亿) | 87.5% | 88.6% | ViT +1.1% |

洞察：
- 小数据时，CNN 的归纳偏置是优势
- 大数据时，ViT 的灵活性变成优势
- ViT 的上限更高（没有人为设计的约束）

**逐步演化实例：从不足到超越**

```
阶段 1：ImageNet 从头训练
- CNN：76.5%（归纳偏置有帮助）
- ViT：70.9%（缺乏归纳偏置，学不到东西）
- 结论：ViT 不如 CNN

阶段 2：ImageNet-21k 预训练 → ImageNet 微调
- CNN：82.1%
- ViT：79.8%
- 结论：ViT 接近但仍不如 CNN

阶段 3：JFT-300M 预训练 → ImageNet 微调
- CNN：87.5%
- ViT：88.6%
- 结论：ViT 超越 CNN！

关键洞察：
- 归纳偏置是"先验知识"，在小数据时有用
- 但大数据可以替代先验知识
- ViT 的上限更高，因为没有人为约束
```

### 概念 3：Patch 大小的影响

**实验对比：不同 Patch 大小的效果**

| Patch 大小 | 序列长度 | ImageNet Top-1 | 训练时间 |
|-----------|---------|----------------|---------|
| 8×8 | 784 | 75.2% | 慢 (4×) |
| 16×16 | 196 | 76.8% | 基准 |
| 32×32 | 49 | 72.1% | 快 (0.5×) |

洞察：
- Patch 太小：计算量爆炸，收益有限
- Patch 太大：丢失细节，性能下降
- 16×16 是最佳平衡点

---

## 第四章：关键实验的细节

### 实验 1：ImageNet 分类主结果

**从头训练（ImageNet-1k）**
| 模型 | Top-1 | Top-5 | 参数量 |
|------|-------|-------|--------|
| ResNet-50 | 76.5% | 93.2% | 25M |
| EfficientNet-B4 | 82.6% | 96.1% | 19M |
| **ViT-Base/16** | **70.9%** | **90.8%** | **86M** |

结论：小数据时 ViT 不如 CNN。

**大规模预训练（JFT-300M → ImageNet）**
| 模型 | Top-1 | Top-5 | 预训练数据 |
|------|-------|-------|-----------|
| BiT-ResNet50 | 87.5% | 98.2% | JFT-300M |
| **ViT-L/16** | **88.6%** | **98.6%** | **JFT-300M** |
| ViT-H/14 | 89.0% | 98.7% | JFT-300M |

结论：大数据时 ViT 超越 CNN。

### 实验 2：迁移学习性能

**CIFAR-100 迁移**
| 模型 | 准确率 | 预训练数据 |
|------|--------|-----------|
| BiT-L | 92.4% | JFT-300M |
| **ViT-L/16** | **92.8%** | **JFT-300M** |

**VTAB（Vision Task Benchmark）**
| 模型 | 自然任务 | 专业任务 | 结构化任务 |
|------|---------|---------|-----------|
| BiT-L | 84.1 | 57.8 | 51.2 |
| **ViT-L/16** | **85.0** | **62.1** | **56.3** |

结论：ViT 在多种下游任务上都有良好的迁移性能。

### 实验 3：消融研究

**位置嵌入类型**
| 位置编码 | ImageNet Top-1 |
|---------|---------------|
| 可学习 1D | 76.8% |
| 可学习 2D | 76.7% |
| 正弦/余弦 | 76.5% |
| 无位置编码 | 75.2% |

结论：位置编码重要，但类型影响不大。

**[CLS] Token vs 全局平均池化**
| 聚合方式 | Top-1 |
|---------|-------|
| [CLS] token | 76.8% |
| 全局平均池化 | 76.5% |

结论：[CLS] 略好，但差异不大。

---

## 第五章：反直觉挑战

**问题 1：ViT 的归纳偏置是什么？**

直觉：ViT 没有归纳偏置。

实际：ViT 有**更少的**归纳偏置，但不是完全没有。

ViT 的归纳偏置：
- Patch 内的局部性（16×16 像素被一起处理）
- 位置嵌入编码了 2D 结构信息
- MLP 层是平移等变的

但 ViT 缺少：
- 跨 patch 的局部性约束
- 明确的平移不变性

**问题 2：为什么 ViT 需要 LayerNorm？**

直觉：LayerNorm 只是稳定训练。

实际：LayerNorm 对 ViT 至关重要，原因有二：

1. **梯度流动**：ViT 很深（12-32 层），没有残差+LayerNorm，梯度会消失。

2. **特征尺度不变性**：LayerNorm 让模型对特征尺度不敏感，这对于处理不同 patch 很重要。

```
ViT 层结构：
x = LayerNorm(x + MultiHeadAttention(x))
x = LayerNorm(x + MLP(x))
```

---

## 第六章：与其他论文的关系

### 上游工作

**ResNet (2015)**
- CNN 的巅峰之作
- 残差连接让训练深层网络成为可能
- ViT 也采用了残差连接

**Transformer (2017)**
- 纯 Attention 架构
- NLP 领域的 SOTA
- ViT 直接应用标准 Transformer

**BERT (2018)**
- [CLS] token 用于分类
- 预训练 + 微调范式
- ViT 完全借鉴了这一范式

### 下游工作

**DeiT (2021)**
- 用知识蒸馏训练 ViT
- 无需大规模数据即可达到 SOTA
- 解决了 ViT 的数据饥饿问题

**Swin Transformer (2021)**
- 层级式 ViT 架构
- 滑动窗口 Attention
- 恢复了一定的局部性，效率更高

**CLIP (2021)**
- 视觉 - 语言联合预训练
- 使用 ViT 作为 vision encoder
- 开创零样本图像分类

**MAE (2021)**
- Masked AutoEncoder 预训练
- 自监督学习 ViT
- 大幅提升表征质量

**Sora (2024)**
- 视频生成模型
- 时空 Patch（3D token）
- ViT 架构的自然延伸

---

## 第七章：如何应用

### 场景 1：使用预训练 ViT

```python
from transformers import ViTForImageClassification, ViTImageProcessor

# 加载预训练模型
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')

# 推理
from PIL import Image
import torch

image = Image.open('cat.jpg').convert('RGB')
inputs = processor(images=image, return_tensors='pt')
outputs = model(**inputs)
logits = outputs.logits

# 获取预测类别
predicted_class = logits.argmax(-1)
```

### 场景 2：在自己的数据集上微调 ViT

```python
from transformers import ViTForImageClassification, TrainingArguments, Trainer

# 加载预训练模型
model = ViTForImageClassification.from_pretrained(
    'google/vit-base-patch16-224',
    num_labels=10,  # 你的类别数
    ignore_mismatched_sizes=True
)

# 训练配置
training_args = TrainingArguments(
    output_dir='./results',
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=10,
    learning_rate=5e-5,
    warmup_ratio=0.1,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
)

# 创建 Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# 开始微调
trainer.train()
```

### 场景 3：选择合适的 ViT 变体

| 场景 | 推荐模型 | 原因 |
|------|---------|------|
| 资源受限 | ViT-Tiny/Small | 参数量小，推理快 |
| 中等数据集 | ViT-Base | 性能与效率平衡 |
| 大规模数据 | ViT-Large/Huge | 充分发挥数据优势 |
| 高分辨率图像 | ViT with small patch (8×8) | 保留更多细节 |

---

## 第八章：延伸思考

1. **ViT 的"归纳偏置缺失"真的是缺点吗？** 还是说这让它能学习到 CNN 学不到的模式？

2. **为什么 16×16 是最佳 Patch 大小？** 这与人类视觉系统有关系吗？

3. **ViT 的 Attention 机制学到了什么？** 是否类似于 CNN 的边缘/纹理检测器？

4. **ViT 能否完全替代 CNN？** 在目标检测、分割等密集预测任务上呢？

5. **自监督学习能否解决 ViT 的数据饥饿问题？** MAE、DINO 等方法的成功说明了什么？

6. **ViT 与 CNN 的融合是未来吗？** ConvNeXt、CvT 等混合架构是否代表新方向？

---

**论文元信息**
- 标题：An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
- 发表会议：ICLR 2021
- 作者：Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby (Google Brain)
- arXiv: 2010.11929
- 阅读时间：2026-03-15
- 方法论：高保真交互式精读协议

---

** Sources:**
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- [ViT: AN IMAGE IS WORTH 16X16 WORDS - Roger Kim](https://kmsrogerkim.github.io/ai/vit/)
- [Paper Review: An Image is Worth 16x16 Words - Medium](https://medium.com/@EleventhHourEnthusiast/an-image-is-worth-16x16-words-transformers-for-image-recognition-at-scale-d5a9ad816a80)
- [Liner Review: ViT Paper](https://liner.com/review/an-image-is-worth-16x16-words-transformers-for-image-recognition)
- [Hugging Face Papers: ViT](https://huggingface.co/papers/2010.11929)
