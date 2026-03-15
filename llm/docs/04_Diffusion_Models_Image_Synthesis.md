# Diffusion Models: 当去噪扩散超越 GAN——图像生成的新范式

## 层 1: 电梯演讲

**一句话概括**：针对 GAN 在图像生成质量上的统治地位，OpenAI 改进了 Diffusion 模型的架构并引入 Classifier Guidance，在 ImageNet 上实现 FID 2.97（128×128）和 4.59（256×256），首次超越 BigGAN，开创扩散模型统治图像生成的时代。

---

## 层 2: 故事摘要 (5 分钟读完)

**核心问题**：2021 年，GAN 是图像生成的绝对王者，能生成高质量图像但训练不稳定、模式坍塌。Diffusion 模型生成质量更好、训练稳定，但图像质量不如 GAN——尤其在高分辨率 ImageNet 上。

**关键洞察**：OpenAI 团队（Prafulla Dhariwal, Alex Nichol）发现扩散模型与 GAN 的差距来自两点：（1）模型架构不够优；（2）GAN 可以用多样性换取质量，扩散模型不行。如何改进？

**解决方案**：（1）通过大量消融实验找到更好的 U-Net 架构（更大、更多 attention、variable width）；（2）提出 **Classifier Guidance**——用分类器梯度引导扩散过程，在多样性和质量之间权衡。

**验证结果**：
- ImageNet 128×128：FID 2.97（超越 BigGAN）
- ImageNet 256×256：FID 4.59（超越 BigGAN）
- ImageNet 512×512：FID 7.72（SOTA）
- 仅需 25 步采样即可匹配 BigGAN 质量

---

## 框架大图

```
┌─────────────────────────────────────────────────────────────────┐
│                        问题空间                                  │
│  ┌──────────────────────────────────────────────────────┐       │
│  │  GAN 的统治地位 (2021)                                │       │
│  │  - BigGAN/StyleGAN 主导图像生成                      │       │
│  │  - 高质量但训练不稳定、模式坍塌                       │       │
│  │  - Diffusion 质量不够好                              │       │
│  └──────────────────────┬───────────────────────────────┘       │
│                         │                                       │
│         ┌───────────────▼───────────────┐                       │
│         │  核心问题                      │                       │
│         │  "Diffusion 能否超越 GAN？"    │                       │
│         │  架构改进 + Classifier Guidance│                       │
│         └───────────────┬───────────────┘                       │
└─────────────────────────┼───────────────────────────────────────┘
                          │
                          ▼
              ┌─────────────────────────┐
              │   Diffusion 核心思想     │
              │                         │
              │  前向：逐渐加噪          │
              │  x0 → x1 → ... → xT     │
              │  (清晰 → 纯噪声)         │
              │                         │
              │  反向：学习去噪          │
              │  xT → ... → x1 → x0     │
              │  (噪声 → 清晰图像)       │
              │                         │
              │  改进：                 │
              │  - 更好的 U-Net 架构      │
              │  - Classifier Guidance  │
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │        验证层            │
              │  ImageNet 128: FID 2.97│
              │  ImageNet 256: FID 4.59│
              │  ImageNet 512: FID 7.72│
              │                         │
              │  关键优势：             │
              │  - 质量更高              │
              │  - 覆盖更好              │
              │  - 训练稳定              │
              └─────────────────────────┘
```

---

## 预期 vs 实际对比表

| 维度 | 预期假设 | 实际发现 | 洞察 |
|------|----------|----------|------|
| **Diffusion vs GAN** | GAN 质量应该更好 | Diffusion 质量超越 GAN | 架构改进 + guidance 是关键 |
| **采样步数** | Diffusion 需要 1000+ 步 | 25 步即可匹配 GAN | 改进采样策略大幅加速 |
| **多样性 - 质量权衡** | 只能牺牲质量保多样性 | 可用多样性换质量 | Classifier Guidance 实现可控权衡 |
| **训练稳定性** | Diffusion 应该更稳定 | 确实更稳定 | 扩散模型无需对抗训练 |

---

## 论文定位图谱

```
                    上游工作 (它解决了谁的问题)
                    ┌─────────────────────────┐
                    │   GAN / BigGAN          │
                    │  - 对抗生成网络          │
                    │  - 训练不稳定、模式坍塌  │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │   DDPM (2020)           │
                    │  - Denoising Diffusion  │
                    │  - 1000 步去噪           │
                    └───────────┬─────────────┘
                                │
          ┌─────────────────────┼─────────────────────┐
          │                     ▼                     │
          │            ┌─────────────────┐            │
          │            │  Guided         │            │
          │            │  Diffusion      │            │
          │            │  (2021) 本研究   │            │
          │            └────────┬────────┘            │
          │                     │                     │
          ┼─────────────────────┼─────────────────────┼
    并行工作                     │              并行工作
┌──────────────────┐            │        ┌──────────────────┐
│  DDIM            │            │        │  Score-based     │
│  确定性采样       │            │        │  分数匹配        │
└──────────────────┘            │        └──────────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │   DALL·E 2 (2022)       │
                    │  - 级联 Diffusion       │
                    │  - 文本→图像生成         │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │   Stable Diffusion      │
                    │  - Latent Diffusion     │
                    │  - 在隐空间扩散          │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │   Imagen / GLIDE        │
                    │  - 纯 Diffusion 文生图   │
                    │  - 超越 GAN 的生成质量    │
                    └───────────┬─────────────┘
                                │
                    ┌───────────▼─────────────┐
                    │   Sora (2024)           │
                    │  - 视频 Diffusion       │
                    │  - 时空 Patch           │
                    └─────────────────────────┘

         下游工作 (谁解决了它的问题/扩展了它)
```

---

## 开场：一个"反直觉"的发现

2021 年初，OpenAI 的办公室里，Prafulla Dhariwal 和 Alex Nichol 看着一组实验数据，感到困惑。

"这说不通，" Prafulla 说，"Diffusion 模型理论上应该比 GAN 更好——训练稳定、覆盖整个分布、没有模式坍塌。但为什么生成的图像质量就是不如 GAN？"

当时是 2021 年，GAN（特别是 BigGAN 和 StyleGAN）是图像生成的绝对王者。它们能生成逼真的人脸、动物、场景——质量高到难以分辨真假。

但 GAN 有致命缺陷：
- **训练不稳定**：生成器和判别器需要精细平衡
- **模式坍塌**：生成样本缺乏多样性
- **调参困难**：超参数稍有不当就崩溃

Diffusion 模型恰好相反：
- **训练稳定**：单一优化目标（去噪）
- **覆盖好**：能生成多样化的样本
- **易于扩展**：更大的模型通常更好

但问题只有一个：**图像质量不够好**。

"也许，" Alex 慢慢地说，"问题不在 Diffusion 本身，而在我们使用的架构？"

这个直觉将引领他们做出一项改变图像生成领域的研究。

---

## 第一章：GAN 的困境与 Diffusion 的崛起

### GAN 的"阿喀琉斯之踵"

2021 年的图像生成领域，GAN 是无可争议的王者。

**BigGAN**（2018）在 ImageNet 上的表现：
```
ImageNet 128×128: FID ≈ 6.9
ImageNet 256×256: FID ≈ 9.6
```

**StyleGAN2**（2019）在 FFHQ 人脸数据集上：
```
FFHQ 1024×1024: FID ≈ 2.8
```

这些数字看起来很小（FID 越小越好），质量确实惊人。但 GAN 的训练过程如同一场"走钢丝"的表演：

```python
# GAN 训练循环（简化）
for real_images in dataloader:
    # 训练判别器
    fake_images = generator(noise)
    d_loss = -log(D(real)) - log(1 - D(fake))
    D.zero_grad()
    d_loss.backward()
    D.step()

    # 训练生成器
    fake_images = generator(noise)
    g_loss = -log(D(fake))
    G.zero_grad()
    g_loss.backward()
    G.step()
```

问题在于：
- G 和 D 必须同步训练——一个太强另一个就学不到东西
- 梯度可能消失或爆炸
- 模式坍塌：G 学会生成"安全"的样本，多样性丢失

### Diffusion 模型的崛起

Diffusion 模型的思想源自 2015 年，但直到 2020 年 DDPM（Denoising Diffusion Probabilistic Models）才真正引起关注。

**Diffusion 的核心思想**：

```
前向过程（固定）：
x0 → 加噪 → x1 → 加噪 → ... → xT (纯噪声)

反向过程（学习）：
xT → 去噪 → x{T-1} → ... → x1 → 去噪 → x0
```

**训练目标**：学习预测噪声
```python
# DDPM 训练（简化）
for image in dataloader:
    # 随机采样时间步 t
    t = random.randint(1, T)

    # 添加噪声
    noise = torch.randn_like(image)
    noisy_image = sqrt(alpha_bar[t]) * image + sqrt(1 - alpha_bar[t]) * noise

    # 预测噪声
    predicted_noise = model(noisy_image, t)

    # 损失：预测噪声 vs 真实噪声
    loss = MSE(predicted_noise, noise)
    loss.backward()
    model.step()
```

DDPM 的优势：
- **单一目标**：无需对抗平衡
- **稳定训练**：标准回归问题
- **理论优美**：与分数匹配、退火朗之万动力学等价

但 DDPM 有一个问题：**生成质量不如 GAN**。

---

## 第二章：试错的旅程

### 第一阶段：架构改进的重要性

团队开始系统分析 Diffusion 与 GAN 的差距。

**假设 1：Diffusion 需要更多训练数据？**
```
实验：在 ImageNet 上训练
结果：增加数据量帮助有限
结论：不是数据问题
```

**假设 2：Diffusion 需要更大模型？**
```
实验：从 50M 参数增加到 500M
结果：有提升，但仍不如 GAN
结论：模型大小不是唯一因素
```

**假设 3：架构不够好？**
```
实验：系统消融 U-Net 架构
结果：显著提升！
结论：架构是关键
```

### 第二阶段：U-Net 架构的系统消融

DDPM 使用的 U-Net 架构来自 Ho et al. (2020)，但这是最优的吗？

团队进行了大量消融实验：

**消融 1：Attention 的作用**
```
基准 U-Net：无 Attention
+ Attention (32×32): FID 7.89 → 6.82
+ Attention (16×16): FID 6.75
+ Attention (所有分辨率): FID 6.52

结论：Attention 显著提升质量
```

**消融 2：ResNet 块数量**
```
基准：每分辨率 2 个 ResNet 块
×2 块：FID 6.52 → 5.72
×3 块：FID 5.58
×4 块：FID 5.45（提升饱和）

结论：更多块有帮助，但有收益递减
```

**消融 3：通道数**
```
基准：128 通道
×1.5 通道：FID 5.45 → 4.98
×2 通道：FID 4.72
×2.5 通道：FID 4.65

结论：更宽模型更好
```

**消融 4：Variable Width**
```
想法：不同分辨率使用不同通道数
低分辨率：更多通道（需要更多计算）
高分辨率：更少通道（序列更长）

结果：FID 4.65 → 4.45

结论：Variable Width 有效
```

### 第三阶段：Classifier Guidance 的突破

架构改进后，Diffusion 质量接近 GAN，但仍有差距。

团队注意到 GAN 的一个关键技巧：**用类别信息换取质量**。

BigGAN 使用：
- 类别条件归一化
- 类别条件判别器

这让它能生成更高质量的特定类别图像——但代价是多样性降低。

Diffusion 能否做到类似的事？

**想法：用分类器梯度引导扩散**

```python
# Classifier Guidance 采样
for t in reversed(range(T)):
    # 预测噪声
    noise_pred = model(x_t, t)

    # 分类器预测
    x_0_pred = denoise(x_t, noise_pred)
    class_logits = classifier(x_0_pred)
    class_loss = -log(p(y | x_0_pred))  # y 是目标类别

    # 分类器梯度
    grad = gradient(class_loss, x_t)

    # 引导采样
    x_t = x_t - step_size * grad
    x_{t-1} = sample(x_t, noise_pred)
```

关键洞察：
- 分类器告诉模型"往哪个方向走更像目标类别"
- 梯度引导采样过程
- 用尺度参数 s 控制引导强度

```
无引导 (s=0): 多样性高，质量低
中等引导 (s=1): 平衡
强引导 (s=5): 质量高，多样性低
```

### 第四阶段：最终架构

结合所有改进，团队得到最终架构：

**Improved U-Net**：
```
- 输入/输出：128/256/512 分辨率
- 通道数：192（base）到 512（large）
- Attention：在 32×32, 16×16, 8×8 分辨率
- ResNet 块：3 个/分辨率
- Variable width：低分辨率更多通道
- 大时间步嵌入：1280 维
```

**训练配置**：
```
- 250k 训练步
- Adam 优化器
- 学习率 1e-4
- Batch size 1024
- 训练时间：约 2 周（256 GPU）
```

---

## 第三章：关键概念 - 大量实例

### 概念 1：Diffusion 如何工作？

**生活类比 1：倒带录像**
想象你有一段录像，内容是墨水滴入水中扩散的过程。

正向扩散：墨水扩散 → 水变浑浊
反向扩散：学习"倒带"→ 从浑浊水中"恢复"墨水

Diffusion 模型做的就是类似的事：
- 正向：图像 → 加噪声 → 纯噪声
- 反向：纯噪声 → 去噪 → 图像

**生活类比 2：雕刻**
想象雕刻家从一块大理石开始：
1. 最初是粗糙的石块（纯噪声）
2. 每次敲掉一些多余的部分（去噪）
3. 最终成为精美的雕塑（清晰图像）

Diffusion 的采样过程就像雕刻——从噪声中"雕刻"出图像。

**代码实例 1：DDPM 前向过程**
```python
import torch
import torch.nn as nn

class GaussianDiffusion:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02):
        self.T = T
        # 线性噪声调度
        self.betas = torch.linspace(beta_start, beta_end, T)
        self.alphas = 1 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x0, t, noise=None):
        """
        前向扩散：q(x_t | x_0)
        x0: 原始图像 (B, C, H, W)
        t: 时间步 (B,)
        """
        if noise is None:
            noise = torch.randn_like(x0)

        # 使用闭式形式采样任意时间步
        sqrt_alpha_bar = torch.sqrt(self.alpha_bar[t]).view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar[t]).view(-1, 1, 1, 1)

        return sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise
```

**代码实例 2：DDPM 反向采样**
```python
    @torch.no_grad()
    def p_sample(self, model, x_t, t):
        """
        反向采样：p(x_{t-1} | x_t)
        """
        # 预测噪声
        noise_pred = model(x_t, t)

        # 计算均值
        alpha = self.alphas[t]
        alpha_bar = self.alpha_bar[t]
        alpha_bar_prev = self.alpha_bar[t - 1] if t > 0 else 1

        # 从 x_t 重建 x_0
        x0_pred = (x_t - torch.sqrt(1 - alpha_bar) * noise_pred) / torch.sqrt(alpha_bar)
        x0_pred = x0_pred.clamp(-1, 1)

        # 计算 x_{t-1} 的均值
        sigma = torch.sqrt(1 - alpha_bar_prev) / torch.sqrt(1 - alpha) * torch.sqrt(1 - alpha)
        mu = torch.sqrt(alpha_bar_prev) * alpha / torch.sqrt(1 - alpha) * x0_pred + \
             sigma * noise_pred

        # 添加噪声（t=0 时不加）
        if t > 0:
            noise = torch.randn_like(x_t)
        else:
            noise = 0

        return mu + sigma * noise, x0_pred

    @torch.no_grad()
    def sample(self, model, shape):
        """
        从噪声开始生成图像
        """
        device = next(model.parameters()).device
        x_t = torch.randn(shape, device=device)

        for t in reversed(range(self.T)):
            x_t, x0_pred = self.p_sample(model, x_t, torch.full((shape[0],), t, device=device))

        return x0_pred
```

### 概念 2：Classifier Guidance 如何工作？

**直观理解**

想象你在迷雾中行走（采样过程）：
- 没有引导：随机游走，可能走到任何地方
- 有引导：有人告诉你"目标在那个方向"

Classifier Guidance 就是那个"指路人"。

**代码实例 3：Classifier Guidance 实现**
```python
class ClassifierGuidance:
    def __init__(self, diffusion_model, classifier, guidance_scale=1.0):
        self.diffusion = diffusion_model
        self.classifier = classifier
        self.guidance_scale = guidance_scale

    @torch.no_grad()
    def sample(self, x_t, t, class_label):
        """
        带分类器引导的采样
        """
        x_t = x_t.requires_grad_(True)

        # 预测噪声
        noise_pred = self.diffusion(x_t, t)

        # 从噪声图像重建清晰图像
        alpha_bar = self.diffusion.alpha_bar[t]
        x0_pred = (x_t - torch.sqrt(1 - alpha_bar) * noise_pred) / torch.sqrt(alpha_bar)

        # 分类器预测
        class_logits = self.classifier(x0_pred)
        class_loss = nn.functional.cross_entropy(class_logits, class_label)

        # 分类器梯度
        grad = torch.autograd.grad(class_loss, x_t)[0]

        # 引导：修改噪声预测
        guided_noise_pred = noise_pred - self.guidance_scale * grad

        # 使用引导后的噪声预测进行采样
        # ...（与 p_sample 类似）

        return x_{t-1}
```

**对比场景：不同引导强度**

| 引导强度 | FID | Inception Score | 多样性 |
|---------|-----|-----------------|--------|
| 无引导 (s=0) | 4.59 | 200 | 高 |
| 中等 (s=1) | 3.94 | 250 | 中 |
| 强引导 (s=5) | 3.52 | 280 | 低 |

洞察：
- 引导强度是质量和多样性的"旋钮"
- 可以根据应用需求调整

### 概念 3：为什么 Diffusion 需要这么多步？

**问题**：DDPM 需要 1000 步采样，而 GAN 只需 1 次前向传播。

**直观理解**：
```
DDPM: x_T → x_{999} → x_{998} → ... → x_1 → x_0  (1000 步)
GAN:  z → G(z) = x  (1 步)
```

**改进方法 1：DDIM（确定性采样）**
```python
# DDIM 采样公式
x_{t-1} = sqrt(alpha_{t-1}) * x0_pred + sqrt(1 - alpha_{t-1}) * noise_pred
```
- 跳步采样：每隔 K 步采样一次
- 1000 步 → 50 步 → 25 步
- 质量几乎不变

**改进方法 2：知识蒸馏**
```
教师模型：1000 步
学生模型：学习 1 步/4 步/8 步生成
```

**逐步演化实例**

```
版本 1：DDPM (1000 步)
- FID: 4.59
- 采样时间：~20 秒
- 问题：太慢

版本 2：DDIM (100 步)
- FID: 4.65
- 采样时间：~2 秒
- 质量略有下降

版本 3：DDIM (25 步)
- FID: 4.85
- 采样时间：~0.5 秒
- 可接受的质量损失

版本 4：Knowledge Distillation (4 步)
- FID: 5.20
- 采样时间：~0.1 秒
- 实用
```

---

## 第四章：关键实验的细节

### 实验 1：无条件 ImageNet 生成

**主结果（FID，越小越好）**
| 模型 | 128×128 | 256×256 | 512×512 |
|------|---------|---------|---------|
| BigGAN-deep | 6.95 | 9.62 | - |
| VQ-VAE-2 | 31.11 | 29.93 | - |
| DDPM | 9.36 | - | - |
| **Guided Diffusion** | **2.97** | **4.59** | **7.72** |

关键洞察：Diffusion 全面超越 GAN。

### 实验 2：Classifier Guidance 效果

**不同引导强度**
| 引导尺度 s | FID (256×256) | Precision | Recall |
|-----------|---------------|-----------|--------|
| 0.0 | 4.59 | 0.72 | 0.63 |
| 1.0 | 3.94 | 0.78 | 0.59 |
| 2.5 | 3.65 | 0.82 | 0.52 |
| 5.0 | 3.52 | 0.85 | 0.45 |

洞察：
- Precision（质量）随 s 提升
- Recall（多样性）随 s 下降
- 可根据需求权衡

### 实验 3：架构消融

**Attention 位置**
| Attention 位置 | FID |
|---------------|-----|
| 无 | 7.89 |
| 32×32 | 6.82 |
| 32×32, 16×16 | 6.52 |
| 32×32, 16×16, 8×8 | 6.45 |

**通道数乘数**
| 乘数 | 参数量 | FID |
|------|--------|-----|
| 1.0× | 103M | 5.45 |
| 1.5× | 232M | 4.98 |
| 2.0× | 412M | 4.72 |
| 2.5× | 643M | 4.65 |

### 实验 4：级联 Upsampling

**两级级联（64→256）**
```
64×64 Diffusion → 256×256 Upsampling Diffusion
```
| 方法 | FID (256×256) |
|------|---------------|
| 直接生成 | 4.59 |
| 级联 + 引导 | 3.94 |

**三级级联（64→256→512）**
| 方法 | FID (512×512) |
|------|---------------|
| 直接生成 | 7.72 |
| 级联 + 引导 | 3.85 |

---

## 第五章：反直觉挑战

**问题 1：为什么 Diffusion 训练比 GAN 稳定？**

直觉：都是深度生成模型，应该差不多。

实际：Diffusion 的训练目标本质上是回归问题。

```
GAN:  min_G max_D L(G, D)  # 对抗博弈，可能不收敛
Diffusion: min_θ E[||ε - ε_θ(x_t, t)||²]  # 简单 MSE 回归
```

Diffusion 没有：
- 生成器 - 判别器对抗
- 梯度消失/爆炸问题
- 模式坍塌

**问题 2：为什么 Classifier Guidance 有效？**

直觉：分类器是在清晰图像上训练的，怎么能指导噪声图像？

实际：Diffusion 的去噪过程可以看作"从噪声中重建清晰图像"。

```
在时间步 t：
- 模型预测噪声 ε_θ(x_t)
- 可以重建 x_0 = (x_t - noise) / sqrt(α_bar)
- 分类器在 x_0 上计算梯度
- 梯度告诉模型"如何修改 x_t 让 x_0 更像目标类别"
```

**问题 3：为什么需要 Variable Width？**

直觉：统一通道数更简单。

实际：不同分辨率的计算成本不同。

```
低分辨率（如 8×8）：
- 序列短，计算便宜
- 可以增加通道数

高分辨率（如 256×256）：
- 序列长（65536 个像素）
- 需要减少通道数
```

Variable Width 在计算预算内最大化容量。

---

## 第六章：与其他论文的关系

### 上游工作

**GAN / BigGAN (2018)**
- 图像生成的 SOTA
- 对抗训练范式
- Diffusion 要超越的目标

**DDPM (2020)**
- 让 Diffusion 重新受到关注
- 1000 步去噪过程
- 本工作的基础

**Score-based Models (2020)**
- 分数匹配视角
- 与 Diffusion 等价
- 理论贡献

### 下游工作

**DALL·E 2 (2022)**
- 使用级联 Diffusion
- 文本→图像生成
- 本工作的直接应用

**Stable Diffusion (2022)**
- Latent Diffusion
- 在压缩空间扩散
- 大幅降低计算成本

**Imagen (2022)**
- 纯 Diffusion 文生图
- 超越 DALL·E 2

**Sora (2024)**
- 视频 Diffusion
- 时空 Patch
- Diffusion 的自然延伸

---

## 第七章：如何应用

### 场景 1：使用预训练 Diffusion 模型

```python
from diffusers import StableDiffusionPipeline

# 加载模型
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

# 生成图像
prompt = "A photo of a cat"
image = pipe(prompt).images[0]
image.save("cat.png")
```

### 场景 2：训练自己的 Diffusion 模型

```python
from diffusers import UNet2DModel, DDPMScheduler, DDPMPipeline

# 定义 U-Net
model = UNet2DModel(
    sample_size=64,
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    block_out_channels=(128, 128, 256, 256, 512, 512),
    down_block_types=(
        "DownBlock2D", "DownBlock2D", "DownBlock2D",
        "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"
    ),
    up_block_types=(
        "UpBlock2D", "AttnUpBlock2D", "UpBlock2D",
        "UpBlock2D", "UpBlock2D", "UpBlock2D"
    ),
)

# 噪声调度器
noise_scheduler = DDPMScheduler(num_train_timesteps=1000)

# 训练
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

for image in dataloader:
    # 采样时间步
    t = torch.randint(0, 1000, (image.size(0),), device=image.device)

    # 添加噪声
    noise = torch.randn_like(image)
    noisy_image = noise_scheduler.add_noise(image, noise, t)

    # 预测噪声
    noise_pred = model(noisy_image, t).sample

    # 计算损失
    loss = nn.functional.mse_loss(noise_pred, noise)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### 场景 3：Classifier Guidance 采样

```python
def guided_sample(model, classifier, class_label, steps=25, guidance_scale=1.0):
    """带分类器引导的 DDIM 采样"""
    model.eval()
    classifier.eval()

    # 从噪声开始
    x = torch.randn(1, 3, 256, 256).cuda()

    for t in reversed(range(1000)):
        x.requires_grad_(True)

        # 预测噪声
        noise_pred = model(x, torch.tensor([t]).cuda()).sample

        # 重建清晰图像
        alpha_bar = scheduler.alphas_cumprod[t]
        x0_pred = (x - torch.sqrt(1 - alpha_bar) * noise_pred) / torch.sqrt(alpha_bar)

        # 分类器梯度
        class_logits = classifier(x0_pred)
        class_loss = nn.functional.cross_entropy(class_logits, class_label)
        grad = torch.autograd.grad(class_loss, x)[0]

        # 引导
        guided_noise = noise_pred - guidance_scale * grad

        # DDIM 更新
        x = ddim_step(x, guided_noise, t)

    return x
```

---

## 第八章：延伸思考

1. **Diffusion 是否真的"理解"图像？** 还是只是在统计上拟合分布？

2. **1000 步采样是必须的吗？** DDIM、蒸馏等方法能否进一步加速？

3. **Classifier Guidance 与 Classifier-free Guidance 哪个更好？** 后者不需要额外训练分类器。

4. **Diffusion 能否用于视频生成？** Sora 给出了肯定的答案——但如何建模时序依赖？

5. **Latent Diffusion vs Pixel Diffusion：哪个更优？** Stable Diffusion 选择了前者，但是否有信息损失？

6. **Diffusion 的终极限制是什么？** 物理世界模拟？通用世界模型？

---

**论文元信息**
- 标题：Diffusion Models Beat GANs on Image Synthesis
- 发表会议：NeurIPS 2021
- 作者：Prafulla Dhariwal, Alex Nichol (OpenAI)
- arXiv: 2105.05233
- 阅读时间：2026-03-15
- 方法论：高保真交互式精读协议

---

** Sources:**
- [Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233)
- [Papers with Code: Diffusion Models Beat GANs](https://paperswithcode.com/paper/diffusion-models-beat-gans-on-image-synthesis)
- [NeurIPS 2021 Paper](https://proceedings.nips.cc/paper/2021/file/49ad23d1ec9fa4bd8d77d02681df5cfa-Paper.pdf)
- [OpenReview: Diffusion Models Beat GANs](https://openreview.net/forum?id=AAWuCvzaVt)
