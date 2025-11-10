## FeatureExtractor
```python
resnet50 = models.resnet50(pretrained=True)
    self.features = nn.Sequential(
        resnet50.conv1, resnet50.bn1, nn.ReLU(inplace=False), resnet50.maxpool
    )
```

用 ResNet50 的 stem（7×7 conv, bn, relu, maxpool），输出通道 64，空间尺寸变为 H/4×W/4

## GradientVarianceFocusHead
核心思想：清晰区域边缘强、局部方差大。对每张图的特征图做：

1. 用 Sobel 算子求梯度强度：grad = mean(|Gx|+|Gy|, dim=1) 得到 [B,1,h,w]

2. 计算局部均值后求方差：var = mean((x - avgpool(x))^2, dim=1) 得到 [B,1,h,w]

3. 两者各做 sigmoid 归一，拼接成2通道，再过一个小卷积头产生 1通道 logits。


下面把 `GradientVarianceFocusHead` 逐行拆开讲清楚：它在干什么、为什么这么设计、数学意义、张量尺寸、可微性与数值细节，以及可以改进的点。

---

### 这个头的总体作用

给**一张特征图**（来自 backbone 的 `[B,C,H,W]`）做两类**清晰度统计**：

1. **梯度强度**（边缘/细节程度）
2. **局部方差**（纹理/细节起伏）

把这两张统计图拼起来作为 2 通道“证据”，再用一个很轻的卷积小头把证据映射成**单通道的 logits**（注意：是*logits*，不是概率/掩码）。后面会把两幅输入图的 logits 对齐到同分辨率，再做 softmax 得到每像素属于“选图1/选图2”的概率。

---

### 代码逐段解释

#### 初始化部分

```python
self.alpha = nn.Parameter(torch.tensor(0.5))
```

* 当前实现里**没有使用**到 `self.alpha`。它通常用于做可学习的权重（例如后面把 grad 与 var 以 `alpha` 融合），但现在没用到，等同于冗余参数。

```python
self.conv = nn.Sequential(
    nn.Conv2d(2, 8, 3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(8, 1, 3, padding=1),
)
```

* 输入是 2 通道（梯度强度、方差），输出 1 通道 logits。
* 两个 3×3 卷积 + 中间 ReLU，属于**极轻量**的非线性映射，用来学习“怎样把这两类证据组合成一张决策图”。

```python
self.register_buffer('sobel_x', tensor(...).view(1,1,3,3))
self.register_buffer('sobel_y', tensor(...).view(1,1,3,3))
```

* 注册为 **buffer**（不是可训练参数），表示固定的 Sobel 核（x/y 方向梯度）。
* `view(1,1,3,3)` 形状上等同单通道卷积核，后面会为每个输入通道复制一份并用 `groups=C` 做**深度可分离卷积**来逐通道求梯度。

#### forward 计算

```python
grad_x = F.conv2d(
    x, 
    self.sobel_x.repeat(x.size(1),1,1,1),  # 变成 [C,1,3,3]
    groups=x.size(1),                      # 深度卷积: 每个通道各自与一个 [1,1,3,3] 核卷
    padding=1
)
grad_y = F.conv2d( ... 同理 ... )
```

* 输入 `x: [B, C, H, W]`
* 通过 `groups=C`，卷积会把每个通道独立地与对应的 Sobel 核相乘求梯度。
* `padding=1` 确保输出空间尺寸仍是 `[H,W]`。
* 得到 `grad_x, grad_y` 形状仍是 `[B, C, H, W]`。

```python
grad = (grad_x.abs() + grad_y.abs()).mean(dim=1, keepdim=True)  # [B,1,H,W]
```

* 这里用 **L1 梯度幅值近似**：`|Gx| + |Gy|`。
* 然后在 **通道维做均值**（`mean(dim=1)`），得到单通道的**整体梯度强度图**。
* 用通道均值而不是最大/加权，是一种**简单稳定**的聚合，让不同语义/尺度的特征通道贡献平均化。

```python
x_mean = F.avg_pool2d(x, 3, stride=1, padding=1)               # 局部均值
var = ((x - x_mean)**2).mean(dim=1, keepdim=True)               # [B,1,H,W]
```

* 用 3×3、步长 1 的平均池化得到每个通道的**局部均值**，再计算 `(x - x_mean)^2` 并在通道维取均值，得到单通道**局部方差图**。
* **直觉**：清晰区域纹理起伏大、局部方差更高。

```python
grad_n = torch.sigmoid(grad)
var_n  = torch.sigmoid(var)
```

* 把两个统计做 **sigmoid** 归一到 (0,1)（*不是*把它们当最终概率，只是把动态范围“捏”到稳定的区间，方便后续小头学习）。
* 如果不做这步，`grad/var` 的数值范围可能相差很大，卷积小头更难学到合理权重。

```python
stats  = torch.cat([grad_n, var_n], dim=1)  # [B,2,H,W]
logits = self.conv(stats)                   # [B,1,H,W]
return logits
```

* 拼接两种证据，交给小头 `conv(2→8→1)` 去**学习组合关系**（比如：某些区域梯度高但方差低、或相反时该怎么判）。
* 输出是 **logits**（任意实数），后续再与另一张图的 logits 拼在一起，经 softmax 得到真正的**权重/概率**。

---

### 数学直观（简化表述）

* 设特征图为 ($x\in\mathbb{R}^{C\times H\times W}$)。

**梯度强度图（单通道）**

$$G(i,j)=\frac{1}{C}\sum_{c=1}^C \big(|(x_c * S_x)(i,j)| + |(x_c * S_y)(i,j)|\big)$$

其中 (S_x,S_y) 是 Sobel 核，(*) 为卷积。

**局部方差图（单通道）**

$$\mu_c(i,j) = \text{avgpool}*{3\times3}(x_c)(i,j),\qquad$$
$$V(i,j)=\frac{1}{C}\sum*{c=1}^C (x_c(i,j)-\mu_c(i,j))^2$$


**归一并拼接**

$$\hat G=\sigma(G),\quad \hat V=\sigma(V),\quad S=\text{concat}[\hat G,\hat V]$$


**小头映射成 logits**

$$\text{logits} = \phi(S) \in \mathbb{R}^{1\times H\times W}$$

($\phi$) 是两层 3×3 Conv + ReLU 的轻量 CNN。

---

### 为什么输出 logits 而不是直接 mask？

* **数值更稳**：二分类/二选一问题里，先在 logits 空间做线性/加性融合（比如再加上 CLIP 先验的 log 概率），最后再 softmax 成概率，**比直接在概率上做算术**更稳健。
* **易于融入先验**：你主干里把 CLIP 的先验概率取 `log` 后作为 **logits 偏置**加进去，这在 logits 空间是自然操作。
* **可配合任意监督**：针对 logits 用标准的交叉熵/ Dice loss 等都很自然。

---

### 可微性与训练

* 除 Sobel 核（buffer，固定不训练）以外，所有步骤（`conv2d`、`avg_pool2d`、拼接、`conv` 小头）都是**可微**的，梯度能回传给 `conv` 小头以及前面的 backbone（如果没冻结）。
* `sigmoid` 只是一种数值压缩，它的梯度也可正常回传。

---

### 尺寸与复杂度

* 输入 `[B,C,H,W]`，输出 `[B,1,H,W]`。
* 复杂度基本是 **两次 depthwise 3×3 Sobel 卷积 + 一次 3×3 平均池化 + 两个 3×3 标准卷积**，相对 backbone 来说很轻。

---

### 数值/实现细节的合理性

* `groups=C` + `repeat(C,1,3,3)` 是**标准做法**，表示每个通道用自己的 Sobel 卷积核（其实是同一核的复制）。
* `padding=1` 保持分辨率不变，避免边界缩小。
* 对 `grad/var` 做 `sigmoid` 是**归一化**而非“当概率”，别被名字误导。
* `inplace=True` 的 ReLU 在这里一般**没问题**，因为作用在 `self.conv` 内部新生成的张量上，不会破坏外部计算图。

---

### 可以考虑的改进

1. **用上 `alpha`**：
   例如在卷积头前做一层可学加权：
   
   $$S'=\text{concat}[\sigma(G),\ \alpha\cdot\sigma(V)]$$
   
   或者把 `alpha` 放进 `conv` 的第一层权重初始化里。

2. **标准化替代 sigmoid**：
   对 ($\hat G$,$\hat V$) 使用 `BatchNorm2d` / `LayerNorm` 或者 min-max 标准化，可能比 `sigmoid` 更线性、梯度更友好（看数据分布而定）。

3. **多尺度统计**：
   不仅 3×3 的均值/方差，也可以加 5×5、7×7 的并行通道再交给小头融合，捕获不同尺度的清晰度线索。

4. **更强的通道聚合**：
   现在对 C 维直接 `mean`。可以尝试 `1×1` 卷积先把通道压到几维（可学习聚合），再做统计，或者直接在特征域上学一个“清晰度通道投影”。

---


这个模块把**清晰区域边缘更强、纹理起伏更大** 这两个经典直觉，变成了**可微的、可学习组合的单通道 logits 估计器**，为后续的两图对比与 softmax 融合提供了**稳定、物理可解释**的证据。虽然有综述文献说了用 梯度强度 + 方差作为“焦点测度”的学术依据，大师没有直接借鉴别人的东西，具体的实现和别人的不同的，更多是把经典焦点测度（gradient magnitude, local variance）嵌入 CNN 可微头中，还是挺多自己提出的东西，然后也有还可以改进的内容，可以后续加上。


## CLIP使用模块


