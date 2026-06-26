# LORAStencil demo notes

## Core idea

真正的创新点应该表述为：

把 Temporal Vectorization 形成的倾斜 space-time tile 预展开为一个小型线性算子，并通过批量化 frontier/tile 的方式映射到 Tensor Core MMA。

要优先证明的不是“Tensor Core 能不能算 stencil”，LoRAStencil 已经证明了。要证明的是：

Temporal 的倾斜矩形是否能形成足够 dense、足够规则、构造成本足够低的 space-time operator，从而比 step-by-step stencil 更适合 Tensor Core。

## Temporal Vectorization gives the tile shape

Temporal Vectorization 的本质不是单纯把空间连续点塞进 SIMD，而是在 iteration space 里组织计算：一个 vector/tile 覆盖不同时间坐标的点，通过倾斜的 space-time 排布避开依赖，并把中间 frontier 按固定顺序推进。

这个倾斜结构很重要。它天然给出了一个局部依赖锥和一组可批量处理的 frontier：输入不是一个普通空间窗口，输出也不是简单的下一时间步连续区间，而是一个带时间斜率的局部 transition。只要 stencil 是线性、常系数 Jacobi/heat/convolution 类型，这个 transition 可以被预展开成一个固定小矩阵。

## LoRAStencil gives the Tensor Core baseline

LoRAStencil 的贡献是证明 stencil 可以映射到 Tensor Core MMA。它把单步 stencil 的 Frobenius inner product 变换为矩阵乘形式；对 rank-1 权重用 RDG 做 `U X V`，对一般对称权重用 PMA 分解成多个 rank-1 矩阵，再用 BVS 避免 FP64 WMMA fragment 布局导致的数据 shuffle。

因此，“Tensor Core 能不能算 stencil”已经不是我们的主要问题。LoRAStencil 的路线主要围绕单步空间 stencil 的低秩权重和 fragment 访存冗余展开；我们的路线应该围绕多步 temporal tile 的 space-time operator 展开。

## Our research question

我们要验证的是：Temporal 倾斜 tile 预展开后，是否会从稀疏 step-by-step stencil 变成一个更 dense、更规则、可复用的小型线性算子。

关键判断标准：

- Density: `Ttile=8/16/32` 时 operator 的非零密度是否足够高，能否提升 Tensor Core 算术强度。
- Regularity: 算子形状和系数是否固定，能否放在 constant/shared memory 中复用。
- Construction cost: operator 是否可以按 stencil 系数和 `Ttile` 预计算，构造成本能否被大量 tile/frontier 批量摊薄。
- Boundary cost: side boundary、intermediate frontier、halo exchange 是否会吞掉 MMA 获得的收益。
- Comparison target: 应比较 step-by-step CUDA stencil 或 LoRAStencil-style 单步 MMA，而不是只证明 Tensor Core 指令存在。

## Near-term prototype

优先做 1D 3-point Jacobi/heat：

- 从 `Ttile=8/16/32` 开始，不做 2D/3D。
- 先用 FP32/TF32 跑机制，FP64 只做 reference。
- 把一个倾斜 space-time tile 的输入 frontier 映射到输出 frontier，显式生成小矩阵。
- 将很多独立 tile/frontier 按列或 batch 维度打包，形成 MMA 友好的 `D = A * B + C`。
- 对比 step-by-step stencil 的正确性、访存、MMA 利用率和端到端时间。

适用范围要先收窄：线性、常系数、规则网格 stencil。变量系数、非线性更新、复杂边界或 Gauss-Seidel 依赖会显著增加 operator 构造和调度复杂度，不应作为第一阶段目标。
