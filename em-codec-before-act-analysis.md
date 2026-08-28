# 在 ACT 前增加 emEncoder/emDecoder 的方案分析

## 1. 方案结论

该方案可行，而且适合作为当前阶段优先验证的低风险基线。

核心做法是将 `emEncoder + emDecoder` 作为 human hand 点云到 robot hand 关节的前置转换器：

```text
human hand point-cloud history
    -> frozen emEncoder
    -> hand latent(128)
    -> frozen emDecoder
    -> pseudo hand joints(6)
    -> 原双 decoder ACT
    -> predicted arm joints(6) + hand joints(6)
```

ACT 内部仍然处理原来的：

```text
[arm joints(6), hand joints(6)]
```

因此不需要把 ACT 改成 `[arm 6, hand latent 128]`，也不存在 128 维 hand latent 在 ACT 内部淹没 6 维 arm 表示的问题。

这个方案的本质不是让 ACT 学习点云 latent，而是利用 codec 为没有 joint 标注的 human 数据生成 robot hand joint 伪标签。

## 2. 为什么采用该方案

当前双 decoder ACT 的 arm 分支已通过 joint loss 和 FK pose loss 获得一定效果。直接修改 ACT 内部状态维度、CVAE 输入和 hand 输出空间，会同时改变多个共享模块，存在 arm 性能退化风险。

前置 codec 方案把改动限制在数据进入 ACT 之前，因此可以保留：

- 当前 `[arm 6 + hand 6]` 状态和动作定义；
- 现有三个 qpos/action projection；
- CVAE posterior 和推理时零 latent 逻辑；
- shared Transformer encoder；
- arm/hand 双 Transformer decoder；
- arm joint loss 和 FK pose loss；
- hand joint output head；
- action chunk 和机器人执行接口；
- 已训练 checkpoint 的参数形状。

如果 codec 的跨域转换质量足够好，这种方式能够以最小模型改动接入 human 数据。

## 3. 完整训练数据流

### 3.1 当前时刻的 hand 状态

对当前 observation frame `t`：

```text
human_pc[t-W+1:t]
    -> point-cloud preprocessing
    -> emEncoder
    -> last latent
    -> L2 normalization
    -> emDecoder
    -> sin/cos
    -> atan2
    -> radians
    -> Inspire raw joint values(0-1000)
    -> ACT hand qpos normalization
    -> hand_qpos_pseudo(t)
```

然后按当前 ACT 方式组成：

```text
qpos_act(t) = concat(arm_qpos(t), hand_qpos_pseudo(t))
```

### 3.2 未来 hand action chunk

仅转换当前 hand qpos 不足以训练 ACT。对于 human 数据，还必须给 action chunk 中的每个有效时刻生成 pseudo hand joint target：

```text
human_pc[tau_k-W+1:tau_k]
    -> emEncoder
    -> emDecoder
    -> hand_action_pseudo(k)
```

最终训练 target 仍保持原格式：

```text
action_act(k) = concat(
    arm_action(k),
    hand_action_pseudo(k)
)
```

这里的 `tau_k` 必须来自 dataset 实际返回的 action frame index。不能简单假设为 `start_ts + k + 1`，因为当前数据加载和动作构造存在首帧偏移逻辑。

### 3.3 Robot 数据

Robot 数据可以继续使用真实 hand joints：

```text
robot sample -> measured hand qpos/action -> ACT
```

Human 数据使用 codec 生成的 pseudo joints：

```text
human sample -> point cloud -> pseudo hand qpos/action -> ACT
```

两类数据最终都进入统一的 robot joint action space。

## 4. 建议采用离线预处理

如果点云的目的只是补齐 human 训练数据，而机器人部署时仍然可以读取真实 hand joints，建议不要把 codec 永久放进 ACT 在线推理图，而是离线生成 human pseudo joints。

推荐流程：

```text
human point-cloud episodes
    -> offline frozen codec
    -> pseudo hand qpos/action files
    -> 原 ACT dataset
    -> 原 ACT training/inference
```

离线方式的优点：

- ACT 代码和 checkpoint 结构基本不变；
- 不增加机器人在线推理延迟；
- 可人工检查和过滤错误伪标签；
- 同一个点云窗口不会在训练中反复编码；
- 更容易审计时间对齐和未来信息泄漏；
- codec 与 ACT 可以分别评估和迭代。

缓存文件应记录：

- episode/frame index；
- codec checkpoint 标识或 hash；
- point-cloud preprocessing 版本；
- history window size；
- decoded joint unit；
- 置信度或异常标志。

## 5. Codec 使用要求

### 5.1 冻结和推理模式

第一阶段应固定：

```text
emEncoder.requires_grad = False
emDecoder.requires_grad = False
emEncoder.eval()
emDecoder.eval()
```

这可以避免 ACT loss 改写已经训练好的 latent 空间，也能关闭 dropout，保证相同点云得到稳定结果。

### 5.2 实际 checkpoint 配置

当前配套 checkpoint 的关键配置为：

- Inspire hand；
- 6 个 hand joints；
- 每帧 256 个点；
- latent dimension 为 128；
- emEncoder 训练序列长度为 50；
- emDecoder 的 latent attention window size 为 20；
- decoder 输出每个关节的 sin/cos；
- joint 角度范围为 `[-pi/2, pi/2]`；
- Inspire 原始 joint 范围为 `0–1000`。

实现时应从 checkpoint/config 读取参数，而不是沿用旧 HTML 讨论中的窗口 10 或关节数 10。

### 5.3 单位转换

`emDecoder` 解码后先得到弧度：

```text
theta = atan2(sin_theta, cos_theta)
```

再转换回当前 ACT 使用的 Inspire 原始关节值：

```text
raw_joint = (theta + pi/2) / pi * 1000
```

随后再使用当前 ACT 的 hand joint mean/std 做归一化。不能把 decoder 的弧度直接当作 ACT 当前的 joint 值。

## 6. 最关键的成立条件

该方案成立的关键不是 codec 在 robot 数据上的重建结果好，而是它能把 human 点云稳定地转换到 ACT 所使用的 robot hand joint 空间。

需要确认：

- human 与 robot 点云采用兼容的坐标系和尺度；
- human 点云能进入 emEncoder 的训练分布；
- decoded joints 与 robot hand 的关节定义、方向和范围一致；
- 相同手势在 human/robot 域中产生接近的 joint 表达；
- decoded pseudo joints 在时间上连续；
- 接触、遮挡和点云缺失时不会产生严重跳变。

如果 codec 只在 robot 点云上训练和验证过，不能直接认为它在 human 点云上同样有效。必须单独完成 human-to-robot 跨域验证。

## 7. History-only 与未来信息泄漏

当前 `emEncoder` 的 temporal Transformer 没有 causal mask，输入窗口内部使用双向注意力。因此生成在线可用的 pseudo joint 时，每个时刻只能输入截至该时刻的历史点云，并取窗口最后一个 latent：

```text
pc[t-W+1:t] -> emEncoder -> z_last(t)
```

不能把整个 episode 一次输入 encoder，再取中间 latent 作为伪标签，否则这些 latent 可能使用未来帧信息，使 human 训练数据产生不可部署的信息泄漏。

需要区分两个时序长度：实际 checkpoint 的 emEncoder 训练序列长度是 50，而 emDecoder 的 latent attention window 是 20。最贴近训练分布的 history-only 解码方式，是对每个目标时刻输入截至该时刻的 50 帧点云序列，经 encoder 和 decoder 后只取最后一个 joint。

episode 开头不足 50 帧时，训练代码原本会跳过过短窗口，并没有天然提供完全等价的在线填充策略。因此必须在实施前选择并验证以下一种方式：跳过 episode 前 49 帧、复制首帧左填充，或使用逐渐增长的前缀。不同 episode 之间必须清空历史缓存。

## 8. Human 与 Robot 数据的分布差异

Human 数据使用 pseudo joints，robot 数据使用真实 joints，两者可能存在系统性偏差。ACT 可能学习到数据来源而不是统一的动作语义。

建议采取以下措施：

- 在 robot 数据上同时计算 codec decoded joints 和 measured joints，统计 codec bias；
- 对 robot joint 输入加入与 codec 误差相近的小幅噪声，降低输入分布差异；
- 对 human pseudo joints 使用置信度过滤或较低 loss/sample 权重；
- 分别记录 human pseudo-joint loss 与 robot real-joint loss；
- 检查模型是否能仅根据 hand joint 统计区分 human/robot 样本；
- 对明显越界、突变或低质量点云对应的伪标签直接过滤。

如果 robot 侧也能获得同类型点云，可增加一组消融：robot 数据同样经过 codec 后进入 ACT。这样输入分布更统一，但会放弃一部分真实 joint 精度，因此不一定是最终方案。

## 9. 优点

- 对 ACT 主体改动最小，最有利于保护 arm 分支和现有 FK 效果。
- 不会引入 128 维 hand latent 与 6 维 arm joint 的表示失衡。
- 不需要新增 latent adapter、hand latent head 或 latent loss。
- 可以直接复用当前双 decoder checkpoint 和训练逻辑。
- Human 与 robot 数据最终都落到同一种 robot joint action space。
- Codec 可以离线运行，易于检查、缓存和替换。
- 若效果不理想，能够明确判断问题来自 codec、跨域映射还是 ACT，而不是多个新模块共同变化。

## 10. 局限和风险

- Codec 解码误差会成为 human 数据的伪标签噪声。
- 系统性伪标签偏差可能被 ACT 学习并放大。
- 128 维 latent 中超出 6 个关节表达能力的几何信息会在 ACT 前丢失。
- 点云中的接触形状、物体约束等信息无法直接提供给 ACT。
- 如果 human 点云与 codec 训练域不同，decoded joints 可能看似平滑但语义错误。
- 为 future action chunk 生成伪标签时，计算量和数据缓存量会上升。
- 该方案解决的是“缺少 hand joints”问题，不等同于让策略直接利用点云几何。

## 11. 推荐实验步骤

### Phase 0：固定当前 ACT 基线

- 固定数据划分、随机种子和 checkpoint；
- 记录 arm joint MAE、FK error、hand joint MAE 和任务成功率；
- 保留一组固定 validation episodes。

### Phase 1：独立验证 codec

- 使用长度 50、以目标时刻结尾的 history-only encoder 窗口；
- 保持 emDecoder 的 latent attention window 为 checkpoint 中的 20；
- 验证 robot 点云重建；
- 验证 human 点云跨域解码；
- 统计每关节 MAE、时间连续性、越界率和失败样本。

Codec 不通过时不进入 ACT 训练。

### Phase 2：生成 human pseudo joints

- 为 observation frame 生成 pseudo hand qpos；
- 为每个 action frame 生成 pseudo hand action；
- 保存 frame index、有效 mask 和置信度；
- 抽样可视化或人工检查完整 episode 的动作曲线。

### Phase 3：保持 ACT 不变进行训练

- Robot 样本使用真实 hand joints；
- Human 样本使用 pseudo hand joints；
- arm 和 FK loss 保持当前逻辑；
- 分别记录两种数据来源上的 loss；
- 首轮对 human hand loss 或 sample weight 使用较保守权重。

### Phase 4：评估是否需要内部 latent 方案

如果 arm 指标稳定、hand 与任务成功率达到目标，则无需立即修改 ACT 内部表示。

只有出现以下情况时，再考虑 `[arm 6 + hand latent 128]`：

- pseudo-joint 重建形成明显性能上限；
- 任务需要 6 个关节无法表达的手部几何信息；
- latent 在 human/robot 间明显比 decoded joints 更一致；
- 希望 hand 输出受到 codec 运动流形的显式约束。

## 12. 验收指标

### Codec 指标

- decoded joint MAE，分别按 human/robot 数据统计；
- 每个关节的 P50/P90/P95 error；
- 相邻帧速度和加速度异常率；
- joint 越界率；
- 点云缺失情况下的失败率；
- history-only 与 full-sequence 结果差异。

### ACT 指标

- arm joint MAE 相对当前基线的变化；
- FK position/orientation error 相对当前基线的变化；
- robot real-joint hand MAE；
- human pseudo-joint hand MAE；
- 混合训练后的任务成功率；
- 仅 robot 数据与 robot+human 数据的对照结果。

建议预先规定 arm 不退化门槛，例如 arm MAE 和 FK mean error 不超过基线的 5%，最终阈值应根据多随机种子实验方差确定。

## 13. 最终建议

优先将 `emEncoder + emDecoder` 作为离线 human hand 伪关节生成器，而不是立即修改 ACT 内部结构：

```text
human point cloud
    -> frozen history-only codec
    -> filtered/confidence-weighted pseudo hand joints
    -> unchanged dual-decoder ACT
```

该方案最大程度保留当前 arm 分支、FK 监督和 ACT 核心预测逻辑，是验证 human 数据能否有效加入训练的最低风险路径。

第一阶段最重要的工作不是改 ACT，而是证明 codec 在严格 history-only 条件下具有可靠的 human-to-robot joint 映射能力。只有当该方案受到 6 维 pseudo joint 表示的明确限制时，才有必要升级到 ACT 内部 hand latent 方案。

## 14. 实施前 Todo List

以下项目应在修改 ACT 或批量生成 human pseudo joints 之前完成。标记为“阻断项”的任务未通过时，不建议开始 ACT 混合训练。

### A. Checkpoint 与代码一致性

- [ ] **阻断项：**确定使用成套的 codec 权重。优先从 `pointcloud_embedding/ckpts/emDecoder_ckpt.pt` 同时加载其中的 encoder 和 decoder state dict，避免分别加载不完全匹配的 checkpoint。
- [ ] 严格检查 checkpoint 加载结果，确认没有未解释的 missing keys 或 unexpected keys。
- [ ] 固定并记录实际配置：Inspire hand、6 joints、256 points、latent 128、encoder seq_len 50、decoder window 20、validation stride 1。
- [ ] 确认 MultiMimic 中的 `emEncoder.py`、`emDecoder.py` 与训练 checkpoint 使用的模型定义一致。
- [ ] 修正并验证包内导入路径，避免依赖原 MyMimic 工程的 `from model...` 路径。
- [ ] 确认 codec 推理时始终执行 `eval()`，并关闭所有参数梯度。

### B. 严格确定 codec 的时序推理方式

- [ ] **阻断项：**验证推荐的滚动窗口方式：`pc[t-49:t] -> emEncoder -> normalize(z_seq) -> emDecoder -> last joint`。
- [ ] 对比当前 realtime 方式：`pc_t -> emEncoder(seq_len=1) -> latent cache -> emDecoder`，量化它与滚动 50 帧方式的差异。
- [ ] 根据重建质量、真机效果和计算开销，在批量生成伪标签前固定唯一的推理方式。
- [ ] **阻断项：**确认 current qpos 的生成窗口绝不包含 `t+1` 及之后的点云。
- [ ] 禁止把整个 episode 一次送入非因果 emEncoder 后，直接取中间 latent 生成 ACT 的 current qpos。
- [ ] 明确 episode 前 49 帧的处理：跳过、首帧左填充或增长前缀，并对所选方式单独验证。
- [ ] 确认 episode 切换时清空点云、latent 和 decoder cache。

### C. 点云预处理一致性

- [ ] **阻断项：**确认 Human/Robot 点云坐标系、单位、手腕基准和点云生成方式与 codec 训练数据兼容。
- [ ] 每帧使用 256 个点，并固定下采样/上采样策略；验证训练与离线伪标签生成没有不一致的随机采样。
- [ ] 保持与训练代码一致的 sequence-level centroid 和 max-distance normalization。
- [ ] 归一化统计只能由截至目标时刻的 trailing window 计算，不能使用完整 episode 或目标时刻之后的点云。
- [ ] 检查零点云、有效点过少、异常尺度、NaN/Inf 和严重遮挡样本。
- [ ] 为无效点云制定明确策略：过滤样本、降低权重或使用安全回退值。

### D. Codec 单独效果验证

- [ ] **阻断项：**在 held-out Robot 点云上验证 `point cloud -> decoded joints`，并与 measured joints 比较。
- [ ] **阻断项：**在 held-out Human 点云上验证 decoded robot joints，并将关节序列部署到真机确认动作、手型、接触和任务结果符合原点云。
- [ ] Human/Robot 验证覆盖不同动作、速度、物体、episode、遮挡和点云质量，而不只检查训练集中的少量样本。
- [ ] 记录每关节误差或可获得的等价行为指标、P50/P90/P95、越界率、速度/加速度异常率和时序延迟。
- [ ] 检查相同或语义相近的 Human/Robot 动作是否解码到一致的 robot joint 空间。
- [ ] 预先定义 codec 通过门槛；未达到门槛时先修复 codec/domain gap，不进入 ACT 训练。

### E. 关节定义与单位转换

- [ ] **阻断项：**确认 emDecoder 输出的 6 个关节顺序、正方向和 ACT/真机完全一致。
- [ ] 验证 `sincos -> atan2 -> radians -> Inspire raw 0–1000` 的转换公式和范围。
- [ ] 确认 ACT normalization 发生在转换为 raw joint 之后，不能将 radians 直接套用当前 joint mean/std。
- [ ] 检查 decoded joints 在裁剪前后的分布；训练数据不能依赖大量裁剪掩盖 codec 错误。
- [ ] 确认 Human pseudo joints 与 Robot measured joints 的统计偏差、噪声和动态范围。

### F. ACT action 时刻对齐

- [ ] **阻断项：**为每个 ACT 样本显式记录 `obs_frame_index`、`action_frame_indices[K]` 和 `is_pad[K]`。
- [ ] 按 dataset 实际 action index 为每个 `tau_k` 构造 `pc[tau_k-49:tau_k]`，不能仅用 `start_ts + k` 猜测。
- [ ] 核对真实数据中的 `action_start = max(0, start_ts - 1)` 及当前 HDF5 action 语义。
- [ ] 确认 observation pseudo hand qpos 与 RGB、arm qpos 使用同一时刻。
- [ ] 确认 future pseudo hand actions 与 arm action、FK pose target 逐 token 对齐。
- [ ] padding token 不生成或不使用无意义的 pseudo-joint loss。
- [ ] 明确 future point cloud 只用于生成对应 future action target，绝不能进入 current observation 条件路径。

### G. 离线伪标签数据设计

- [ ] 决定采用离线预计算，而不是在 ACT 每个 training step 中重复运行 codec。
- [ ] 保存完整 pseudo-joint episode，再复用当前 ACT qpos/action 切片逻辑，减少两套索引实现不一致的风险。
- [ ] 缓存中保存 episode/frame index、codec checkpoint hash、预处理版本、窗口配置和有效性标志。
- [ ] 为每帧生成置信度或质量标志，并支持过滤、降权和重新生成。
- [ ] 抽样检查完整 episode 的点云、pseudo joints、Robot joints/动作和 ACT action chunk 是否同步。
- [ ] Codec 或预处理版本变化后，使旧缓存自动失效，避免混用不同版本伪标签。

### H. Human/Robot 混合训练策略

- [ ] 固定 Robot 数据继续使用 measured joints，Human 数据使用 pseudo joints 的首版基线。
- [ ] 分别统计 Human pseudo-joint loss 和 Robot real-joint loss，不能只看混合平均值。
- [ ] 根据 codec 误差为 Human 样本设置初始 sample/loss 权重，低置信度样本过滤或降权。
- [ ] 检查 Human pseudo joints 与 Robot real joints 是否存在容易被模型识别的数据源偏差。
- [ ] 评估是否需要给 Robot joints 加入与 codec 误差相近的小幅噪声，或增加 Robot 也经过 codec 的消融实验。
- [ ] 保持 ACT 内部 `[arm 6 + hand 6]`、双 decoder、CVAE、FK loss 和输出接口不变，避免同时引入无关架构改动。

### I. 基线、验收与回滚条件

- [ ] **阻断项：**固定当前 Robot-only 双 decoder ACT checkpoint、数据划分、随机种子和验证集。
- [ ] 记录修改前的 arm joint MAE、FK position/orientation error、hand joint MAE、任务成功率和推理延迟。
- [ ] 预先定义 arm 不退化门槛，例如 arm MAE 与 FK mean error 不超过基线 5%，并结合多随机种子方差最终确定。
- [ ] 设置对照实验：Robot-only 基线、Robot + Human pseudo joints、不同 Human 权重，以及可选的 Robot codec joints。
- [ ] 如果 arm/FK 超过退化门槛，优先停止并检查数据比例、伪标签时序、归一化和 domain bias，而不是立即修改 arm 网络。
- [ ] 只有当前置 codec 方案通过时序、跨域和 ACT 验收后，才决定是否需要更复杂的内部 latent 迁移。

### J. 开始修改 ACT 前的最终 Go/No-Go

- [ ] paired codec checkpoint 可以无异常加载。
- [ ] 滚动 history-only 解码方式已经固定并通过验证。
- [ ] Human 与 Robot 真机动作验证均达到预设门槛。
- [ ] current qpos 与 future action target 的 frame index 已完全对齐。
- [ ] 点云预处理、joint 单位和 episode 边界处理已经锁定。
- [ ] pseudo-joint 缓存格式、版本信息和异常过滤已经确定。
- [ ] 当前 ACT 基线与 arm/FK 不退化门槛已经记录。

只有以上 Go/No-Go 项全部通过，才建议开始修改 dataset 接口并进行 Human/Robot 混合 ACT 训练。
