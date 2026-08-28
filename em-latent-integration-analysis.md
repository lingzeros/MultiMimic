# ACT 双 Decoder 接入 MyMimic 手部点云 Latent 的架构分析

## 1. 结论摘要

当前最稳妥的迁移方式，不是把现有 `hand decoder` 整段替换为 MyMimic，也不建议第一步照搬 MyMimic 中“手部 token 注入共享 memory”的融合方式，而是只替换 ACT 内部对手部状态和手部动作的表示：

- 现状内部状态：`[arm joints(6), hand joints(6)]`。
- 目标内部状态：`[arm joints(6), hand latent(128)]`。
- 手部观测由历史点云窗口经过冻结的 `emEncoder` 得到 `z_hand_obs`。
- 手部监督由与 ACT 动作时刻对齐的未来点云窗口经过冻结的 `emEncoder` 得到 `z_hand_gt`。
- 现有 hand Transformer decoder 保留，只把输出头从关节维度改为 128 维 latent。
- 预测 latent 经归一化后送入冻结的 `emDecoder`，还原为手部关节值。
- arm Transformer decoder、arm 输出头、FK pose 监督及推理主流程保持不变。

这种方案保留了当前模型最有价值的核心逻辑：图像编码、CVAE、共享 Transformer encoder、双 decoder、chunk prediction、arm FK loss。变化被限制在手部输入表示、手部输出表示以及相应的适配层和 loss 上。

为尽量不损伤 arm，建议使用“三层隔离”：

1. **结构隔离**：第一阶段不向共享视觉 memory 新增手部 token；手部 latent 仅通过现有 proprio/action 投影接口进入模型。
2. **参数隔离**：先冻结共享 encoder、arm decoder/head、视觉 backbone、CVAE latent projection 和共享 query。
3. **行为隔离**：保留旧模型作为 teacher，对 arm 输出和三个原有输入 embedding 做蒸馏约束。

## 2. 基于当前仓库代码确认的模型事实

### 2.1 当前双 Decoder 结构

当前 `detr/models/detr_vae.py` 中：

- `arm_dim = 6`。
- `hand_dim = state_dim - 6`。
- 图像特征经过共享 Transformer encoder。
- arm 和 hand 使用独立 Transformer decoder。
- 两个分支拥有独立输出 head，最后再拼接为完整动作。

这意味着 hand decoder 本身已经具备独立建模能力，迁移时没有必要删除它。真正需要替换的是：

- 输入侧由 hand joints 形成的状态表示；
- 训练目标侧由 hand joints 形成的动作表示；
- 输出侧 hand head 的目标空间。

### 2.2 目前仍然共享的关键接口

尽管输出是双 decoder，以下三个投影当前仍按完整 `state_dim` 接收 arm 和 hand 拼接向量：

- `input_proj_robot_state`：当前时刻 qpos 输入。
- `encoder_joint_proj`：CVAE posterior 的当前 qpos token。
- `encoder_action_proj`：CVAE posterior 的未来 action tokens。

因此，仅修改 hand 输出 head 并不能完成迁移。若不同时修改这三个接口，CVAE posterior 和 Transformer 条件仍然依赖 hand joints，训练与推理的表示就不一致。

### 2.3 当前 arm 监督与已有结果

`policy.py` 当前对完整拼接动作计算 L1，同时额外记录 arm/hand L1；FK pose loss 只施加在前 6 维 arm 关节上。已有实验记录显示，引入 FK loss 后成功率有明显提升。这说明迁移时应把 arm 路径视为需要保护的稳定基线，而不应把共享层全部重新训练。

### 2.4 当前数据的动作时刻语义

仓库的数据转换逻辑使 `action[t] = qpos[t+1]`；真实数据加载中使用 `action_start = max(0, start_ts - 1)`。因此当 `start_ts > 0` 时，action chunk 的第 0 项实际对应当前 `qpos[start_ts]`，而不是简单的下一帧。

手部 latent target 必须跟随数据集最终返回的 action 索引生成，不能直接假设第 `k` 个 query 总是对应 `start_ts + k + 1`。最安全的实现方式是让数据集显式返回每个 action token 对应的 episode frame index，然后据此构造点云窗口。

## 3. MyMimic 编解码器的实际接口语义

### 3.1 emEncoder

MyMimic 的 `emEncoder` 由两部分构成：

- 每帧点云先经 PointNet 风格的空间编码器；
- 帧特征序列再经 temporal Transformer encoder，输出每一帧的 latent。

当前仓库中配套 checkpoint 的实际配置是：

- Inspire hand；
- 6 个关节；
- 每帧 256 个点；
- latent 维度 128；
- decoder window size 为 20；
- Stage 2 encoder 训练序列长度为 50；
- 关节原始单位为 Inspire 的 `0–1000`，训练时转换为弧度 `[-pi/2, pi/2]`。

这与 `em-latent-migration.html` 中讨论的 10 关节、窗口 10 并不一致。实现必须以实际 checkpoint 元数据为准，不能把 HTML 中的示例常量写死进代码。

### 3.2 emDecoder

`emDecoder` 接收一段 latent 序列，先对 latent 做 L2 归一化，再输出每个关节的 `(sin(theta), cos(theta))`。最终角度由 `atan2(sin, cos)` 得到。

当前 `forward` 虽保留 teacher forcing 相关参数，但实际计算并未使用 ground-truth 关节作为自回归输入。因此接入 ACT 时无需围绕 teacher forcing 设计新的训练路径。

需要注意：

- `emDecoder` 的直接输出不是 ACT 当前使用的 `0–1000` 原始关节值；
- decoder 先得到弧度，再由 `raw = (theta + pi/2) / pi * 1000` 转回 Inspire 原始范围；
- ACT 的标准化与反标准化仍应以原始 `0–1000` 关节值为边界；
- 不可直接把 decoder 弧度输出与当前归一化 joint target 混合计算 L1。

### 3.3 一个必须先验证的时序风险

现有 `emEncoder` temporal Transformer 没有 causal mask，在输入序列内部是双向注意力。Stage 2 又以长度 50 的序列训练，因此训练得到的某一中间 latent 可能利用其后帧信息。

在线控制只能访问过去和当前点云。接入 ACT 前必须单独验证：使用长度 20 的历史窗口、只取最后一个 latent 时，冻结的 `emEncoder + emDecoder` 是否仍能稳定重建当前手部关节。

如果该测试不达标，应先对 codec 做以下一种修正，再训练 ACT：

- 给 temporal encoder 加 causal mask 并微调；或
- 使用长度 20 的滚动历史窗口、只监督最后 token 重新微调；或
- 重新训练一个严格 history-only 的 encoder，保持 decoder latent 空间不变。

在这个问题没有被验证前，不建议直接开始端到端 ACT 训练，否则可能把 codec 的离线未来信息泄漏误认为策略效果。

## 4. 对 em-latent-migration.html 方案的取舍

HTML 中以下方向是正确的：

- 保留 RGB、CVAE 和双 decoder 主体；
- hand 输入改用点云 encoder latent；
- hand 分支预测 latent；
- 用冻结的 `emDecoder` 将 latent 还原为关节动作；
- 分阶段训练，优先保护 arm。

但不建议第一阶段直接采用其中“将 hand token 注入共享 global memory，并加入 type/domain embedding”的完整融合方案。原因是：

- shared encoder 的注意力分布会立刻改变；
- 即使 arm decoder 不读取最后的 hand token，它读取到的其他 memory token 也已经被 hand token 改写；
- arm 性能发生变化后，很难区分是点云表示、共享 encoder、query 变化还是 hand loss 尺度导致；
- 这超出了本次“替换 hand 表示且尽量不影响 arm”的最小改动目标。

MyMimic 的完整融合结构更适合作为第二轮实验，而不是第一次迁移的起点。

## 5. 推荐的目标数据流

### 5.1 训练输入

对当前时刻 `t`：

```text
RGB(t) --------------------------> 原视觉 backbone / shared encoder

arm_qpos(t) ---------------------> arm normalization
pointcloud[t-W+1 : t] ----------> emEncoder(frozen)
                                  -> last latent
                                  -> L2 normalize
                                  -> z_hand_obs(t)

internal_qpos(t) = concat(arm_qpos_norm(t), z_hand_obs(t))
```

其中 `W=20` 应从 checkpoint 元数据读取。episode 开头不足 20 帧时使用与 codec 训练/验证一致的左侧填充规则。

### 5.2 训练目标与 CVAE posterior

对 action chunk 中每个有效 token 对应的真实 frame index `tau_k`：

```text
pointcloud[tau_k-W+1 : tau_k] --> emEncoder(frozen)
                                  -> last latent
                                  -> L2 normalize
                                  -> z_hand_gt(k)

internal_action(k) = concat(arm_action_norm(k), z_hand_gt(k))
```

CVAE posterior 继续接收：

```text
[CLS, internal_qpos(t), internal_action(0:K)]
```

这样训练时 posterior 不再依赖未来 hand joints，而是依赖未来点云所对应的 hand latent；推理时 posterior 仍按原 ACT 逻辑使用零 latent，不改变预测模型的核心方法。

原始 hand joints 仍可作为 codec 重建监督、蒸馏辅助目标和评估标签，但不再是新模型的 hand 输入。

### 5.3 Hand 分支输出

```text
hand Transformer decoder hidden states
    -> hand_latent_head: hidden_dim -> 128
    -> L2 normalize
    -> z_hand_pred[0:K]
    -> emDecoder(frozen)
    -> predicted sin/cos
    -> atan2
    -> radians
    -> Inspire raw 0-1000
    -> physical hand commands
```

保留现有 hand Transformer decoder 的好处是，它仍负责根据视觉、当前状态、CVAE style latent 和 action query 预测未来 hand chunk；`emDecoder` 只承担 latent-to-joints 的运动流形解码，不取代 ACT 的时序决策逻辑。

### 5.4 Arm 分支

Arm 路径保持：

```text
arm Transformer decoder hidden states
    -> original arm head
    -> arm joints
    -> original arm L1 + FK pose loss
```

第一阶段不要修改 arm head、FK 实现、arm action 的单位、反标准化和执行接口。

## 6. 三个共享投影的兼容改造

将内部维度直接从 12 改成 `6 + 128` 虽然最简单，但会使三个输入投影的权重形状全部变化，无法直接继承旧模型，而且 128 维 hand latent 很容易压过 6 维 arm。更稳妥的做法是把旧投影拆成“固定 arm 路径 + 新 hand adapter”。

以旧的 qpos 投影为例：

```text
E_old(q) = W_arm * q_arm + W_hand * q_hand + b

E_new(q) = W_arm * q_arm + b + A_hand(z_hand)
```

其中：

- `W_arm` 和 `b` 从旧 checkpoint 精确拷贝并先冻结；
- `A_hand` 是新的小型 adapter，例如 `Linear(128, hidden_dim)` 或两层 MLP；
- adapter 输出先乘一个可学习 gate，gate 初值可设得较小；
- adapter 训练目标之一是逼近旧模型的 `W_hand * q_hand_norm`。

三个接口应分别配置 adapter，不建议共享参数：

1. `proprio_hand_adapter`：替代 `input_proj_robot_state` 中的 hand 列。
2. `cvae_joint_hand_adapter`：替代 `encoder_joint_proj` 中当前 hand qpos 的列。
3. `cvae_action_hand_adapter`：替代 `encoder_action_proj` 中未来 hand action 的列。

它们语义不同，特别是第三个接口直接影响 posterior 的 `mu/logvar`，强制共享会增加不必要的耦合。

这种分解有四个优点：

- arm 输入到 hidden space 的线性映射保持逐元素一致；
- 能继续加载当前成功 checkpoint；
- hand latent 的 128 维不会因为简单拼接而在数值上主导 shared layers；
- 可以单独观测和限制每个 adapter 对共享表示的扰动。

### 6.1 另一种更省改动但风险更高的方案

也可以直接把 `state_dim` 改为 134，并重新建立三个 `Linear(134, hidden_dim)`，再把旧权重的前 6 列复制进去。该方案可用于后续消融实验，但不应作为保护 arm 的首选，因为 bias、hand 列初始化和输入统计变化都会立刻改变共享表示。

## 7. Arm 效果保护策略

### 7.1 第一阶段冻结范围

建议最初冻结：

- 图像 backbone 和 image input projection；
- shared Transformer encoder；
- arm Transformer decoder；
- arm output head；
- CVAE 的 `latent_out_proj`；
- shared action query embedding；
- `emEncoder` 和 `emDecoder`，并固定为 `eval()`；
- 三个旧投影中的 arm 权重和 bias。

最初训练：

- 三个 hand adapter；
- 新的 `hand_latent_head`；
- 必要时 hand Transformer decoder；
- 可学习 gate、latent 归一化相关的小层。

如果 hand decoder 与 arm decoder 确实完全独立，解冻 hand decoder 不会直接改写 arm decoder；但它仍通过 loss 与 shared memory 有间接关系，所以 shared encoder 在第一阶段应保持冻结。

### 7.2 处理 shared query embedding

当前 arm/hand decoder 共享 action query。如果直接训练 query，arm decoder 的输入会变化，等于绕过了对 arm 分支的冻结保护。建议二选一：

- 第一阶段把原 query 完全冻结；或
- 从旧 query 复制出 `hand_query = old_query + delta_hand`，只训练 `delta_hand`，arm 继续使用冻结的旧 query。

第二种更灵活，也更容易在后续阶段让 hand 适应 latent 目标。

### 7.3 Teacher 蒸馏

保留一份冻结的旧 checkpoint 作为 teacher。对同一批 RGB、arm qpos 和原 hand qpos，teacher 走原 joint 路径；student 使用点云 latent。建议加入：

- `L_arm_distill`：student arm chunk 与 teacher arm chunk 的 L1/MSE。
- `L_prop_embed`：student proprio embedding 与 teacher proprio embedding 的 MSE。
- `L_cvae_joint_embed`：当前 qpos token embedding 蒸馏。
- `L_cvae_action_embed`：未来 action token embedding 蒸馏，只对有效 token 计算。

embedding 蒸馏的意义不是让 latent 还原全部关节细节，而是把新 hand 表示投射到旧 ACT 已熟悉的条件空间，减少 shared encoder 和 CVAE posterior 的分布漂移。

当原始 hand joints 确认已不再进入 student 的预测图后，它们仍可供 teacher 使用。这不违反“输入从 joints 替换为点云 latent”的目标，因为 teacher 只在训练期提供保护信号，部署时完全移除。

### 7.4 逐步解冻

只有当以下条件同时满足时，才考虑解冻 shared encoder 的最后 1–2 层：

- codec history-only 测试通过；
- hand latent/decoded-joint 指标稳定；
- frozen-arm 阶段的 arm 离线指标不差于基线；
- arm 输出蒸馏误差已较小。

解冻时：

- shared encoder 使用显著更小的学习率；
- arm decoder/head 继续冻结；
- 保留 arm distillation 与原 FK loss；
- 设置“arm 退化即回滚”的早停规则，而不是只看总 loss。

## 8. Loss 设计

不能继续对 `[arm(6), hand latent(128)]` 的拼接结果计算单一平均 L1。否则 hand latent 会因维度多而占据绝大多数梯度，且 arm joints 与单位球 latent 也没有可比较的尺度。

建议总 loss 显式拆分：

```text
L_total = lambda_arm       * L_arm_joint
        + lambda_fk        * L_fk_pose
        + lambda_latent    * L_hand_latent
        + lambda_decode    * L_hand_decode
        + lambda_kl        * L_kl
        + lambda_preserve  * L_preserve
```

### 8.1 Arm loss

- `L_arm_joint`：保持当前 arm 前 6 维的 L1 逻辑。
- `L_fk_pose`：保持当前 FK position/orientation 监督和权重策略。
- `L_arm_distill`：在保护阶段加入，可并入 `L_preserve`。

### 8.2 Hand latent loss

因为 codec 在单位球 latent 上工作，优先使用 cosine distance：

```text
L_cos = mean(1 - dot(normalize(z_pred), normalize(z_gt)))
```

可辅以小权重 MSE：

```text
L_latent = L_cos + beta * MSE(z_pred_norm, z_gt_norm)
```

对 padding token 必须使用与当前 action loss 一致的 `is_pad` mask。

### 8.3 Decoded hand loss

建议训练初期保留一项通过冻结 `emDecoder` 反传到 `z_pred` 的重建 loss：

- 优先比较预测与 GT 的 sin/cos，避免角度周期边界问题；
- 可额外记录弧度 MAE 和转换回 `0–1000` 后的 MAE；
- decoder 参数冻结，但计算图不能包在 `no_grad()` 中，否则梯度无法回到 latent head。

即：`requires_grad=False` 只冻结 decoder 参数，不等于切断 decoder 对输入 latent 的梯度。

### 8.4 Preserve loss

```text
L_preserve = w1 * L_arm_distill
           + w2 * L_prop_embed
           + w3 * L_cvae_joint_embed
           + w4 * L_cvae_action_embed
```

这些 loss 可以在模型稳定后逐步衰减，但 arm distillation 建议至少保留到 shared encoder 解冻阶段结束。

### 8.5 Loss 日志

训练日志必须单独记录：

- arm joint MAE；
- FK position/orientation error；
- hand latent cosine similarity；
- hand decoded joint MAE；
- KL；
- teacher/student arm discrepancy；
- 三个 adapter 的输出范数和 gate 值。

只记录一个 `loss` 总数无法判断 arm 是否正在被 hand latent 迁移拖累。

## 9. 点云数据契约与时间对齐

### 9.1 点云必须与 codec 预训练一致

点云输入至少要锁定以下元数据：

- 坐标系；
- 单位；
- 点数 256；
- 点顺序/采样方法；
- wrist pose 的定义；
- episode 与相机帧的时间戳映射；
- 缺失帧和无效点处理；
- 历史窗口的左填充规则。

MyMimic 数据处理中，同一段序列的点索引跨帧保持对应关系，且点云归一化使用历史窗口计算的 centroid/radius。若当前数据重新逐帧随机采样 256 点，虽然 PointNet 对单帧点顺序不敏感，但时序特征的统计仍可能偏离预训练数据。

### 9.2 归一化不能偷看未来

对在线时刻 `t` 的历史窗口，centroid/radius 只能由 `t-W+1:t` 的可见点云计算。训练时生成 `z_hand_obs(t)` 也必须使用相同规则，不能用整个 episode 或未来 action chunk 的点云统计量。

对 target `z_hand_gt(tau)`，其窗口可使用截至 `tau` 的历史点云，因为这是监督标签；但必须确保该 target encoder 的窗口定义与 codec 训练时一致。

### 9.3 建议返回显式 frame index

为了消除 `action_start`、首帧和 padding 的歧义，dataset 最好同时返回：

```text
obs_frame_index
action_frame_indices[K]
is_pad[K]
```

然后由统一函数 `build_pc_history(episode, frame_index, W)` 构造 observation 和每个 target 的窗口。不要在 ACT policy 内根据 `start_ts + offset` 再猜测索引。

### 9.4 离线预计算还是在线编码

训练阶段建议优先离线预计算每帧的 history-only `z_hand` 并缓存：

- `emEncoder` 冻结时结果确定；
- 可显著减少每个 action chunk 重复编码重叠窗口的计算量；
- 更容易审计时间对齐和排查数据泄漏。

但缓存生成器必须与部署时在线窗口代码共享同一套预处理函数。验证阶段可抽样在线重算，并与缓存 latent 比较，防止两条路径悄然分叉。

## 10. 在接入 ACT 前先做 codec 独立验收

在策略训练之前，先冻结 ACT，只验证点云 codec。建议至少运行以下四组测试：

### 10.1 原配置重建

在与 MyMimic 预训练分布一致的数据上复现 checkpoint 的重建误差，确认加载、单位转换、latent 归一化和点云预处理没有问题。

### 10.2 History-only 最后 token 重建

对每个时刻只输入最近 20 帧，取最后 latent，经 decoder 解码当前关节。该结果最接近真实部署条件，是 ACT 接入的硬门槛。

### 10.3 滚动窗口连续性

检查相邻时刻 latent cosine distance、解码关节速度和加速度是否出现非物理跳变。若窗口滑动一帧会导致 latent 大幅跳动，ACT 将很难学习 chunk 预测。

### 10.4 当前数据域重建

在当前 ACT 数据集上做同样测试，并按 episode/动作幅度统计误差。若当前点云生成、相机、手型或 wrist pose 与 MyMimic 预训练域不同，必须先解决 domain gap，不能期望 ACT 自行补偿一个已冻结且失配的 codec。

建议把 codec 验收脚本做成独立工具，并输出：

- sin/cos MSE；
- radians MAE；
- Inspire raw `0–1000` MAE；
- 每个关节 MAE；
- latent 相邻帧 cosine similarity；
- 按 episode 的 P50/P90/P95。

## 11. 推荐训练阶段

### Phase 0：固定基线

目标：保存当前 joint-hand 双 decoder + FK 模型的可重复基线。

- 固定数据划分和随机种子；
- 保存 arm joint、FK、hand joint、成功率与推理延迟；
- 保存一批固定 validation samples 上的 teacher 输出和中间 embedding；
- 确认当前 checkpoint 可稳定复现。

### Phase 1：Codec 门控测试

目标：确认 `pointcloud history -> emEncoder -> emDecoder -> joints` 在严格在线条件下可用。

- 不训练 ACT；
- 使用 W=20 history-only 窗口；
- 验证时间对齐、单位转换和当前数据域；
- 不通过则先修 codec，而不是进入策略训练。

### Phase 2：Adapter 对齐预训练

目标：在不改变 ACT 主体的前提下，把 hand latent 映射到旧 joint embedding 空间。

- 冻结旧 ACT 和 codec；
- 只训练三个 hand adapter；
- 用原 hand joints 计算 teacher embedding target；
- 分别最小化三个 embedding distillation loss；
- 验证 adapter 输出范数与旧 hand projection 输出范数接近。

这一阶段不要求策略输出正确，只解决输入分布替换问题。

### Phase 3：Hand latent head 训练

目标：让保留的 hand Transformer decoder 预测 codec latent。

- shared encoder、arm decoder/head、旧 query、codec 继续冻结；
- 加载 Phase 2 adapters；
- 训练 hand decoder、hand latent head，或先只训练 latent head；
- 使用 latent loss + decoded sin/cos loss + KL；
- 同时保留 arm/FK loss 的日志和 teacher arm distillation。

如果只训练 latent head 无法收敛，再解冻 hand decoder；不要一开始同时解冻 shared encoder。

### Phase 4：有限联合微调

目标：提升 hand 在任务上下文中的预测能力，同时控制 arm 回归。

- 可解冻 shared encoder 最后 1–2 层；
- shared 层学习率显著低于 hand 层；
- arm decoder/head继续冻结；
- 保留 FK 与 arm distillation；
- 每个 epoch 与固定 teacher validation set 比较。

### Phase 5：可选的 MyMimic 深度融合

只有低风险方案达到或超过 hand joint 基线后，才评估 MyMimic 的 global hand token、type/domain embedding 或独立 hand memory。

此时推荐先做“hand-only memory augmentation”：

- shared visual encoder 输出保持不变；
- 在 encoder 之后追加 hand token；
- 该 token 只提供给 hand decoder；
- arm decoder 继续只读取原 visual memory。

它比把 token 放进 shared encoder 更能保护 arm，也能检验 hand decoder 是否确实需要额外的显式点云 memory。

## 12. 推理和 chunk 执行

### 12.1 在线状态

每个 rollout 维护长度 20 的原始点云环形缓存：

- episode reset 时清空；
- 不足 20 帧时按训练约定左填充；
- 每个控制时刻执行同一套点云归一化；
- `emEncoder.eval()`，取最后 latent；
- ACT 输出完整 arm chunk 与 hand latent chunk；
- `emDecoder.eval()` 一次解码 hand latent chunk。

### 12.2 Temporal aggregation

若当前 ACT 使用 temporal aggregation，不建议直接在 latent 空间对来自不同历史 chunk 的预测做加权平均。单位球上普通欧氏平均可能缩短向量范数，并产生 decoder 从未见过的 latent。

首选方式是：

1. 每个历史 chunk 的 hand latent 先分别 L2 normalize；
2. 分别经 `emDecoder` 解成 physical hand joints；
3. 在 physical arm joints 和 physical hand joints 上分别执行原 temporal aggregation；
4. 拼接为最终机器人命令。

如确实要在 latent 空间聚合，至少应在加权和之后重新 L2 normalize，并单独做消融验证。

### 12.3 异常与安全边界

推理路径应监控：

- 点云缺失或有效点过少；
- normalization radius 过小；
- latent 出现 NaN/Inf；
- decoded angle 超出机器人允许范围；
- 相邻时刻 hand command 跳变过大。

故障回退策略应在部署前明确，例如保持上一帧 hand command、使用安全默认手型或终止当前 rollout。不要让异常 latent 直接进入硬件执行。

## 13. 验收门槛与实验矩阵

### 13.1 Arm 不退化门槛

迁移模型应在同一 validation split 上满足：

- arm joint MAE 不明显差于当前基线；
- FK position/orientation error 不明显差于当前基线；
- teacher/student arm chunk 差异处于预先设定范围；
- 完整任务成功率的置信区间无显著退化。

“不明显差”应在实验前写成数值门槛。一个可用的初始规则是 arm joint MAE 和 FK position mean 各自不超过基线的 5%，同时 P90 不超过基线的 10%；最终阈值应结合现有多随机种子方差确定。

### 13.2 Hand 有效性门槛

- latent cosine similarity 优于简单的上一帧复制基线；
- decoded hand joint MAE 不差于原 hand-joint ACT 基线；
- hand 命令连续性满足执行要求；
- 实机/仿真任务成功率有统计意义上的持平或提升。

### 13.3 最小消融实验

建议依次比较：

1. 当前 joint-hand 双 decoder 基线。
2. latent hand，直接重建三个 134 维投影。
3. latent hand，arm 权重保留 + 三 adapter。
4. 方案 3 + embedding/arm distillation。
5. 方案 4 + decoded sin/cos loss。
6. 方案 5 + shared encoder 最后层微调。
7. 可选：hand-only memory augmentation。
8. 可选：MyMimic global token 进入 shared encoder。

每次只增加一个因素，才能回答“hand 改善来自哪里”和“arm 退化由什么引起”。

## 14. 建议的代码边界

以下是实现时建议的职责划分，不要求一次完成所有重构。

### 14.1 Codec wrapper

建立单一的 hand codec wrapper，统一处理：

- checkpoint/config 元数据读取；
- 点云窗口预处理；
- `emEncoder` 调用与 last-token 选择；
- latent L2 normalization；
- `emDecoder` 调用；
- sin/cos、radians、raw1000 的转换；
- 冻结/eval 状态。

不要让单位转换散落在 dataset、policy 和 rollout 三处。

### 14.2 Dataset

扩展 dataset 返回：

- RGB 与 arm qpos；
- observation hand point-cloud history 或预计算 latent；
- action arm joints；
- target hand latent；
- 可选原 hand joints，作为重建/蒸馏标签；
- frame indices 与 `is_pad`。

数据缓存中应记录 codec checkpoint hash、窗口大小、预处理版本，防止换 checkpoint 后继续误用旧 latent。

### 14.3 DETRVAE

保留函数级核心流程，只调整：

- 三个 joint/action 投影为 arm-preserving projection + hand adapter；
- hand output head 改成 `hidden_dim -> 128`；
- forward 返回 `arm_pred`、`hand_latent_pred`，而不是过早拼成同单位动作；
- 可选返回中间 embedding，便于蒸馏。

arm 和 latent 不应在模型内部伪装成同一种 action tensor后再计算统一 loss。

### 14.4 Policy

Policy 负责：

- 分别计算 arm、FK、latent、decoded-hand、KL、preserve loss；
- 应用 padding mask；
- 训练时通过 frozen decoder 回传到 latent head；
- 推理时把 hand latent 解码为 physical joints；
- 输出兼容现有执行端的 `[arm joints, hand joints]`。

### 14.5 配置

新增配置应显式包含：

- codec checkpoint 路径与 hash；
- `hand_latent_dim=128`；
- `pc_num_points=256`；
- `pc_history_window=20`；
- point-cloud normalization/version；
- freeze/unfreeze 列表；
- 各项 loss 权重；
- teacher checkpoint；
- decoder joint unit 与 raw range。

不要从 `state_dim` 或文件名隐式推断这些参数。

## 15. 实现前需要修正或确认的细节

- 当前 `pointcloud_embedding/models/emDecoder.py` 中存在面向原 MyMimic 工程的 `from model.emEncoder` 导入方式；迁入当前包时应改为可靠的包内相对导入。
- checkpoint 同时包含 encoder/decoder state dict，加载时要严格检查 missing/unexpected keys。
- codec 冻结后仍需显式 `eval()`，避免 dropout 使同一点云窗口得到不同 latent。
- 若训练使用预计算 latent，增强策略必须在预计算前完成或禁用；不能对缓存 latent 假装执行点云增强。
- CVAE 的 `is_pad` mask 必须同时覆盖 arm action、hand latent target 和 action embedding 蒸馏。
- episode 边界处必须清空点云历史，不能把上一条 episode 的点云填到下一条。
- decoded hand joints 应在执行前做机器人范围裁剪，但训练指标应同时记录裁剪前值，以暴露模型异常。

## 16. 内部 latent 迁移方案的推荐实现

如果后续选择让 ACT 内部直接使用 latent，推荐采用“**内部表示替换 + adapter 隔离 + 冻结 codec + teacher 保护**”：

```text
hand joints input
    -> 替换为 point-cloud history -> frozen emEncoder -> normalized latent

hand joints target/output
    -> 替换为 target latent / predicted latent -> frozen emDecoder -> hand joints

原 hand Transformer decoder
    -> 保留，只将其输出空间改为 128 维 latent

arm branch / FK / CVAE / visual encoder / chunk logic
    -> 第一阶段保持原结构和参数行为
```

该路线的第一轮不把 hand token 注入 shared encoder，也不同时重构 query、memory 和 decoder。先证明 codec 在 history-only 条件下可靠，再通过三个 adapter 对齐旧 embedding，随后训练 hand latent head。只有在 arm 指标稳定且 hand latent 方案已经成立后，才把 MyMimic 的 global token/type/domain embedding 作为独立的后续增强实验。

这条路线最符合当前目标：真正将 hand 信息的输入和输出都迁移到点云 latent 空间，同时最大限度保留已经验证有效的 arm 分支与 ACT 核心预测逻辑。

## 17. 备选方案：在 ACT 前增加 emEncoder/emDecoder

另一条更保守的路线是：不修改双 decoder ACT 的内部状态和输出空间，而是在 human hand 数据进入 ACT 前，先使用点云 codec 将点云转换成伪关节值：

```text
human hand point cloud history
    -> frozen emEncoder
    -> hand latent
    -> frozen emDecoder
    -> pseudo hand joints(6)
    -> 原双 decoder ACT
    -> predicted hand joints(6)
```

这条路线是可行且有意义的，尤其适合作为第一组低风险实验。它本质上不是让 ACT 学习 latent，而是把 `emEncoder + emDecoder` 当作 human 数据的关节伪标签生成器或跨域 retargeting 前端。ACT 仍然处理原来的 `[arm joints(6), hand joints(6)]`，因此其三个输入投影、CVAE、双 decoder、arm 分支、FK loss、hand joint head 和执行接口都可以保持不变，也不存在 128 维 latent 在 ACT 内部淹没 6 维 arm 表示的问题。

若 human 数据要用于完整的 ACT chunk 训练，不能只转换当前 hand 输入，还必须为 action chunk 中的每个有效时刻生成对应的 pseudo hand joint target：

```text
current point-cloud window  -> pseudo hand qpos
future point-cloud windows  -> pseudo hand action chunk
```

点云窗口必须按照 dataset 实际返回的 action frame index 构建，并遵守 history-only 规则，避免 `emEncoder` 的双向时序注意力引入未来信息。

该方案成立的关键前提不是 codec 在 robot 点云上的普通重建效果好，而是它能把 **human 点云稳定地解码到与 ACT robot hand joints 相同的关节定义、方向、范围和时序语义**。如果只验证过 robot 点云重建，不能据此推断 human 到 robot 的跨域映射同样可靠。需要额外验证 human 点云上的动作合理性、连续性和跨域一致性。

该方案的主要优点是：

- 对当前 ACT 改动最小，最有利于保护已经取得效果的 arm 分支；
- human 和 robot 数据最终都能落到统一的 robot joint action space；
- 可以离线预计算并缓存 pseudo joints，训练和部署成本较低；
- 容易与当前 joint-hand 基线进行公平比较和问题定位。

主要局限是：

- codec 的解码误差会直接成为 human 数据的伪标签噪声；
- 128 维 latent 中超出 6 个关节表达能力的几何信息会在进入 ACT 前丢失；
- human pseudo joints 与 robot 真实 joints 可能存在分布偏差；
- 如果未来目标是让策略直接利用点云几何，而不只是补齐缺失 joint，该方案能力有限。

如果点云仅用于补齐 human 训练数据，而机器人部署时仍能读取真实 hand joints，建议把 codec 放在 **离线数据预处理链路**，不必把它作为 ACT 的在线网络模块。human 数据使用 codec 生成的 pseudo joints；robot 数据继续使用真实 joints。训练时应对 pseudo joint 设置置信度、异常过滤或较低的样本/loss 权重，防止 codec 错误污染策略。

如果 human 与 robot 的 joint 输入分布差异较大，可以额外进行以下处理：

- 对 robot 点云也运行相同 codec，比较 decoded joints 与真实 joints 的偏差；
- 在 robot joint 输入中加入与 codec 重建误差相匹配的噪声；
- 对低置信度、跳变或越界的 human pseudo joints 过滤或降权；
- 分别记录 human pseudo-joint loss 与 robot real-joint loss。

因此，建议的实验顺序可以调整为：

1. 先完成严格 history-only 的 codec 跨域验证。
2. 离线生成 human pseudo hand joints。
3. 保持当前双 decoder ACT 完全不变，训练并检查 arm、FK 和任务成功率。
4. 如果该方案已满足任务需求，就没有必要立即把 ACT 内部改成 128 维 latent。
5. 只有当 pseudo-joint 瓶颈明显、codec 解码损失较大或任务确实需要保留更多点云几何信息时，再采用前文的内部 latent 迁移方案。

综合来看，**前置 `emEncoder + emDecoder` 是更低风险、更容易验证的第一阶段方案；内部 `[arm 6 + hand latent 128]` 则是信息保留能力更强但改动和训练风险更高的后续方案。**

补充说明：前文提出的 adapter 不是当前 ACT 已有模块，而是内部 latent 迁移方案中新增加的模块。其首选用途是把 128 维 latent 映射成原 hand projection 在 `hidden_dim` 空间中的贡献，并非简单压缩成 6 维关节。如果采用本节的前置 codec 方案，`emDecoder` 已经完成 `latent -> 6 joints`，因此不需要再增加该 adapter。
