import torch
import torch.nn as nn
import torch.nn.functional as F
from model.emEncoder import (
    PointCloudSequenceEncoder,
    LearnablePositionalEncoding,
    temporal_contrastive_loss,
    mmd_loss,
)

# ══════════════════════════════════════════════════════════════════════════════
# 工具函数：关节角度表示转换
# ══════════════════════════════════════════════════════════════════════════════
def angles_to_sincos(theta: torch.Tensor) -> torch.Tensor:
    """
    将关节角度转换为 (sin, cos) 双分量表示，消除角度奇点。
 
    原理：
        裸角度值在 0/2π 边界处不连续（例如 359° 和 1° 数值差 358°，实际仅差 2°），
        MSE 损失在此处产生错误的大梯度。
        用 (sin θ, cos θ) 表示后，空间变为连续的 R²，任意两角度间距
        由欧氏距离正确衡量，不存在奇点。
 
    参数：
        theta : [..., J]  关节角度，单位弧度
    返回：
        [..., J*2]  交错排列 [sin(θ₁), cos(θ₁), sin(θ₂), cos(θ₂), ...]
    """
    sin_t = torch.sin(theta)
    cos_t = torch.cos(theta)
    return torch.stack([sin_t, cos_t], dim=-1).flatten(start_dim=-2)

def sincos_to_angles(sincos: torch.Tensor) -> torch.Tensor:
    """
    将 (sin, cos) 双分量表示还原为关节角度。
 
    参数：
        sincos : [..., J*2]
    返回：
        [..., J]  关节角度，单位弧度，范围 (-π, π]
    """
    sc = sincos.reshape(*sincos.shape[:-1], -1, 2)   # [..., J, 2]
    return torch.atan2(sc[..., 0], sc[..., 1])        # [..., J]
 
# 窗口因果 Mask 生成
def build_windowed_mask(T: int, window: int, device: torch.device) -> torch.Tensor:
    '''
    生成一个窗口因果 Mask，形状为 [T, T]。
     掩码规则（PyTorch Transformer 约定）：
        mask[i, j] =  0.0   → i 可以 attend 到 j
        mask[i, j] = -inf   → i 不能 attend 到 j
    '''
    causal =  torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))
    out_of_win = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=-window)
    valid = causal & ~out_of_win   # True = 在窗口内的有效位置
    mask = torch.zeros(T, T, device=device)
    mask[~valid] = float('-inf')
    return mask    # [T, T]

class WindowedDecoder(nn.Module):
    '''
    带固定历史窗口的解码器
    '''
    def __init__(
            self,
            latent_dim:  int   = 128,
            joint_dim:   int   = 16,
            num_layers:  int   = 3,
            num_heads:   int   = 4,
            window_size: int   = 10,
            ffn_dim:     int   = None,
            max_len:     int   = 512,
            dropout:     float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.joint_dim   = joint_dim
        self.window_size = window_size
        ffn_dim = ffn_dim or 4 * latent_dim

        # ── 1. 可学习位置编码 ────────────────────────────
        self.pos_enc = LearnablePositionalEncoding(
            d_model=latent_dim, dropout=dropout, max_len=max_len
        )

        # ── 2. 因果 Transformer 层堆叠 ────────────────────────────────────
        layer = nn.TransformerEncoderLayer(
            d_model        = latent_dim,
            nhead          = num_heads,
            dim_feedforward = ffn_dim,
            dropout        = dropout,
            batch_first    = True,
            norm_first     = True,   # Pre-LN，训练更稳定
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)

        # ── 3. 逐帧直读分支（direct head）────────────────────────────────
        # 直接从「当前帧 latent z_t」读出当前姿态，路径短、梯度直接，
        # 避免高动态关节的细粒度差异在 Transformer 时间维上被平均掉。
        hidden = latent_dim * 2
        self.direct_head = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, joint_dim * 2),   # → [B, T, J*2]
        )

        # ── 4. 时序修正分支（temporal head）──────────────────────────────
        # 利用历史窗口的上下文（趋势、速度连续性）对 direct 分支做修正，
        # 输出层零初始化，训练初期等价于「纯逐帧直读」，再逐步学习时序修正。
        self.temporal_head = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, joint_dim * 2),   # → [B, T, J*2]
        )
        nn.init.zeros_(self.temporal_head[-1].weight)
        nn.init.zeros_(self.temporal_head[-1].bias)

    def forward(self, z: torch.Tensor, gt_sincos: torch.Tensor = None, teacher_forcing_ratio: float = 1.0) -> torch.Tensor:
        """
        前向传播：完全基于因果 Transformer
        """
        B, T, D = z.shape

        # ── 分支 1：逐帧直读 ──────────────────────────────────────────────
        # 直接作用在原始 latent 上（不加位置编码、不经过时间注意力），
        # 保证每帧的局部差异以最短路径进入输出。
        direct_sincos = self.direct_head(z)                 # [B, T, J*2]

        # ── 分支 2：时序修正 ──────────────────────────────────────────────
        x = self.pos_enc(z)
        mask = build_windowed_mask(T, self.window_size, z.device)
        h = self.transformer(x, mask=mask)
        temporal_sincos = self.temporal_head(h)             # [B, T, J*2]

        # 两分支相加：direct 提供姿态主体，temporal 提供时序修正
        raw_sincos = direct_sincos + temporal_sincos

        # 归一化到单位圆；eps 防止 raw≈0 时 normalize 出 NaN（Stage3 训练初期 z' 未对齐时常见）
        sincos = F.normalize(
            raw_sincos.view(B, T, -1, 2), dim=-1, eps=1e-6,
        ).view(B, T, -1)
        return sincos
    
    @torch.no_grad()
    def decode_frame(self, z_t, cache: list) -> tuple:
        '''
        单帧推理接口
        原理：
            维护长度不超过 window_size 的滑动缓存（历史 z 序列）。
            每帧到来时拼接缓存，做一次完整的窗口注意力，取最后一帧输出。
        '''
        cache.append(z_t)
        if len(cache) > self.window_size:
            cache = cache[-self.window_size:]
 
        z_window = torch.cat(cache, dim=0).unsqueeze(0)   # [1, t_win, D]
        sincos   = self.forward(z_window)                 # [1, t_win, J*2]
        sincos_t = sincos[:, -1, :]                       # [1, J*2]
        theta_t  = sincos_to_angles(sincos_t)             # [1, J]
        return theta_t, cache
 
# ══════════════════════════════════════════════════════════════════════════════
# 损失函数
# ══════════════════════════════════════════════════════════════════════════════
def reconstruction_loss(
        pred_sincos: torch.Tensor,
        gt_theta:    torch.Tensor,
) -> torch.Tensor:
    """
    关节角度重建损失（主监督信号）。
 
    将真值角度转为 sin/cos 后计算 MSE，避免角度奇点带来的梯度错误。
 
    参数：
        pred_sincos : [B, T, J*2]
        gt_theta    : [B, T, J]
    """
    gt_sincos = angles_to_sincos(gt_theta)
    return F.mse_loss(pred_sincos, gt_sincos)

def velocity_loss(
        pred_sincos: torch.Tensor,
        gt_theta: torch.Tensor
) -> torch.Tensor:
    '''
    关节速度匹配
    '''
    gt_sincos = angles_to_sincos(gt_theta)
    pred_vel = pred_sincos[:, 1:] - pred_sincos[:, :-1]  # [B, T-1, J*2]
    gt_vel = gt_sincos[:, 1:] - gt_sincos[:, :-1]
    return F.mse_loss(pred_vel, gt_vel)

def acceleration_loss(
        pred_sincos: torch.Tensor,
        gt_theta: torch.Tensor
) -> torch.Tensor:
    '''
    关节加速度匹配损失
    '''
    gt_sincos = angles_to_sincos(gt_theta)
    pred_acc = pred_sincos[:, 2:] - 2*pred_sincos[:,1:-1] + pred_sincos[:, :-2]
    gt_acc = gt_sincos[:, 2:] - 2*gt_sincos[:, 1:-1] + gt_sincos[:, :-2]
    return F.mse_loss(pred_acc, gt_acc)

def joint_training_loss(
        pred_sincos: torch.Tensor,
        gt_theta:    torch.Tensor,
        z_robot:     torch.Tensor,
        z_human:     torch.Tensor = None,
        w_recon:     float = 1.0,
        w_vel:       float = 0.1,
        w_acc:       float = 0.01,
        w_tc:        float = 0.1,
        w_mmd:       float = 0.05,
) -> dict:
    '''联合训练总损失'''
    l_recon = reconstruction_loss(pred_sincos, gt_theta)
    l_vel   = velocity_loss(pred_sincos, gt_theta)
    l_acc   = acceleration_loss(pred_sincos, gt_theta)
    l_tc    = temporal_contrastive_loss(z_robot)

    z_r_flat = z_robot.reshape(-1, z_robot.shape[-1])
    if z_human is not None:
        z_h_flat = z_human.reshape(-1, z_human.shape[-1])
        l_mmd = mmd_loss(z_h_flat, z_r_flat)
    else:
        l_mmd = torch.tensor(0.0, device=z_robot.device)

    total = (w_recon*l_recon + w_vel*l_vel + w_acc*l_acc + w_tc*l_tc + w_mmd*l_mmd)

    return {
        'total': total,
        'recon': l_recon.detach(),
        'vel': l_vel.detach(),
        'acc': l_acc.detach(),
        'tc': l_tc.detach(),
        'mmd': l_mmd.detach()
    }


# ══════════════════════════════════════════════════════════════════════════════
# 完整系统：Encoder + Decoder
# ══════════════════════════════════════════════════════════════════════════════

class HandMotionSystem(nn.Module):
    """
    emEncoder + WindowedCausalTransformerDecoder 的完整系统封装。
 
    训练流程：
        1. 灵巧手点云 → emEncoder（lr=1e-5）→ z_robot
        2. 人手点云   → emEncoder（共享权重）→ z_human（可选）
        3. z_robot    → Decoder（lr=1e-3）→ pred_sincos
        4. joint_training_loss 计算联合损失
        5. 一次 backward，两组参数各自以不同步长更新
 
    推理流程：
        - 离线：infer_sequence，输入完整序列，一次前向
        - 实时：infer_realtime，逐帧调用，维护滑动缓存
 
    参数：
        encoder_cfg : dict，PointCloudSequenceEncoder 构造参数
        decoder_cfg : dict，WindowedDecoder 构造参数
    """
    def __init__(self, encoder_cfg: dict = None, decoder_cfg: dict = None):
        super().__init__()
        self.encoder = PointCloudSequenceEncoder(**(encoder_cfg or {}))
        self.decoder = WindowedDecoder(**(decoder_cfg or {}))
 
    def forward(
            self,
            x_robot: torch.Tensor,
            x_human: torch.Tensor = None,
            gt_theta: torch.Tensor = None,
            teacher_forcing_ratio: float = 1.0,
    ) -> dict:
        """
        训练时前向传播。
 
        参数：
            x_robot : [B, T, N, 3]  灵巧手点云序列
            x_human : [B, T, N, 3]  人手点云序列（可选，用于 MMD）
            gt_theta: [B, T, J] 真实关节角度 (可选，用于 Teacher Forcing)
            teacher_forcing_ratio: Scheduled Sampling 概率
        返回：
            dict，包含 pred_sincos、z_robot、z_human
        """
        # latent 单位球归一化：必须与 Stage1 训练一致。
        # Stage1 仅在归一化后的 latent（方向）上做跨域对齐 / 可解码，半径未受约束且两域不同；
        # 这里只取共享的方向分量喂给 decoder，保证 Stage1/2 表征空间一致，
        # 并让 decoder 在 Stage3 能从机械手迁移到人手。
        z_robot = F.normalize(self.encoder(x_robot), dim=-1)
        z_human = F.normalize(self.encoder(x_human), dim=-1) if x_human is not None else None
        gt_sincos = angles_to_sincos(gt_theta) if gt_theta is not None else None
        pred_sincos = self.decoder(z_robot, gt_sincos, teacher_forcing_ratio)
        return {
            'pred_sincos': pred_sincos,
            'z_robot':     z_robot,
            'z_human':     z_human,
        }
 
    @torch.no_grad()
    def infer_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """
        离线序列推理：输入完整的点云序列，输出整个关节序列 (弧度)
        :param x: [B, T, N, 3]
        :return:  [B, T, J]
        """
        self.eval()
        z = F.normalize(self.encoder(x), dim=-1)   # 与 Stage1/训练保持一致
        sincos = self.decoder(z)
        return sincos_to_angles(sincos)

    @torch.no_grad()
    def infer_realtime(
            self,
            x_t:   torch.Tensor,
            cache: list,
    ) -> tuple:
        """
        在线实时推理 (流式处理单帧)
        :param x_t:   [B, N, 3] 当前帧点云
        :param cache: 过去的特征列表 (长度 <= window_size)
        :return: (预测角度 [B, J], 更新后的 cache)
        """
        self.eval()
        # 1. 提取当前帧特征（与训练一致，做单位球归一化）
        z_t = F.normalize(self.encoder(x_t.unsqueeze(1)).squeeze(1), dim=-1) # [B, latent_dim]

        # 2. 维护缓存
        if not isinstance(cache, list):
            cache = []
        cache.append(z_t)
        if len(cache) > self.decoder.window_size:
            cache = cache[-self.decoder.window_size:]
        
        z_seq = torch.stack(cache, dim=1) # [B, T', latent_dim]

        # 3. 实时解码
        sincos = self.decoder(z_seq)      # [B, T', J*2]
        sincos_t = sincos[:, -1, :]       # 取最新一帧 [B, J*2]
        
        theta_t = sincos_to_angles(sincos_t)
        return theta_t, cache
 
    def get_optimizer(
            self,
            encoder_lr:   float = 1e-5,
            decoder_lr:   float = 1e-3,
            weight_decay: float = 1e-4,
    ) -> torch.optim.Optimizer:
        """
        差异化学习率优化器。
 
        encoder_lr 极小（1e-5）：
            保护跨形态对齐结构，同时允许关节监督微调编码器补充细粒度信息。
        decoder_lr 正常（1e-3）：
            decoder 从随机初始化开始，需要足够大的学习步长。
        """
        return torch.optim.AdamW([
            {'params': self.encoder.parameters(), 'lr': encoder_lr},
            {'params': self.decoder.parameters(), 'lr': decoder_lr},
        ], weight_decay=weight_decay)
    

# ══════════════════════════════════════════════════════════════════════════════
# 使用示例
# ══════════════════════════════════════════════════════════════════════════════
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
 
    B, T, N  = 4, 20, 256
    J        = 25
    LATENT_D = 128
    WINDOW   = 10
 
    encoder_cfg = dict(
        point_feat_dim  = 256,
        latent_dim      = LATENT_D,
        temporal_layers = 2,
        temporal_heads  = 4,
        dropout         = 0.1,
    )
    decoder_cfg = dict(
        latent_dim  = LATENT_D,
        joint_dim   = J,
        num_layers  = 3,
        num_heads   = 4,
        window_size = WINDOW,
        dropout     = 0.1,
    )
 
    system    = HandMotionSystem(encoder_cfg, decoder_cfg).to(device)
    optimizer = system.get_optimizer(encoder_lr=1e-5, decoder_lr=1e-3)
 
    # ── 参数量统计 ──────────────────────────────────────────────────────────
    print('=' * 60)
    print('模型参数量')
    print('=' * 60)
    enc_p = sum(p.numel() for p in system.encoder.parameters())
    dec_p = sum(p.numel() for p in system.decoder.parameters())
    print(f'  emEncoder : {enc_p:,}')
    print(f'  Decoder   : {dec_p:,}')
    print(f'  总计      : {enc_p + dec_p:,}')
 
    # ── 训练步 ─────────────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('训练模式')
    print('=' * 60)
    system.train()
    x_robot  = torch.randn(B, T, N, 3).to(device)
    x_human  = torch.randn(B, T, N, 3).to(device)
    gt_theta = torch.randn(B, T, J).to(device)
 
    out    = system(x_robot, x_human)
    losses = joint_training_loss(
        pred_sincos = out['pred_sincos'],
        gt_theta    = gt_theta,
        z_robot     = out['z_robot'],
        z_human     = out['z_human'],
    )
    optimizer.zero_grad()
    losses['total'].backward()
    optimizer.step()
 
    print(f"  pred_sincos : {out['pred_sincos'].shape}")
    print(f"  z_robot     : {out['z_robot'].shape}")
    for k, v in losses.items():
        print(f"  loss/{k:<8}: {v.item():.6f}")
 
    # ── 离线序列推理 ────────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('推理模式：离线序列')
    print('=' * 60)
    theta_seq = system.infer_sequence(torch.randn(1, T, N, 3).to(device))
    print(f'  输入 [1,{T},{N},3] → 输出 {tuple(theta_seq.shape)}')
 
    # ── 实时逐帧推理 ────────────────────────────────────────────────────────
    print('\n' + '=' * 60)
    print('推理模式：实时逐帧')
    print('=' * 60)
    cache = {}
    for t in range(T):
        theta_t, cache = system.infer_realtime(
            torch.randn(1, N, 3).to(device), cache
        )
    print(f'  单帧输出 shape : {theta_t.shape}')
    print(f'  缓存帧数       : {len(cache.get("z_cache", []))} / {WINDOW}（窗口上限）')
    print(f'  关节角度范围   : [{theta_t.min().item():.3f}, '
          f'{theta_t.max().item():.3f}] rad')
 
 
if __name__ == '__main__':
    main()



