import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 正弦位置编码
class PositionalEncoding(nn.Module):
    '''
    原理：
        对序列中第 t 个位置、第 i 维特征，编码值为：
            PE(t, 2i)   = sin(t / 10000^(2i/d_model))
            PE(t, 2i+1) = cos(t / 10000^(2i/d_model))
 
        - 不同位置对应唯一的编码向量（由不同频率的正弦/余弦混合而成）
        - 任意两个位置之间的相对距离可以通过线性变换表达
          → Transformer 的 attention 可以"感知"帧的先后顺序
 
    优点：
        - 无需额外参数，不会过拟合
        - 天然支持比训练时更长的序列（泛化性好）
 
    参数：
        d_model : 特征维度，须与 TemporalEncoder 的 in_dim 一致
        max_len : 支持的最大序列长度（帧数上限）
        dropout : 作用于编码后的特征，轻微防止过拟合
    """
    '''
    def __init__(self, d_model: int, max_len: int=512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # 预计算编码矩阵，形状[max_len, d_model]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 分母项：10000^(2i/d_model)，在 log 空间计算以提升数值稳定性
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )  
        pe[:, 0::2] = torch.sin(position * div_term)   # 偶数维 → sin
        pe[:, 1::2] = torch.cos(position * div_term)   # 奇数维 → cos
        # 注册为 buffer：随模型保存/加载，但不参与梯度计算
        self.register_buffer('pe', pe.unsqueeze(0))    # [1, max_len, d_model]

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        x : [B, T, d_model]
        返回: [B, T, d_model]  （叠加位置编码后）
        """
        # x + PE 的前 T 个位置向量（广播至 batch 维）
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# 可学习位置编码
class LearnablePositionalEncoding(nn.Module):
    """
    可学习位置编码
 
    原理：
        将每个位置 t 对应一个可训练的向量 e_t ∈ R^{d_model}，
        在训练过程中通过反向传播自动学习最适合当前任务的位置表示。
 
    优点：
        - 更灵活，能自适应地学习动作序列的时序结构
        - 当序列中不同位置有非均匀的重要性差异时表现更好
 
    缺点：
        - 序列长度不能超过 max_len（超出则无对应 embedding）
        - 引入额外参数，小数据集上有过拟合风险
 
    参数：
        d_model : 特征维度
        max_len : 训练时见过的最大序列长度
        dropout : 作用于编码后的特征
    """
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # 可训练的 embedding 表，形状 [max_len, d_model]
        self.pe = nn.Embedding(max_len, d_model)
        nn.init.normal_(self.pe.weight, mean=0.0, std=0.02)   # 小方差初始化，稳定早期训练
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [B, T, d_model]
        返回: [B, T, d_model]
        """
        T = x.size(1)
        positions = torch.arange(T, device=x.device)          # [T]
        x = x + self.pe(positions).unsqueeze(0)               # [1, T, d_model] 广播
        return self.dropout(x)

# pointNet-style 单帧点云编码器 将 单帧的 无序点云 转成一个固定维度的 特征向量
class PointNetEncoder(nn.Module):
    '''
    encode one frame of point cloud into a vector
    
    [B,N,3] -> [B,C]
    Input:[B,N,3]  N points, XYZ coordinates
    Output:[B,C]  global feature per frame
    '''
    def __init__(self,out_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3,64),
            nn.ReLU(inplace=True),
            nn.Linear(64,128),
            nn.ReLU(inplace=True),
            nn.Linear(128,out_dim)
        )

    def forward(self, x):
        '''
        x:[B, N, 3]
        '''
        feat = self.mlp(x)   # [B, N, C]
        feat = torch.max(feat, dim=1)[0] #[B, C] (max pooling)
        return feat
    
# 时序编码器  学习时序特征，保证将 ”逐帧几何特征” 转成 “动作状态表示”
class TemporalEncoder(nn.Module):
    '''
    Model temporal dependency betwwen frames.
    
    [B, T, C] -> [B, T, D]
    Input: frame_feat 
    Output: z

    -Allow each z_t to use temporal context
    -Avoids treating frames independently
    '''
    def __init__(self, in_dim, out_dim, num_layers=2, num_heads=4, max_len=512, dropout=0.1):
        super().__init__()
        # 计算位置编码
        self.pos_enc = PositionalEncoding(
                d_model=in_dim, max_len=max_len, dropout=dropout
            )
        # 初始化编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_dim,
            nhead=num_heads,
            dim_feedforward=4 * in_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        x = self.pos_enc(x)      # 添加位置编码
        h = self.transformer(x)  # [B, T, C] temporal reasoning
        z = self.proj(h)         # [B, T, D]
        return z
    
# full Point Cloud Sequence Encoder
class PointCloudSequenceEncoder(nn.Module):
    '''
    Full Encoder, Encode a sequence of point clouds into a sequence of latent vectors.
    [B, T, N, 3] -> [B, T, D]
    
    -Each z_t represents an abstract "action state"
    -Shared between human hand and robotic hand
    '''
    def __init__(
            self,
            point_feat_dim=256,
            latent_dim=128,
            temporal_layers=2,
            temporal_heads=4,
            max_len=512,
            dropout=0.1
    ):
        super().__init__()
        self.point_encoder = PointNetEncoder(out_dim=point_feat_dim)
        self.temporal_encoder = TemporalEncoder(
            in_dim=point_feat_dim,
            out_dim=latent_dim,
            num_layers=temporal_layers,
            num_heads=temporal_heads,
            max_len=max_len,
            dropout=dropout
        )
    
    def forward(self, x):
        '''
        x: [B, T, N, 3]
        '''
        B, T, N, _ = x.shape
        # encode each frame independently
        x = x.view(B * T, N, 3)
        frame_feat = self.point_encoder(x)   # [B*T, C]
        frame_feat = frame_feat.view(B,T,-1) # [B, T, C]
        z = self.temporal_encoder(frame_feat) # [B, T, D]
        return z
    
# Loss Functions

## Temporal Contrastive Loss
def temporal_contrastive_loss(z, temperature=0.1):
    '''
    Enforce temporal consistency in latent space.

    Positive pairs: (z_t^i, z_{t+1}^i) same sequence, adjacent frames
    Negative pairs: All other frames from the same sequence AND other sequences

    -Same action evolving over time stays close
    -Different times in same action stay separated (prevents temporal collapse)
    -Different demonstrations stay separated
    '''
    B, T, D = z.shape
    z = F.normalize(z,dim=-1)

    # [B*(T-1), D]
    z_curr = z[:, :-1, :].reshape(-1, D)
    # [B*(T-1), D]
    z_next = z[:, 1:, :].reshape(-1, D)

    # Full similarity matrix [B*(T-1), B*(T-1)]
    logits = torch.matmul(z_curr, z_next.T) / temperature
    
    # Correct pairs are on the diagonal
    labels = torch.arange(B * (T - 1), device=z.device)

    loss = F.cross_entropy(logits, labels)

    return loss

## Cross-morphology Alignment(MMD)
def rbf_kernel(x,y,sigma):
    '''
    Radial Basis Function kernel.
    Measures similarity between distributions in RKHS
    '''
    x_norm = (x ** 2).sum(dim=1).view(-1,1)
    y_norm = (y ** 2).sum(dim=1).view(1,-1)
    dist = x_norm + y_norm - 2*torch.mm(x,y.t())
    return torch.exp(-dist/(2*sigma**2))

def mmd_loss(z_human, z_robot, sigmas=(0.5, 1.0, 2.0)):
    '''
    Align latent distributions of human and robotic hand
    '''
    loss = 0.0
    for sigma in sigmas:
        K_hh = rbf_kernel(z_human, z_human, sigma)
        K_rr = rbf_kernel(z_robot, z_robot, sigma)
        K_hr = rbf_kernel(z_human, z_robot, sigma)

        loss += K_hh.mean() + K_rr.mean() -2*K_hr.mean()

    return loss/len(sigmas)

def temporal_smoothness_loss(z):
    '''时序平滑损失'''
    return ((z[:,1:]-z[:,:-1]) ** 2).mean()


# ══════════════════════════════════════════════════════════════════════════════
# 关节重建辅助头（路线 A）
# ══════════════════════════════════════════════════════════════════════════════
class JointReconstructionHead(nn.Module):
    '''
    逐帧关节重建辅助头（仅 Stage1 训练时使用，不参与部署）。

    作用：
        在对比 / MMD / 平滑目标之外，额外强制 latent z_t 保留
        “可解码出关节姿态” 的信息，使编码器从单纯的「动作语义压缩器」
        变为兼具「姿态测量器」能力，解决 latent 不可解码的根因。

    设计：
        - 该头独立于 PointCloudSequenceEncoder，Stage2/Stage3 只复用 encoder 主体，
          因此 Stage1 保存的 encoder 权重结构保持不变，下游加载无需改动。
        - 逐帧作用在 latent 上，对应 “linear probe” 的可读出性，但与编码器联合训练，
          梯度会反向塑造 latent 表征。

    [B, T, D] -> [B, T, J*2]  (单位圆归一化的 sin/cos)

    head_type:
        'linear' : 单层线性头。姿态信息被强制线性地存放在 latent 本身，
                   latent 越是干净的姿态流形，MMD 越能把可解码性迁移给人手
                   （推荐，与 linear-probe 判据一致）。
        'mlp'    : 两层非线性头。表达力更强，但可能由头部自己扛下解码，
                   使 latent 保持松散，迁移性变差。
    '''
    def __init__(self, latent_dim: int, joint_dim: int, dropout: float = 0.1,
                 head_type: str = 'linear'):
        super().__init__()
        if head_type == 'linear':
            self.net = nn.Linear(latent_dim, joint_dim * 2)
        elif head_type == 'mlp':
            hidden = latent_dim * 2
            self.net = nn.Sequential(
                nn.Linear(latent_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, joint_dim * 2),
            )
        else:
            raise ValueError(f"未知的 head_type: {head_type}，应为 'linear' 或 'mlp'")

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        B, T, _ = z.shape
        raw = self.net(z)
        sincos = F.normalize(raw.view(B, T, -1, 2), dim=-1).view(B, T, -1)
        return sincos


def joint_reconstruction_loss(pred_sincos: torch.Tensor, gt_theta: torch.Tensor) -> torch.Tensor:
    '''
    关节角度重建损失（路线 A 辅助监督）。

    将真值角度转为 (sin, cos) 后计算 MSE，避免角度在 0/2π 边界处的奇点。

    参数：
        pred_sincos : [B, T, J*2]
        gt_theta    : [B, T, J]  弧度
    '''
    sin_t = torch.sin(gt_theta)
    cos_t = torch.cos(gt_theta)
    gt_sincos = torch.stack([sin_t, cos_t], dim=-1).flatten(start_dim=-2)
    return F.mse_loss(pred_sincos, gt_sincos)


def main():
    '''使用案例'''
    B, T, N = 4, 10, 256
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # human and robot point cloud sequences
    x_human = torch.randn(B, T, N, 3).to(device)
    x_robot = torch.randn(B, T, N, 3).to(device)

    encoder = PointCloudSequenceEncoder().to(device)

    # Encode
    z_h = encoder(x_human)
    z_r = encoder(x_robot)

    # compute loss
    loss_tc = temporal_contrastive_loss(z_h) + temporal_contrastive_loss(z_r)
    loss_mmd = mmd_loss(z_h.reshape(-1, z_h.shape[-1]),
                        z_r.reshape(-1,z_r.shape[-1]))
    loss_smooth = temporal_smoothness_loss(z_h)

    total_loss = loss_tc + 0.1 * loss_mmd + 0.01 * loss_smooth

    print("Temporal contrastive:",loss_tc.item())
    print("MMD alignment:", loss_mmd.item())
    print("Smoothness:", loss_smooth.item())
    print("Total loss:", total_loss.item())

if __name__ == "__main__":
    main()