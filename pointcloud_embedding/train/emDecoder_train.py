'''
stage 2: 训练 emDecoder

当前版本训练策略：
    - emEncoder: 加载阶段一预训练权重，可冻结或极小学习率微调
    - Decoder  : 从随机初始化开始训练
    - 损失     : 以重建 loss 为主，辅以高动态关节的时间方差匹配项，抑制“直线解”

数据：自动识别 Inspire(hand_joints/6)、L10(hand_joints/10) 与
      L21(hand_qpos/25)，并按手型将原始驱动值转为训练弧度表示。
'''

import os
import sys
import time
from datetime import datetime
import argparse
from pathlib import Path
from tqdm import tqdm
import wandb

sys.path.insert(0, str(Path(__file__).parent.parent))

# 须在 import torch 之前加载，以应用 warning 过滤
from constants import *

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
 
from model.emDecoder import *
from model.emEncoder import temporal_contrastive_loss, temporal_smoothness_loss
from mimic_utils.hand_joint_utils import (
    L21_IGNORED_JOINT_IDS_STR,
    apply_fixed_joint_values,
    hand_type_from_joint_dim,
    joints_to_radians,
    parse_joint_id_list,
    resolve_fixed_joint_ids,
    resolve_hand_joint_key,
    resolve_ignored_joint_ids,
)
from mimic_utils.compute_joint_stats import resolve_joint_stats_path

import h5py
import numpy as np
from scipy.signal import savgol_filter

# ══════════════════════════════════════════════════════════════════════════════
# 日志记录器
# ══════════════════════════════════════════════════════════════════════════════
class Logger(object):
    """
    重定向标准输出，使 print 的内容同时输出到终端和 log.txt 文件中
    """
    def __init__(self, filename="log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════
def parse_frame_stride_choices(value) -> tuple[int, ...]:
    """解析 "1,2" / iterable，返回去重排序的正整数 stride。"""
    tokens = value.split(",") if isinstance(value, str) else value
    choices = tuple(sorted({int(token) for token in tokens if str(token).strip()}))
    if not choices or choices[0] < 1:
        raise ValueError(f"帧步长必须为正整数，实际 {value!r}")
    return choices


class RobotHandDataset(Dataset):
    '''
    灵巧手配对数据集: 点云序列 + 关节角度序列

    参数：
        clouds_path : str，点云数据路径（目录，包含 .npy 点云文件）
        joints_path : str，关节角度数据路径（目录，包含 .hdf5 文件）
                          目录模式下与 clouds_path 相同，按命名规则配对：
                          如 episode_0_pointclouds.npy 对应 episode_0.hdf5
                          自动识别 Inspire hand_joints[T,6] /
                          L10 hand_joints[T,10] / L21 hand_qpos[T,25]
        seq_len     : 统一序列长度 T
        num_points  : 每帧点数 N
        augment     : 是否启用数据增强
    '''
    def __init__(
            self,
            clouds_path: str,
            joints_path: str,
            seq_len:     int  = 20,
            num_points:  int  = 256,
            augment:     bool = False,
            smooth_window: int = 5,
            smooth_poly: int = 2,
            joint_unit: str = 'radians',
            frame_stride: int = 2,
            frame_stride_choices=None,
            index_frame_stride: int = None,
            show_progress: bool = False,
            progress_desc: str = "加载 RobotHandDataset",
    ):
        self.seq_len = seq_len
        self.num_points = num_points
        self.augment = augment
        self.smooth_window = smooth_window
        self.smooth_poly = smooth_poly
        self.joint_unit = joint_unit
        self.frame_stride = int(frame_stride)
        if self.frame_stride < 1:
            raise ValueError(f"frame_stride 必须 >= 1，实际 {self.frame_stride}")
        choices = frame_stride_choices or (self.frame_stride,)
        self.frame_stride_choices = tuple(sorted({int(x) for x in choices}))
        if not self.frame_stride_choices or self.frame_stride_choices[0] < 1:
            raise ValueError(f"frame_stride_choices 必须为正整数，实际 {choices}")
        self.index_frame_stride = int(
            index_frame_stride
            if index_frame_stride is not None
            else max(self.frame_stride_choices)
        )
        if self.index_frame_stride < max(self.frame_stride_choices):
            raise ValueError(
                f"index_frame_stride={self.index_frame_stride} 小于训练最大 stride="
                f"{max(self.frame_stride_choices)}，无法保证窗口不越界"
            )
        self.hand_type = None   # 'inspire' | 'l10' | 'l21'
        self.joint_dim = None

        cp = Path(clouds_path)
        jp = Path(joints_path)

        self.clouds = []
        self.joints = []

        if cp.is_dir() and jp.is_dir():
            cloud_files = sorted(cp.glob('*_pointclouds.npy'))
            file_iter = cloud_files
            if show_progress:
                file_iter = tqdm(cloud_files, desc=progress_desc, leave=False)
            for cloud_file in file_iter:
                # 解析出 episode 名称，例如 'episode_0'
                episode_name = cloud_file.name.replace('_pointclouds.npy', '')
                joint_file = jp / f"{episode_name}.hdf5"
                
                if not joint_file.exists():
                    print(f"[警告] 找不到对应的关节文件: {joint_file}，跳过该样本")
                    continue
                
                # 1. 加载点云 (支持 numpy 对象数组)
                try:
                    cloud_data = np.load(str(cloud_file), allow_pickle=True)
                    if cloud_data.dtype == 'O':
                        # 如果是 numpy object 数组（通常是由 list 转化而来），我们需要把它转化为普通的 float 数组
                        cloud_data = np.array([np.array(item, dtype=np.float32) for item in cloud_data])
                    else:
                        cloud_data = cloud_data.astype(np.float32)
                    cloud_tensor = torch.from_numpy(cloud_data).float()
                except Exception as e:
                    print(f"读取点云失败 {cloud_file}: {e}")
                    continue

                # 加载关节驱动值（Inspire/L10 hand_joints / L21 hand_qpos）
                try:
                    with h5py.File(str(joint_file), 'r') as f:
                        key, hand_type, expected_dim = resolve_hand_joint_key(f)
                        joint_data = f[key][:].astype(np.float32)

                    if joint_data.ndim != 2:
                        raise ValueError(
                            f"{joint_file}[{key}] 期望 [T,J]，实际 shape={joint_data.shape}"
                        )
                    j = int(joint_data.shape[-1])
                    if j != expected_dim:
                        raise ValueError(
                            f"{joint_file}[{key}] 末维应为 {expected_dim}（{hand_type}），实际 J={j}"
                        )
                    if self.hand_type is None:
                        self.hand_type = hand_type
                        self.joint_dim = j
                    elif self.hand_type != hand_type or self.joint_dim != j:
                        raise ValueError(
                            f"手型不一致：已锁定 {self.hand_type}/J={self.joint_dim}，"
                            f"但 {joint_file} 为 {hand_type}/J={j}"
                        )

                    # 手型感知的单位转换：
                    #   Inspire 0~1000 -> [-pi/2, pi/2]（避免有界端点 sin/cos 重合）
                    #   L10/L21 0~255 -> [-pi, pi]
                    joint_data = joints_to_radians(
                        joint_data,
                        self.joint_unit,
                        hand_type=hand_type,
                        joint_dim=j,
                    ).astype(np.float32)

                    # 对关节数据进行 Savitzky-Golay 滤波平滑，消除真实数据采集时的高频噪声和锯齿
                    if self.smooth_window > 0 and joint_data.shape[0] >= self.smooth_window:
                        window_length = self.smooth_window
                        if window_length % 2 == 0:
                            window_length += 1 # window_length 必须为奇数
                        joint_data = savgol_filter(joint_data, window_length, self.smooth_poly, axis=0)
                        joint_data = joint_data.astype(np.float32)

                    joint_tensor = torch.from_numpy(joint_data)
                except Exception as e:
                    print(f"[错误] 加载关节数据失败 {joint_file}: {e}")
                    continue

                # 确保时序维度一致
                min_T = min(cloud_tensor.shape[0], joint_tensor.shape[0])
                self.clouds.append(cloud_tensor[:min_T])
                self.joints.append(joint_tensor[:min_T])
        else:
            raise ValueError("clouds_path 和 joints_path 必须同时为目录")

        assert len(self.clouds) == len(self.joints), "点云与关节数据未能配对"
        assert len(self.clouds) > 0, "未找到任何配对的训练数据"
        if self.hand_type is not None:
            print(
                f"[{progress_desc}] 手型={self.hand_type.upper()} "
                f"J={self.joint_dim} episodes={len(self.clouds)}"
            )

        # ── 构建滑动窗口索引 ──────────────────────────────────────────────
        # 旧实现每个 episode 只取 1 个窗口（样本量=episode 数，约 239），
        # 导致每 epoch 步数极少、动态内容欠采样。这里改为按 frame_stride 的
        # 滑动窗口，样本量提升到上万；同时 frame_stride 与 Stage1 encoder
        # 训练时一致（默认 2），保证喂给 encoder 的时序速度匹配。
        self.index = []
        # 索引统一按最大/指定 stride 的长度约束建立，
        # 使 stride 1/2 共用同一批合法 episode 与 start，避免数据分布不同。
        required = (self.seq_len - 1) * self.index_frame_stride + 1
        self.skipped_short_episodes = 0
        for ei, c in enumerate(self.clouds):
            T = c.shape[0]
            if T >= required:
                for start in range(0, T - required + 1):
                    self.index.append((ei, start))
            else:
                # 与 Stage1 一致：短序列不用循环 repeat/stride=1 伪造时序。
                self.skipped_short_episodes += 1
        assert len(self.index) > 0, "未能构建任何滑动窗口样本"
        print(
            f"[{progress_desc}] 窗口={len(self.index)} "
            f"(seq_len={self.seq_len}, sample_stride={self.frame_stride_choices}, "
            f"index_stride={self.index_frame_stride}, "
            f"跳过过短 episode={self.skipped_short_episodes})"
        )

    def __len__(self) -> int:
        return len(self.index)
    
    def __getitem__(self, idx: int):
        epi_idx, start = self.index[idx]
        cloud = self.clouds[epi_idx].float()
        joint = self.joints[epi_idx].float()

        T = cloud.shape[0]
        current_stride = self.frame_stride
        if self.augment and len(self.frame_stride_choices) > 1:
            choice_idx = int(torch.randint(len(self.frame_stride_choices), (1,)).item())
            current_stride = self.frame_stride_choices[choice_idx]
        required = (self.seq_len - 1) * current_stride + 1

        # 时序维度处理：按 frame_stride 抽取 seq_len 帧（点云与关节同步切片）
        if start >= 0 and T >= required:
            sl = slice(start, start + required, current_stride)
            cloud = cloud[sl]
            joint = joint[sl]
        else:
            raise RuntimeError(
                f"索引与数据长度不一致: episode={epi_idx}, start={start}, "
                f"T={T}, required={required}"
            )

        # 点数维度:下采样
        N = cloud.shape[1]
        if N > self.num_points:
            if self.augment:
                # 训练集：随机丢弃点
                idx_pts = torch.randperm(N)[:self.num_points]
            else:
                # 验证集：固定取前 num_points 个点，或者固定步长采样
                # 避免每次取到不同的点导致特征 Z 产生微小抖动
                idx_pts = torch.linspace(0, N - 1, steps=self.num_points).long()
            cloud = cloud[:, idx_pts, :]
        elif N < self.num_points:
            if self.augment:
                idx_pts = torch.randint(0, N, (self.num_points,))
            else:
                # 验证集循环补齐
                idx_pts = torch.arange(self.num_points) % N
            cloud = cloud[:, idx_pts, :]

        # [重要修复] 点云中心化与归一化（全局序列归一化 Global Sequence Normalization）
        # 必须与当前 emEncoder 训练时保持绝对一致。
        # stride 切片可能非连续，reshape 兼容 contiguous/non-contiguous tensor。
        global_centroid = cloud.reshape(-1, 3).mean(dim=0, keepdim=True)  # [1, 3]
        cloud = cloud - global_centroid                                # [T, N, 3]

        distances = torch.sqrt(torch.sum(cloud**2, dim=-1))            # [T, N]
        global_max_dist = torch.clamp(torch.max(distances), min=1e-6)  # 标量
        cloud = cloud / global_max_dist                                # [T, N, 3]

        # 数据增强
        if self.augment:
            cloud = self._augment_cloud(cloud)

        return cloud, joint  # [T, N, 3], [T, J]
    
    @staticmethod
    def _augment_cloud(cloud: torch.Tensor) -> torch.Tensor:
        '''
        点云增强：仅小幅抖动。
        注意：取消了旋转增强，因为冻结的 Encoder 可能对旋转不具备不变性，
        随意旋转会导致提取的特征 Z 变成乱码（分布外数据）。
        '''
        cloud = cloud + torch.randn_like(cloud) * 0.005
        return cloud
    

# ══════════════════════════════════════════════════════════════════════════════
# 损失计算
# ══════════════════════════════════════════════════════════════════════════════

def weighted_angle_reconstruction_loss(
        pred_sincos: torch.Tensor,
        gt_theta: torch.Tensor,
        joint_weight: torch.Tensor,
        joint_mask: torch.Tensor,
) -> torch.Tensor:
    """
    逐关节加权角度重建损失。

    思路：
        1. 将预测的 sin/cos 还原成角度；
        2. 在角度空间中计算最短角度差（wrap 到 [-pi, pi]）；
        3. 根据训练集统计得到的逐关节权重做加权；
        4. 再做均方误差平均。
    """
    pred_theta = sincos_to_angles(pred_sincos)  # [B, T, J]
    angle_diff = (pred_theta - gt_theta + torch.pi) % (2 * torch.pi) - torch.pi

    weight = joint_weight.to(gt_theta.device).view(1, 1, -1)  # [1, 1, J]
    mask = joint_mask.to(gt_theta.device).view(1, 1, -1).float()  # [1, 1, J]
    weighted_error = (angle_diff ** 2) * weight * mask
    denom = torch.clamp(
        mask.sum() * gt_theta.shape[0] * gt_theta.shape[1],
        min=1.0
    )
    return weighted_error.sum() / denom


def variance_matching_loss(
        pred_sincos: torch.Tensor,
        gt_theta: torch.Tensor,
        dynamic_joint_mask: torch.Tensor,
        eps: float = 1e-4,
) -> torch.Tensor:
    """
    高动态关节的方差匹配损失。

    只约束“该动起来的关节”在时间维上的波动幅度不要塌成直线，
    作为主重建损失之外的轻量辅助项。
    """
    pred_theta = sincos_to_angles(pred_sincos)  # [B, T, J]
    mask = dynamic_joint_mask.to(gt_theta.device)
    if mask.sum() == 0:
        return pred_theta.new_tensor(0.0)

    pred_dyn = pred_theta[:, :, mask]  # [B, T, K]
    gt_dyn = gt_theta[:, :, mask]

    pred_var = torch.var(pred_dyn, dim=1, unbiased=False)  # [B, K]
    gt_var = torch.var(gt_dyn, dim=1, unbiased=False)      # [B, K]

    return torch.mean((torch.log(pred_var + eps) - torch.log(gt_var + eps)) ** 2)


def build_joint_mask(joint_dim: int, ignored_joint_ids: str) -> torch.Tensor:
    mask = torch.ones(joint_dim, dtype=torch.bool)
    ignored = []
    for token in ignored_joint_ids.split(','):
        token = token.strip()
        if not token:
            continue
        idx = int(token)
        if 0 <= idx < joint_dim:
            mask[idx] = False
            ignored.append(idx)
    print(f"[关节] 忽略 loss 的关节: {ignored}")
    print(f"       有效关节数: {int(mask.sum().item())}/{joint_dim}")
    return mask

def decoder_train_loss(
        pred_sincos: torch.Tensor,
        gt_theta: torch.Tensor,
        joint_weight: torch.Tensor,
        joint_mask: torch.Tensor,
        dynamic_joint_mask: torch.Tensor,
        z_robot: torch.Tensor = None,
        # 训练损失权重
        w_recon: float = 1.0,
        w_var: float = 0.02,
        w_tc: float = 0.0,
        w_smooth: float = 0.0,
) -> dict:
    """
    Loss：
        L_recon   关节角度重建精度（主监督）
        L_var     高动态关节时间方差匹配，抑制“直线解”
        L_tc      latent 时序对比正则（联合训练时保持 Stage1 表征结构，防坍塌）
        L_smooth  latent 时序平滑正则（防止解冻后特征突变）

    说明：
        L_tc / L_smooth 仅在「解冻 encoder 的联合训练」中起作用。
        关节重建梯度会微调 encoder 补充姿态细节，而这两个正则
        把 latent 约束在 Stage1 学到的流形附近，避免表征坍塌或过度对齐。
    """

    l_recon = weighted_angle_reconstruction_loss(
        pred_sincos=pred_sincos,
        gt_theta=gt_theta,
        joint_weight=joint_weight,
        joint_mask=joint_mask,
    )
    l_var = variance_matching_loss(pred_sincos, gt_theta, dynamic_joint_mask)

    if z_robot is not None and (w_tc > 0 or w_smooth > 0):
        l_tc = temporal_contrastive_loss(z_robot)
        l_smooth = temporal_smoothness_loss(z_robot)
    else:
        l_tc = pred_sincos.new_tensor(0.0)
        l_smooth = pred_sincos.new_tensor(0.0)

    total = w_recon * l_recon + w_var * l_var + w_tc * l_tc + w_smooth * l_smooth

    return {
        'total': total,
        'recon': l_recon.detach(),
        'var': l_var.detach(),
        'tc': l_tc.detach(),
        'smooth': l_smooth.detach(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 主训练循环
# ══════════════════════════════════════════════════════════════════════════════
def resolve_stage1_compat_args(args: argparse.Namespace) -> dict:
    """从 Stage1 checkpoint 恢复 encoder 结构与默认序列长度。"""
    ckpt_path = Path(args.encoder_ckpt)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"找不到预训练 encoder 权重：{ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    raw = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    if hasattr(raw, "__dict__") and not isinstance(raw, dict):
        raw = vars(raw)
    ckpt_args = raw if isinstance(raw, dict) else {}

    structural = {
        "point_feat_dim": 256,
        "latent_dim": 128,
        "temporal_layers": 2,
        "temporal_heads": 4,
    }
    for name, fallback in structural.items():
        trained_value = int(ckpt_args.get(name, fallback))
        requested = getattr(args, name)
        if requested is None:
            setattr(args, name, trained_value)
        elif int(requested) != trained_value:
            raise ValueError(
                f"--{name}={requested} 与 Stage1 checkpoint 的 {trained_value} 不一致"
            )

    trained_seq_len = int(ckpt_args.get("seq_len", 20))
    if args.seq_len is None:
        args.seq_len = trained_seq_len
    elif int(args.seq_len) != trained_seq_len:
        print(
            f"[提示] Stage2 seq_len={args.seq_len}，Stage1 seq_len={trained_seq_len}。"
            " Encoder 支持可变长度，但建议 Stage2/Stage3 保持一致。"
        )

    if args.encoder_dropout is None:
        args.encoder_dropout = float(ckpt_args.get("dropout", 0.1))
    return ckpt_args


def train(args: argparse.Namespace) -> None:
    resolve_stage1_compat_args(args)
    # ------------------
    # 创建带时间戳的统一保存目录
    # ------------------
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_dir = Path(args.ckpt_dir) / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 启动日志记录器
    log_file = save_dir / "log.txt"
    sys.stdout = Logger(filename=str(log_file))
    print(f"[{timestamp}] 启动训练，所有文件将保存在: {save_dir}")
    print(
        f"[Stage1兼容] seq_len={args.seq_len}, point_feat_dim={args.point_feat_dim}, "
        f"latent_dim={args.latent_dim}, temporal={args.temporal_layers}x{args.temporal_heads}, "
        f"encoder_dropout={args.encoder_dropout}"
    )
    
    # 初始化 Weights & Biases (W&B)
    wandb.login(key=WANDB_API_KEY)
    wandb.init(
        project="Mymimic-emDecoder", 
        name=timestamp, 
        config=vars(args),
        dir=str(save_dir)
    )
    print(f'[日志] W&B 日志记录已启动')

    # ----设备----
    device = torch.device(
        'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    )
    print(f'[设备] {device}')

    # ----数据集----
    print('[数据] 加载中......')
    train_stride_choices = parse_frame_stride_choices(args.train_frame_strides)
    if args.frame_stride < max(train_stride_choices):
        raise ValueError(
            f"--frame_stride={args.frame_stride} 必须 >= --train_frame_strides "
            f"的最大值 {max(train_stride_choices)}"
        )
    if args.val_frame_stride > args.frame_stride:
        raise ValueError(
            f"--val_frame_stride={args.val_frame_stride} 不能大于索引约束 "
            f"--frame_stride={args.frame_stride}"
        )
    print(
        f"[时间采样] train stride={train_stride_choices} 等概率随机；"
        f"val stride={args.val_frame_stride}；index stride={args.frame_stride}"
    )
    
    if args.robot_val_clouds and args.robot_val_joints:
        robot_train = RobotHandDataset(
            clouds_path=args.robot_clouds,
            joints_path=args.robot_joints,
            seq_len=args.seq_len,
            num_points=args.num_points,
            augment=True,
            smooth_window=args.smooth_window,
            smooth_poly=args.smooth_poly,
            joint_unit=args.joint_unit,
            frame_stride=args.frame_stride,
            frame_stride_choices=train_stride_choices,
            index_frame_stride=args.frame_stride,
            show_progress=True,
            progress_desc="加载训练集",
        )
        robot_val = RobotHandDataset(
            clouds_path=args.robot_val_clouds,
            joints_path=args.robot_val_joints,
            seq_len=args.seq_len,
            num_points=args.num_points,
            augment=False,
            smooth_window=args.smooth_window,
            smooth_poly=args.smooth_poly,
            joint_unit=args.joint_unit,
            frame_stride=args.val_frame_stride,
            index_frame_stride=args.frame_stride,
            show_progress=True,
            progress_desc="加载验证集",
        )
    else:
        robot_full = RobotHandDataset(
            clouds_path=args.robot_clouds,
            joints_path = args.robot_joints,
            seq_len     = args.seq_len,
            num_points  = args.num_points,
            augment     = True,
            smooth_window=args.smooth_window,
            smooth_poly=args.smooth_poly,
            joint_unit=args.joint_unit,
            frame_stride=args.frame_stride,
            frame_stride_choices=train_stride_choices,
            index_frame_stride=args.frame_stride,
            show_progress=True,
            progress_desc="加载完整数据集",
        )
        n_val = max(1, int(len(robot_full) * args.val_ratio))
        n_train = len(robot_full) - n_val
        robot_train, robot_val = random_split(
            robot_full, [n_train, n_val],
            generator = torch.Generator().manual_seed(42),
        )
        print("[警告] 使用 random_split 拆分验证集，验证集将带有数据增强（不推荐）")

    # 动态获取真实的关节数（J）以覆盖默认的 joint_dim 参数
    hand_type = getattr(robot_train, "hand_type", None)
    if hasattr(robot_train, "dataset"):  # random_split Subset
        hand_type = getattr(robot_train.dataset, "hand_type", hand_type)
    if len(robot_train) > 0:
        _, sample_joint = robot_train[0]
        actual_joint_dim = sample_joint.shape[-1]
        if actual_joint_dim != args.joint_dim:
            print(f"[提示] 根据数据动态更新关节数: {args.joint_dim} -> {actual_joint_dim}")
            args.joint_dim = actual_joint_dim
    if hand_type is None:
        hand_type = hand_type_from_joint_dim(args.joint_dim)
    args.hand_type = hand_type
    print(f"[数据] 手型={hand_type.upper()} joint_dim={args.joint_dim}")

    val_hand = getattr(robot_val, "hand_type", None)
    if hasattr(robot_val, "dataset"):
        val_hand = getattr(robot_val.dataset, "hand_type", val_hand)
    if val_hand is not None and hand_type is not None and val_hand != hand_type:
        raise ValueError(f"训练/验证手型不一致：train={hand_type} val={val_hand}")

    # L10 无预留关节；未显式传入时按手型自动
    args.ignored_joint_ids = resolve_ignored_joint_ids(
        args.ignored_joint_ids, hand_type=hand_type, joint_dim=args.joint_dim
    )
    args.fixed_joint_ids = resolve_fixed_joint_ids(
        args.fixed_joint_ids, hand_type=hand_type, joint_dim=args.joint_dim
    )
    fixed_joint_ids = parse_joint_id_list(args.fixed_joint_ids, args.joint_dim)
    fixed_joint_rad = float(joints_to_radians(
        np.asarray([args.fixed_joint_raw255], dtype=np.float32),
        "raw",
        hand_type=hand_type,
        joint_dim=args.joint_dim,
    )[0])
    if fixed_joint_ids:
        print(
            f"[关节] 训练时固定为 raw_driver={args.fixed_joint_raw255} "
            f"(rad={fixed_joint_rad:.6f}) 的关节: {fixed_joint_ids}"
        )

    # ----加载逐关节统计量（用于加权重建损失；默认从数据集根目录读取）----
    joint_stats_path = resolve_joint_stats_path(
        args.robot_joints,
        args.joint_dim,
        explicit=args.joint_stats_path,
        legacy_task=CURRENT_TASK,
        legacy_utils_dir=Path(MYMIMIC_PATH) / "mimic_utils",
    )
    args.joint_stats_path = str(joint_stats_path)
    joint_stats = np.load(joint_stats_path)
    if 'std' not in joint_stats:
        raise KeyError(f"统计文件中缺少 'std' 键：{joint_stats_path}")
    joint_std_np = joint_stats['std'].astype(np.float32)
    if joint_std_np.shape[0] != args.joint_dim:
        raise ValueError(
            f"关节统计维度与数据不一致：stats={joint_std_np.shape[0]}, data={args.joint_dim}。"
            f" 请用当前 Inspire/L10/L21 数据重新运行 compute_joint_stats.py。"
        )
    safe_std_np = np.clip(joint_std_np, a_min=args.std_floor, a_max=None)
    joint_mask = build_joint_mask(args.joint_dim, args.ignored_joint_ids).to(device)
    std_min = float(safe_std_np.min())
    std_max = float(safe_std_np.max())
    if std_max - std_min < 1e-8:
        joint_weight_np = np.ones_like(safe_std_np, dtype=np.float32)
    else:
        dynamic_score = (safe_std_np - std_min) / (std_max - std_min)
        joint_weight_np = (
            args.joint_weight_min
            + (args.joint_weight_max - args.joint_weight_min) * dynamic_score
        ).astype(np.float32)
    active_mask_np = joint_mask.cpu().numpy().astype(bool)
    joint_weight_np[~active_mask_np] = 0.0
    active_mean = float(np.mean(joint_weight_np[active_mask_np])) if np.any(active_mask_np) else 1.0
    joint_weight_np = joint_weight_np / max(active_mean, 1e-8)
    joint_weight = torch.from_numpy(joint_weight_np).to(device)
    dynamic_joint_mask = torch.zeros(args.joint_dim, dtype=torch.bool)
    active_joint_ids = np.where(active_mask_np)[0]
    dynamic_candidates = active_joint_ids[
        joint_std_np[active_joint_ids] >= args.dynamic_std_min
    ]
    if dynamic_candidates.size > 0:
        top_k = min(args.dynamic_topk, dynamic_candidates.size)
        active_std = joint_std_np[dynamic_candidates]
        top_local_idx = np.argsort(active_std)[-top_k:]
        dynamic_joint_ids = dynamic_candidates[top_local_idx]
        dynamic_joint_mask[dynamic_joint_ids] = True
    dynamic_joint_mask = dynamic_joint_mask.to(device)
    print(f"[统计] 已加载逐关节统计文件: {joint_stats_path}")
    print(f"       std 范围: min={safe_std_np.min():.6f}, max={safe_std_np.max():.6f}")
    print(f"       weight 范围: min={joint_weight.min().item():.6f}, max={joint_weight.max().item():.6f}")
    print(f"       高动态关节: {torch.nonzero(dynamic_joint_mask, as_tuple=False).squeeze(1).tolist()}")

    train_loader = DataLoader(
        robot_train,
        batch_size  = args.batch_size,
        shuffle     = True,
        drop_last   = True,
        num_workers = 2,
        pin_memory  = True,
    )
    val_loader = DataLoader(
        robot_val,
        batch_size  = args.batch_size,
        shuffle     = False,  # [重要修复] 验证集绝对不能 shuffle，否则每次 Batch 里的负样本组合不同，导致 val_tc 剧烈波动！
        drop_last   = False,
        num_workers = 2,
        pin_memory  = True,
    )
    print(f'  灵巧手 训练 {len(robot_train)} / 验证 {len(robot_val)} 条')

    # ----模型构建----
    # encoder的参数
    encoder_cfg = dict(
        point_feat_dim = args.point_feat_dim,
        latent_dim = args.latent_dim,
        temporal_layers = args.temporal_layers,
        temporal_heads = args.temporal_heads,
        dropout = args.encoder_dropout
    )
    # decoder的参数
    decoder_cfg = dict(
        latent_dim  = args.latent_dim,
        joint_dim   = args.joint_dim,
        num_layers  = args.dec_layers,
        num_heads   = args.dec_heads,
        window_size = args.window_size,
        dropout     = args.dropout
    )
    system = HandMotionSystem(encoder_cfg, decoder_cfg).to(device)

    # ------------------
    # Encoder 冻结 / 解冻（--freeze_encoder / --no-freeze_encoder）
    # 解冻：带约束联合训练，极小 lr 微调 latent；冻结：只训 decoder。
    # ------------------
    if args.freeze_encoder:
        print("[模型] 冻结 emEncoder 权重，关闭特征空间协同微调")
        for param in system.encoder.parameters():
            param.requires_grad = False
    else:
        print("[模型] 解冻 emEncoder 权重，开启带约束的联合训练（路线 B）")
        for param in system.encoder.parameters():
            param.requires_grad = True

    encoder_lr = 0.0 if args.freeze_encoder else args.lr_encoder

    # ----加载stage1中训练的encoder权重----
    enc_ckpt_path = Path(args.encoder_ckpt)
    if not enc_ckpt_path.exists():
        raise FileNotFoundError(
            f'找不到预训练 encoder 权重：{args.encoder_ckpt}\n'
        )
    enc_ckpt = torch.load(enc_ckpt_path, map_location=device, weights_only=True)
    system.encoder.load_state_dict(enc_ckpt['model'])
    print(f'[模型] 已加载预训练 encoder: {args.encoder_ckpt}')

    enc_p = sum(p.numel() for p in system.encoder.parameters())
    dec_p = sum(p.numel() for p in system.decoder.parameters())
    print(f'       emEncoder {enc_p:,} 参数  lr={encoder_lr:.1e}')
    print(f'       Decoder   {dec_p:,} 参数  lr={args.lr_decoder:.1e}  （从零训练）')

    # ----差异化学习率优化器----
    optimizer = system.get_optimizer(
        encoder_lr=encoder_lr,
        decoder_lr=args.lr_decoder,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr_decoder*0.01
    )

    # ----从断点恢复----
    start_epoch = 0
    best_val = float('inf')
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        system.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch=ckpt['epoch'] + 1
        best_val = ckpt.get('best_val', float('inf'))
        print(f'[恢复] 从 epoch {start_epoch} 继续，历史最优 val recon = {best_val:.6f}')

    # ----损失权重----
    loss_weights = dict(
        w_recon = args.w_recon,
        w_var = args.w_var,
        w_tc = args.w_tc,
        w_smooth = args.w_smooth,
    )
    # ----主循环----
    print(f'\n[训练] 开始，共 {args.epochs} 个epoch\n')
    loss_keys = ['total', 'recon', 'var', 'tc', 'smooth']

    epoch_pbar = tqdm(range(start_epoch, args.epochs), desc="Training Progress")
    for epoch in epoch_pbar:
        # =========== 训练阶段 ===========
        system.train()
        # [关键] 冻结的 encoder 必须保持 eval 模式：
        # 否则其 dropout 在训练时仍开启，会让同一帧的 latent 每步抖动，
        # decoder 面对“目标 latent 一直变”只能退化成输出均值 → 关节幅度塌缩。
        # 这样才能与 latent 探针成功时（eval、干净 latent）的条件一致。
        if args.freeze_encoder:
            system.encoder.eval()
        train_metrics = {k: 0.0 for k in loss_keys}
        t0 = time.time()

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:03d}/{args.epochs} [Train]", leave=False)
        for x_robot, gt_theta in train_pbar:
            x_robot = x_robot.to(device, non_blocking=True)
            gt_theta = gt_theta.to(device, non_blocking=True)
            if fixed_joint_ids:
                gt_theta = apply_fixed_joint_values(
                    gt_theta, fixed_joint_ids, fixed_joint_rad
                )

            out = system(x_robot=x_robot)
            pred_sincos = out['pred_sincos']

            losses = decoder_train_loss(
                pred_sincos=pred_sincos,
                gt_theta=gt_theta,
                joint_weight=joint_weight,
                joint_mask=joint_mask,
                dynamic_joint_mask=dynamic_joint_mask,
                z_robot=out['z_robot'],
                **loss_weights
            )

            optimizer.zero_grad()
            losses['total'].backward()
            # 梯度裁剪：联合训练下 encoder 也可训练，裁剪整个 system 防止梯度爆炸
            nn.utils.clip_grad_norm_(system.parameters(), max_norm=1.0)
            optimizer.step()   # 更新参数

            for k in loss_keys:
                train_metrics[k] += (losses[k].detach().item()
                                     if k == 'total' else losses[k].item())
                
            # 实时更新进度条显示
            train_pbar.set_postfix({
                'loss': f"{losses['total'].item():.4f}",
                'recon': f"{losses['recon'].item():.4f}",
                'var': f"{losses['var'].item():.4f}"
            })
        
        n_steps = len(train_loader)
        for k in loss_keys:
            train_metrics[k] /= n_steps

        # =========== 验证阶段 ===========
        system.eval()
        val_metrics = {k: 0.0 for k in loss_keys}
        n_val_steps = 0

        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1:03d}/{args.epochs} [Val]", leave=False)
        with torch.no_grad():
            for x_robot, gt_theta in val_pbar:
                x_robot = x_robot.to(device, non_blocking=True)
                gt_theta = gt_theta.to(device, non_blocking=True)
                if fixed_joint_ids:
                    gt_theta = apply_fixed_joint_values(
                        gt_theta, fixed_joint_ids, fixed_joint_rad
                    )

                out = system(x_robot=x_robot)
                pred_sincos = out['pred_sincos']

                losses = decoder_train_loss(
                    pred_sincos=pred_sincos,
                    gt_theta=gt_theta,
                    joint_weight=joint_weight,
                    joint_mask=joint_mask,
                    dynamic_joint_mask=dynamic_joint_mask,
                    z_robot=out['z_robot'],
                    **loss_weights
                )
                for k in loss_keys:
                    val_metrics[k] += losses[k].item()
                n_val_steps += 1
                
                # 实时更新进度条显示
                val_pbar.set_postfix({
                    'loss': f"{losses['total'].item():.4f}",
                    'recon': f"{losses['recon'].item():.4f}"
                })
        if n_val_steps > 0:   # 计算epoch平均损失
            for k in loss_keys:
                val_metrics[k] /= n_val_steps

        scheduler.step()
        elapsed = time.time() - t0

        # 将每个 epoch 的总结信息显示在主进度条的后缀中，避免刷屏
        epoch_pbar.set_postfix({
            'T_tot': f"{train_metrics['total']:.4f}",
            'T_rec': f"{train_metrics['recon']:.4f}",
            'V_tot': f"{val_metrics['total']:.4f}",
            'V_rec': f"{val_metrics['recon']:.4f}",
            'lr': f"{scheduler.get_last_lr()[1]:.2e}",
            'time': f"{elapsed:.1f}s"
        })

        # 记录到 W&B
        wandb.log({
            'Loss/train_total': train_metrics["total"],
            'Loss/train_recon': train_metrics["recon"],
            'Loss/train_var': train_metrics["var"],
            'Loss/train_tc': train_metrics["tc"],
            'Loss/train_smooth': train_metrics["smooth"],
            'Loss/val_total': val_metrics["total"],
            'Loss/val_recon': val_metrics["recon"],
            'Loss/val_var': val_metrics["var"],
            'Loss/val_tc': val_metrics["tc"],
            'Loss/val_smooth': val_metrics["smooth"],
            'LR/learning_rate_enc': scheduler.get_last_lr()[0],
            'LR/learning_rate_dec': scheduler.get_last_lr()[1],
            'epoch': epoch
        })

        # ---- Checkpoint 保存 ----
        # 以 val recon loss 选最优 checkpoint（最直接反映解码质量）
        val_recon = val_metrics['recon']
        state = {
            'epoch':     epoch,
            'model':     system.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_val':  best_val,
            'args':      vars(args),
            # 单独保存子模块，方便后续只取其中一个
            'encoder':   system.encoder.state_dict(),
            'decoder':   system.decoder.state_dict(),
        }
        torch.save(state, save_dir / 'decoder_last.pt')
 
        if val_recon < best_val:
            best_val = val_recon
            state['best_val'] = best_val
            torch.save(state, save_dir / 'decoder_best.pt')
            # 使用 tqdm.write 避免破坏进度条布局
            tqdm.write(f'  ★ Epoch {epoch+1:03d} 新的最优 val recon = {best_val:.6f}，已保存 decoder_best.pt')
 
    print(f'\n[完成] 最优 val recon loss = {best_val:.6f}')
    print(f'       权重已保存至 {save_dir}')
    print('\n推理加载方式：')
    print('  ckpt = torch.load("checkpoints/decoder_best.pt")')
    print('  system.load_state_dict(ckpt["model"])')
    wandb.finish()


# ══════════════════════════════════════════════════════════════════════════════
# CLI 入口
# ══════════════════════════════════════════════════════════════════════════════
 
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='阶段二：训练 Decoder（recon 主导 + 动态方差辅助）',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
 
    # ── 数据 ─────────────────────────────────────────────────────────────────
    data = p.add_argument_group('数据')
    data.add_argument('--robot_clouds', type=str, default=ROBOT_PC_PATH_TRAIN,
                      help='灵巧手点云目录（须与 HDF5 手型一致）')
    data.add_argument('--robot_joints', type=str, default=ROBOT_HDF5_PATH_TRAIN,
                      help='关节 HDF5（Inspire: hand_joints[T,6]；L10: [T,10]；L21: hand_qpos[T,25]）')
    data.add_argument('--robot_val_clouds', type=str, default=ROBOT_PC_PATH_VAL,
                      help='灵巧手验证集点云数据路径')
    data.add_argument('--robot_val_joints', type=str, default=ROBOT_HDF5_PATH_VAL,
                      help='灵巧手验证集关节角度数据路径')
    data.add_argument('--seq_len',      type=int, default=None,
                      help='默认自动读取 Stage1 checkpoint 的 seq_len')
    data.add_argument('--num_points',   type=int, default=256)
    data.add_argument('--joint_dim',    type=int, default=None,
                      help='默认从 HDF5 自动推断（Inspire=6 / L10=10 / L21=25）')
    data.add_argument('--joint_unit',   type=str, default='raw',
                      choices=['radians', 'degrees', 'raw', 'raw255', 'raw1000'],
                      help='原始关节单位；raw 自动使用 Inspire=0~1000、L10/L21=0~255')
    data.add_argument(
        '--joint_stats_path',
        type=str,
        default=None,
        help='逐关节统计 npz；默认从 --robot_joints 所在数据集根目录读取 '
             'joint_stats_l{J}.npz（或 joint_stats.npz）；仍兼容 mimic_utils/{task}_*.npz',
    )
    data.add_argument(
        '--ignored_joint_ids',
        type=str,
        default=None,
        help='loss 中忽略的关节编号；默认 L21 忽略预留关节，Inspire/L10 不忽略。'
             f' L21 默认: {L21_IGNORED_JOINT_IDS_STR}',
    )
    data.add_argument(
        '--fixed_joint_ids',
        type=str,
        default=None,
        help='训练时将 GT 写成常数的关节编号；默认 L10 固定 1(thumb_cmc_yaw)，L21 不固定。'
             ' 传空串可关闭。',
    )
    data.add_argument(
        '--fixed_joint_raw255',
        type=float,
        default=0.0,
        help='固定关节的原始驱动值（旧参数名保留兼容；Inspire 按 0~1000）',
    )
    data.add_argument('--val_ratio',    type=float, default=0.1,
                      help='(已弃用) 如果未指定 val 路径则使用的验证集划分比例')
    data.add_argument('--smooth_window', type=int, default=5,
                      help='Savitzky-Golay 滤波平滑的窗口大小，去除高频锯齿噪声')
    data.add_argument('--smooth_poly',  type=int, default=2,
                      help='Savitzky-Golay 滤波的多项式阶数')
    data.add_argument('--frame_stride', type=int, default=2,
                      help='建立训练/验证窗口的最大步长约束；默认 2，与 Stage1 base stride 一致')
    data.add_argument('--train_frame_strides', type=str, default='1,2',
                      help='Stage2 训练时等概率随机抽取的 stride，逗号分隔')
    data.add_argument('--val_frame_stride', type=int, default=1,
                      help='Stage2 验证固定 stride；默认 1，对齐 Stage3/在线控制频率')
 
    # ── 模型结构（须与阶段一保持一致）────────────────────────────────────────
    model = p.add_argument_group('模型（须与 train_encoder.py 一致）')
    model.add_argument('--point_feat_dim',  type=int,   default=None,
                       help='默认从 Stage1 checkpoint 读取')
    model.add_argument('--latent_dim',      type=int,   default=None,
                       help='默认从 Stage1 checkpoint 读取')
    model.add_argument('--temporal_layers', type=int,   default=None,
                       help='默认从 Stage1 checkpoint 读取')
    model.add_argument('--temporal_heads',  type=int,   default=None,
                       help='默认从 Stage1 checkpoint 读取')
    model.add_argument('--encoder_dropout', type=float, default=None,
                       help='默认从 Stage1 checkpoint 读取；冻结 encoder 时 eval 下不生效')
 
    # ── Decoder 结构 ──────────────────────────────────────────────────────────
    dec = p.add_argument_group('Decoder 结构')
    dec.add_argument('--dec_layers',   type=int, default=3)
    dec.add_argument('--dec_heads',    type=int, default=4)
    dec.add_argument('--window_size',  type=int, default=10,
                     help='历史窗口帧数 W（影响推理时延）')
    dec.add_argument('--dropout',      type=float, default=0.1,
                     help='emDecoder dropout（与 Stage1 encoder dropout 独立）')
 
    # ── 训练超参 ──────────────────────────────────────────────────────────────
    train_g = p.add_argument_group('训练')
    train_g.add_argument('--epochs',       type=int,   default=100)
    train_g.add_argument('--batch_size',   type=int,   default=16)
    train_g.add_argument('--lr_encoder',   type=float, default=1e-5,
                         help='emEncoder 学习率（联合训练用极小 lr 微调，补充姿态细节）')
    train_g.add_argument(
        '--freeze_encoder',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='是否冻结 Stage1 emEncoder：--freeze_encoder 只训 decoder；'
             '--no-freeze_encoder 才解冻做联合微调（默认冻结，保护 Stage1 跨域对齐）',
    )
    train_g.add_argument('--lr_decoder',   type=float, default=3e-4,
                         help='Decoder 学习率')
    train_g.add_argument('--weight_decay', type=float, default=1e-4)
    train_g.add_argument('--w_recon',      type=float, default=1.0,
                         help='重建损失权重（阶段二默认由 recon 主导）')
    train_g.add_argument('--std_floor',    type=float, default=0.03,
                         help='计算逐关节权重时 std 的下限，防止几乎恒定关节权重退化')
    train_g.add_argument('--joint_weight_min', type=float, default=0.2,
                         help='静态关节的最小权重')
    train_g.add_argument('--joint_weight_max', type=float, default=3.0,
                         help='高动态关节的最大权重')
    train_g.add_argument('--dynamic_topk', type=int, default=10,
                         help='方差匹配辅助项中使用的高动态关节数量（从有效关节里按 std 选 Top-K）')
    train_g.add_argument('--dynamic_std_min', type=float, default=0.01,
                         help='进入方差匹配辅助项的最小关节 std；排除 Inspire 等手型中的恒定驱动轴')
    train_g.add_argument('--w_var', type=float, default=0.05,
                         help='高动态关节时间方差匹配损失权重，用于抑制直线解')
    train_g.add_argument('--w_tc', type=float, default=0.0,
                         help='latent 时序对比正则权重（联合训练防表征坍塌，仅 encoder 解冻时生效）')
    train_g.add_argument('--w_smooth', type=float, default=0.0,
                         help='latent 时序平滑正则权重（联合训练防特征突变）')
 
    # ── 工程 ─────────────────────────────────────────────────────────────────
    eng = p.add_argument_group('工程')
    eng.add_argument('--encoder_ckpt', type=str,
                     default=BEST_CKPT_PATH,
                     help='阶段一预训练 encoder 权重路径')
    eng.add_argument('--ckpt_dir',     type=str, default='checkpoints/emDecoder',
                     help='checkpoint 保存目录')
    eng.add_argument('--resume',       type=str, default='',
                     help='从指定 checkpoint 恢复训练')
    eng.add_argument('--cpu',          action='store_true')
 
    return p.parse_args()
 
 
if __name__ == '__main__':
    train(parse_args())
 
