'''
Stage 1:单独训练emEncoder
目标：
    在无配对、无任务标签的条件下，让 emEncoder 学到一个跨形态的动作潜空间，
    使人手和灵巧手做相同/相似动作时，编码结果在潜空间中距离相近。
 
数据要求：
    - 人手点云序列  [N_h 条演示，每条 T 帧，每帧 N 个点]
    - 灵巧手点云序列 [N_r 条演示，每条 T 帧，每帧 N 个点]
    - 两者无需配对，无需任何标签

训练结果：
    checkpoints/encoder_best.pt   验证集 loss 最优的权重
    checkpoints/encoder_last.pt   最后一个 epoch 的权重
'''

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import sys
import time
from datetime import datetime
import argparse
import random
from pathlib import Path
from tqdm import tqdm
import wandb
import numpy as np

# 须在 import torch 之前加载，以应用 warning 过滤
from constants import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
 
from model.emEncoder import (
    PointCloudSequenceEncoder,
    temporal_contrastive_loss,
    mmd_loss,
    temporal_smoothness_loss,
    JointReconstructionHead,
    joint_reconstruction_loss,
)
from mimic_utils.hand_joint_utils import joints_to_radians, resolve_hand_joint_key

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
class PointCloudDataset(Dataset):
    """
    单一手型的点云序列数据集。
 
    数据格式约定（两种方式任选其一）：
 
    方式 A —— 目录加载（推荐用于真实数据）：
        data_dir/
            seq_000.pt   →  torch.Tensor [T, N, 3]
            seq_001.pt
            ...
        每个 .pt 文件存储一条演示的点云序列，形状 [T, N, 3]。
 
    方式 B —— 单文件加载（推荐用于合成/预处理后的数据）：
        data_path.pt  →  torch.Tensor [num_seqs, T, N, 3]
 
    参数：
        data_path  : str，目录路径（方式A）或 .pt 文件路径（方式B）
        seq_len    : 截取/填充后的统一序列长度 T
        num_points : 每帧保留/下采样的点数 N
        augment    : 是否启用数据增强（随机旋转、抖动）
    """
    def __init__(
            self,
            data_path: str,
            seq_len: int = 20,
            num_points: int = 256,
            augment: bool = False,
            preload_ratio: float = 0.1,
            val_ratio: float = 0.0,
            is_val: bool = False,
            frame_stride: int = 2,
            joints_path: str = None,
            joint_unit: str = 'raw255',
            smooth_window: int = 5,
            smooth_poly: int = 2,
            filter_zero_windows: bool = True,
    ):
        """
        初始化 Dataset，但不读取所有数据。
        只扫描文件列表并计算总的样本数。
        
        Args:
            preload_ratio (float): 每次预加载数据的比例 (默认 10%)。
            val_ratio (float): 将当前数据集划分为验证集的比例。
            is_val (bool): 当前是否为验证集模式（配合 val_ratio 使用）。
            frame_stride (int): 帧抽取间隔，防止相邻帧太相似导致过拟合。
            joints_path (str): 可选。灵巧手关节角度目录（含 .hdf5）。
                               提供时，__getitem__ 额外返回与点云帧同步的关节序列，
                               用于 Stage1 路线 A 的关节重建辅助监督；
                               人手数据不传该参数，行为与原来完全一致。
                               自动识别 Inspire(6)、L10(10) 或 L21(25)。
            joint_unit (str): 关节原始单位（raw/raw255/raw1000/degrees/radians）。
            smooth_window / smooth_poly: 关节 Savitzky-Golay 滤波参数。
            filter_zero_windows (bool): 在建立滑动窗口索引时，过滤任意抽样帧为全零的窗口。
        """
        self.seq_len = seq_len
        self.num_points = num_points
        self.augment = augment
        self.frame_stride = frame_stride
        self.filter_zero_windows = filter_zero_windows
        self.data_path = str(data_path)
        self.is_val_split = is_val
        self.window_stats = {
            'candidate': 0,
            'filtered_all_zero': 0,
            'kept': 0,
        }

        # 关节分支配置（仅灵巧手使用）
        self.use_joints = joints_path is not None
        self.joints_dir = Path(joints_path) if self.use_joints else None
        self.joint_unit = joint_unit
        self.smooth_window = smooth_window
        self.smooth_poly = smooth_poly
        # 首次成功解析后固定：('inspire'|'l21'|'l10', joint_dim)
        self.hand_type = None
        self.joint_dim = None

        p = Path(data_path)
        self.file_paths = []
        self.is_single_file = False
        
        # 记录文件列表或单文件信息
        if p.is_dir():
            for f in sorted(p.glob('*')):
                if f.suffix in ['.pt', '.npy']:
                    self.file_paths.append(f)
        elif p.suffix in ['.pt', '.npy']:
            self.file_paths.append(p)
            self.is_single_file = True
        else:
            raise ValueError(f"data_path 须为目录或者.pt/.npy文件，当前为:{data_path}")
            
        if len(self.file_paths) == 0:
             raise ValueError(f"在 {data_path} 中未找到有效数据文件")

        # ------------------
        # 在读取文件前，如果有多个文件，按文件级别划分 train/val，避免同文件内时序泄露
        # ------------------
        if val_ratio > 0 and len(self.file_paths) > 1:
            import random
            rng = random.Random(42)  # 固定种子保证 train/val 划分一致
            files_copy = list(self.file_paths)
            rng.shuffle(files_copy)
            n_val = max(1, int(len(files_copy) * val_ratio))
            n_train = len(files_copy) - n_val
            if is_val:
                self.file_paths = files_copy[n_train:]
            else:
                self.file_paths = files_copy[:n_train]

        # 为了计算有效样本总数和构建索引，我们需要做一次轻量级扫描
        self.samples_meta = [] # 记录每个有效片段的元数据：(file_idx, seq_idx, start_frame)
        
        # 仅在需要时打印一次日志，避免 4 次初始化打印 4 次
        self._scan_data(quiet=True)
        
        # ------------------
        # 如果只有单文件且需要划分，则按序列(seq)或时序(frame)划分
        # ------------------
        if val_ratio > 0 and self.is_single_file:
            import random
            rng = random.Random(42)
            unique_seqs = list(set([m[1] for m in self.samples_meta]))
            
            if len(unique_seqs) > 1:
                # 多个序列，按序列划分
                rng.shuffle(unique_seqs)
                n_val = max(1, int(len(unique_seqs) * val_ratio))
                n_train = len(unique_seqs) - n_val
                valid_seqs = set(unique_seqs[n_train:] if is_val else unique_seqs[:n_train])
                self.samples_meta = [m for m in self.samples_meta if m[1] in valid_seqs]
            else:
                # 只有一个超长序列，按前后帧顺序划分，不能随机打乱以防严重的数据泄露
                meta_copy = list(self.samples_meta)
                n_val = max(1, int(len(meta_copy) * val_ratio))
                n_train = len(meta_copy) - n_val
                if is_val:
                    self.samples_meta = meta_copy[n_train:]
                else:
                    self.samples_meta = meta_copy[:n_train]

        self.total_samples = len(self.samples_meta)
        if self.total_samples == 0:
            raise ValueError(f"数据集中没有找到有效的长度 >= {seq_len} 的序列")

        # Chunk 加载控制
        self.preload_ratio = preload_ratio
        self.chunk_size = max(1, int(len(self.file_paths) * preload_ratio))
        if self.is_single_file:
             self.chunk_size = 1 # 如果是单文件，无法分块，只能一次性加载（后续可针对大单文件再优化）
             
        self.current_chunk_idx = -1
        self.loaded_sequences = {} # 缓存当前 chunk 加载的数据：{file_idx: [seq1, seq2...]}
        self.loaded_joints = {}    # 与 loaded_sequences 同步的关节缓存：{file_idx: [joint1, ...]}
        
        # 为了保证训练时的随机性，我们在每次 epoch 获取数据前，可以在外部或者这里控制打乱。
        # 这里我们在初始化时按 chunk 组织并打乱 chunk 内的顺序，以兼顾随机性和内存读取效率
        self._shuffle_within_chunks()

    def _candidate_strides(self):
        """返回 __getitem__ 可能使用的所有 stride。

        训练增强会随机改变 stride；建索引时对所有可能值都检查，
        从而保证运行时无论抽到哪个 stride，窗口都不含全零帧。
        """
        if not self.augment:
            return (self.frame_stride,)
        if self.frame_stride > 1:
            return tuple(sorted({self.frame_stride - 1, self.frame_stride, self.frame_stride + 1}))
        return (1, 2)

    def _frame_indices(self, total_frames: int, start: int, stride: int):
        """按与 __getitem__ 完全相同的越界回退规则生成帧索引。"""
        end = start + (self.seq_len - 1) * stride + 1
        if end > total_frames:
            fallback_start = max(0, total_frames - self.seq_len * stride)
            frame_idx = list(range(fallback_start, total_frames, stride))[:self.seq_len]
            if len(frame_idx) < self.seq_len:
                frame_idx = list(range(max(0, total_frames - self.seq_len), total_frames))
        else:
            frame_idx = list(range(start, end, stride))
        return frame_idx

    def _shuffle_within_chunks(self):
        """
        为了避免跨 chunk 频繁加载（Chunk Thrashing），我们只在 chunk 内部打乱样本。
        同时，为了增加全局随机性，我们打乱不同 chunk 被加载的先后顺序。
        """
        import random
        chunk_groups = []
        
        # 按 chunk 分组收集样本
        num_chunks = (len(self.file_paths) + self.chunk_size - 1) // self.chunk_size
        for chunk_idx in range(num_chunks):
            start_file_idx = chunk_idx * self.chunk_size
            end_file_idx = min((chunk_idx + 1) * self.chunk_size, len(self.file_paths))
            
            # 找到属于这个 chunk 的所有样本
            chunk_samples = [m for m in self.samples_meta if start_file_idx <= m[0] < end_file_idx]
            random.shuffle(chunk_samples) # 内部打乱
            if chunk_samples:
                chunk_groups.append(chunk_samples)
            
        # 打乱 chunk 加载的先后顺序
        random.shuffle(chunk_groups)
        
        # 展平合并
        new_meta = []
        for group in chunk_groups:
            new_meta.extend(group)
            
        self.samples_meta = new_meta

    def _joint_path_for(self, cloud_file: Path) -> Path:
        '''根据点云文件名推断配对的关节 .hdf5 路径（episode_X_pointclouds.npy -> episode_X.hdf5）'''
        episode_name = cloud_file.stem.replace('_pointclouds', '')
        return self.joints_dir / f"{episode_name}.hdf5"

    def _remember_hand_spec(self, hand_type: str, joint_dim: int, joint_path: Path) -> None:
        if self.hand_type is None:
            self.hand_type = hand_type
            self.joint_dim = joint_dim
            return
        if self.hand_type != hand_type or self.joint_dim != joint_dim:
            raise ValueError(
                f"手型不一致：已锁定 {self.hand_type}/J={self.joint_dim}，"
                f"但 {joint_path} 为 {hand_type}/J={joint_dim}。"
                " 请勿在同一训练集混用 Inspire、L10 与 L21。"
            )

    def _load_joint_array(self, joint_path: Path) -> torch.Tensor:
        '''
        加载单个 episode 的关节角度并预处理（单位转换 + Savitzky-Golay 平滑）。
        支持 Inspire hand_joints[T,6] / L10[T,10] / L21 hand_qpos[T,25]。
        '''
        import numpy as np
        import h5py
        from scipy.signal import savgol_filter

        with h5py.File(str(joint_path), 'r') as f:
            key, hand_type, expected_dim = resolve_hand_joint_key(f)
            joint_data = f[key][:].astype(np.float32)

        if joint_data.ndim != 2:
            raise ValueError(f"{joint_path}[{key}] 期望 [T,J]，实际 shape={joint_data.shape}")
        j = int(joint_data.shape[-1])
        if j != expected_dim:
            raise ValueError(
                f"{joint_path}[{key}] 末维应为 {expected_dim}（{hand_type}），实际 J={j}"
            )
        self._remember_hand_spec(hand_type, j, joint_path)

        joint_data = joints_to_radians(
            joint_data, self.joint_unit, hand_type=hand_type, joint_dim=j
        ).astype(np.float32)

        # 高频锯齿平滑
        if self.smooth_window > 0 and joint_data.shape[0] >= self.smooth_window:
            window_length = self.smooth_window
            if window_length % 2 == 0:
                window_length += 1
            joint_data = savgol_filter(joint_data, window_length, self.smooth_poly, axis=0)
            joint_data = joint_data.astype(np.float32)

        return torch.from_numpy(joint_data)

    def _scan_data(self, quiet=False):
        """轻量级扫描，获取数据维度信息以构建样本索引，不读取完整数据"""
        file_iter = self.file_paths
        if not quiet:
            file_iter = tqdm(self.file_paths, desc="扫描数据结构", leave=False)
            
        for file_idx, f in enumerate(file_iter):
            try:
                if f.suffix == '.pt':
                    # torch.load 默认会加载全部，为了轻量，这里如果数据非常大可以考虑用 mmap，但通常扫描还好
                    data = torch.load(f, weights_only=True)
                    if data.ndim == 3: # [T, N, 3]
                         seqs = [data]
                    elif data.ndim == 4: # [num_seqs, T, N, 3]
                         seqs = data
                    else:
                         continue
                elif f.suffix == '.npy':
                    import numpy as np
                    # np.load 使用 mmap_mode='r' 时，如果文件是 object 数组或被额外压缩打包过，会直接抛出 ValueError。
                    # 为了安全和通用，我们退回到常规加载方式。反正只获取 shape，在有大量文件且内存不够时，
                    # 真正的懒加载需要预先保存 meta 信息，但目前常规加载也是可以承受的。
                    try:
                        npy_data = np.load(f, allow_pickle=True, mmap_mode='r')
                        if npy_data.dtype == object:
                            raise ValueError("Object array")
                    except ValueError:
                        # 无法 memory-map (通常是因为 dtype=object 或格式问题)，使用普通加载
                        npy_data = np.load(f, allow_pickle=True)
                        
                    if npy_data.dtype == object:
                        npy_data = np.stack(npy_data)
                    
                    if npy_data.ndim == 3:
                         seqs = [npy_data]
                    elif npy_data.ndim == 4:
                         seqs = npy_data
                    else:
                         continue
                
                # 关节分支：定位配对的关节文件并取其长度，用于和点云对齐
                joint_T = None
                if self.use_joints:
                    jp = self._joint_path_for(f)
                    if not jp.exists():
                        print(f"[警告] 找不到配对关节文件 {jp}，跳过点云 {f.name}")
                        continue
                    import h5py
                    with h5py.File(str(jp), 'r') as jf:
                        try:
                            key, hand_type, expected_dim = resolve_hand_joint_key(jf)
                        except KeyError as e:
                            print(f"[警告] {jp} {e}，跳过")
                            continue
                        shape = jf[key].shape
                        if len(shape) < 2 or int(shape[-1]) != expected_dim:
                            print(
                                f"[警告] {jp}[{key}] 期望 [T,{expected_dim}]（{hand_type}），"
                                f"实际 {shape}，跳过"
                            )
                            continue
                        self._remember_hand_spec(hand_type, expected_dim, jp)
                        joint_T = shape[0]

                # 构建滑动窗口索引
                for seq_idx, seq in enumerate(seqs):
                    T = seq.shape[0]
                    # 关节存在时，用点云与关节的公共长度，保证窗口索引对两者都有效
                    if joint_T is not None:
                        T = min(T, joint_T)
                    # 需要保证有足够的帧能抽取出 seq_len 长度的序列
                    required_frames = (self.seq_len - 1) * self.frame_stride + 1
                    if T >= required_frames:
                        # 只读取用于判断的全零帧掩码，不保留完整点云副本。
                        # “全零帧”指该帧所有点的 XYZ 都精确等于 0。
                        if self.filter_zero_windows:
                            if torch.is_tensor(seq):
                                zero_frames = torch.all(seq[:T] == 0, dim=tuple(range(1, seq.ndim))).cpu().numpy()
                            else:
                                zero_frames = np.all(np.asarray(seq[:T]) == 0, axis=tuple(range(1, seq.ndim)))
                        else:
                            zero_frames = None

                        for start in range(T - required_frames + 1):
                            self.window_stats['candidate'] += 1
                            if zero_frames is not None:
                                contains_zero = any(
                                    bool(np.any(zero_frames[self._frame_indices(T, start, stride)]))
                                    for stride in self._candidate_strides()
                                )
                                if contains_zero:
                                    self.window_stats['filtered_all_zero'] += 1
                                    continue
                            self.samples_meta.append((file_idx, seq_idx, start))
                            self.window_stats['kept'] += 1
            except Exception as e:
                print(f"扫描文件 {f} 失败: {e}")

        split_name = '验证' if self.is_val_split else '训练'
        print(
            f"[窗口过滤][{split_name}] candidate={self.window_stats['candidate']}, "
            f"filtered_all_zero={self.window_stats['filtered_all_zero']}, "
            f"kept={self.window_stats['kept']} ({self.data_path})"
        )

    def load_chunk(self, chunk_idx):
        """加载指定块的数据到内存，并清理旧数据"""
        if chunk_idx == self.current_chunk_idx:
            return
            
        self.loaded_sequences.clear() # 清除之前的缓存，释放内存
        self.loaded_joints.clear()
        import gc
        gc.collect()
        
        start_file_idx = chunk_idx * self.chunk_size
        end_file_idx = min((chunk_idx + 1) * self.chunk_size, len(self.file_paths))
        
        # tqdm progress for chunk loading
        file_range = range(start_file_idx, end_file_idx)
        # 如果当前只有一个文件，就不显示进度条了，避免闪烁
        if len(file_range) > 1:
            file_iter = tqdm(file_range, desc=f"预加载 Chunk {chunk_idx}", leave=False)
        else:
            file_iter = file_range

        for file_idx in file_iter:
            f = self.file_paths[file_idx]
            if f.suffix == '.pt':
                data = torch.load(f, weights_only=True)
                if data.ndim == 3:
                    self.loaded_sequences[file_idx] = [data]
                elif data.ndim == 4:
                    self.loaded_sequences[file_idx] = [data[i] for i in range(data.shape[0])]
            elif f.suffix == '.npy':
                import numpy as np
                npy_data = np.load(f, allow_pickle=True)
                if npy_data.dtype == object:
                    npy_data = np.stack(npy_data).astype(np.float32)
                else:
                    npy_data = npy_data.astype(np.float32)
                data = torch.from_numpy(npy_data)
                
                if data.ndim == 3:
                    self.loaded_sequences[file_idx] = [data]
                elif data.ndim == 4:
                    self.loaded_sequences[file_idx] = [data[i] for i in range(data.shape[0])]

            # 关节分支：同步加载配对关节序列（仅支持单序列点云文件）
            if self.use_joints:
                n_seqs = len(self.loaded_sequences.get(file_idx, []))
                if n_seqs == 0:
                    continue
                if n_seqs != 1:
                    raise ValueError(
                        f"关节配对仅支持单序列点云文件，但 {f.name} 含 {n_seqs} 条序列"
                    )
                jp = self._joint_path_for(f)
                if jp.exists():
                    self.loaded_joints[file_idx] = [self._load_joint_array(jp)]

        self.current_chunk_idx = chunk_idx

    def __len__(self) -> int:
        return self.total_samples
    
    def __getitem__(self, idx: int):
        file_idx, seq_idx, start = self.samples_meta[idx]
        
        # 检查当前数据是否在缓存中
        if file_idx not in self.loaded_sequences:
            # 动态决定加载哪个 chunk
            target_chunk = file_idx // self.chunk_size
            self.load_chunk(target_chunk)
            
        seq_full = self.loaded_sequences[file_idx][seq_idx].float()

        # 关节序列（如启用）。用点云与关节的公共长度做边界，保证两者索引同步
        if self.use_joints:
            joint_full = self.loaded_joints[file_idx][seq_idx].float()
            T_eff = min(seq_full.shape[0], joint_full.shape[0])
        else:
            joint_full = None
            T_eff = seq_full.shape[0]
        
        # ------------------
        # 时序帧选择（含训练增强）——对点云与关节使用同一组帧索引，确保严格对齐
        # ------------------
        current_stride = self.frame_stride
        if self.augment:
            # 随机变速：以 50% 概率稍微改变帧间隔（加速或减速）
            if random.random() < 0.5:
                stride_offsets = [-1, 0, 1] if self.frame_stride > 1 else [0, 1]
                current_stride = max(1, self.frame_stride + random.choice(stride_offsets))
        
        # 按照动态 stride 计算帧索引（与建索引时的全零帧检查共用同一逻辑）
        frame_idx = self._frame_indices(T_eff, start, current_stride)
        frame_idx = torch.as_tensor(frame_idx, dtype=torch.long)

        # 用高级索引取帧（返回副本，避免回写污染 chunk 缓存）
        seq = seq_full[frame_idx]                           # [T, N, 3]
        joint = joint_full[frame_idx] if self.use_joints else None  # [T, J]

        if self.augment:
            # 随机帧丢弃（Frame Dropping / Jittering）：
            # 以 30% 概率将某几帧替换为其前一帧，点云与关节同步替换
            if random.random() < 0.3 and seq.shape[0] == self.seq_len:
                num_drops = random.randint(1, 3) # 丢弃 1~3 帧
                drop_indices = random.sample(range(1, self.seq_len), num_drops) # 不丢弃第0帧
                for drop_idx in drop_indices:
                    seq[drop_idx] = seq[drop_idx - 1].clone()
                    if self.use_joints:
                        joint[drop_idx] = joint[drop_idx - 1].clone()
        
        # ------------------
        # 空间数据预处理（仅作用于点云，不影响关节角度）
        # ------------------
        N = seq.shape[1]
        if N > self.num_points:
            idx_pts = torch.randperm(N)[: self.num_points]
            seq = seq[:, idx_pts, :]
        elif N < self.num_points:
            # 有放回上采样
            idx_pts = torch.randint(0, N, (self.num_points,))
            seq = seq[:, idx_pts, :]

        # ------------------
        # 点云中心化与归一化（全局序列归一化 Global Sequence Normalization）
        # ------------------
        # 1. 计算整条序列的统一质心
        global_centroid = seq.view(-1, 3).mean(dim=0, keepdim=True)  # [1, 3]
        # 所有帧减去统一质心
        seq = seq - global_centroid

        # 2. 计算整条序列中离质心最远的距离
        distances = torch.sqrt(torch.sum(seq**2, dim=-1)) # [T, N]
        global_max_dist = torch.max(distances) # 标量
        # 防止除以 0
        global_max_dist = torch.clamp(global_max_dist, min=1e-6)
        # 整条序列统一除以这个最大距离
        seq = seq / global_max_dist # [T, N, 3]

        # 数据增强（旋转/抖动/缩放，仅作用于点云）
        if self.augment:
            seq = self._augment(seq)

        if self.use_joints:
            return seq, joint   # [T, N, 3], [T, J]
        return seq              # [T, N, 3]
    
    @staticmethod
    def _augment(seq: torch.Tensor) -> torch.Tensor:
        """
        轻量在线增强，对整条序列施加同一变换，保持时序一致性。
 
        - 随机旋转（绕 Z 轴 ±180°）：增加朝向鲁棒性
        - 随机抖动（高斯噪声，σ=0.01）：模拟传感器噪声
        - 随机缩放（0.9~1.1）：模拟不同距离拍摄
        """
        T, N, _ = seq.shape
 
        # 绕 Z 轴随机旋转
        angle = torch.rand(1).item() * 2 * 3.14159 - 3.14159
        ca, sa = torch.cos(torch.tensor(angle)), torch.sin(torch.tensor(angle))
        rot = torch.tensor([[ca, -sa, 0.],
                             [sa,  ca, 0.],
                             [0.,  0., 1.]])                       # [3, 3]
        seq = (seq.reshape(-1, 3) @ rot.T).reshape(T, N, 3)
 
        # 随机抖动
        seq = seq + torch.randn_like(seq) * 0.01
 
        # 随机缩放
        scale = 0.9 + torch.rand(1).item() * 0.2
        seq   = seq * scale
 
        return seq
        
class PairedBatchSampler:
    '''
    双流批采样器
    每次迭代同时从人手和灵巧手数据中各取一个batch，保证两个batch大小一致，且在同一个训练步中处理
    '''

    def __init__(
            self,
            human_dataset: Dataset,
            robot_dataset: Dataset,
            batch_size: int,
            shuffle: bool = True,
    ):
        # 因为 Dataset 使用了 Chunk 动态加载，为了避免多线程 Worker 每个都复制一份 Chunk 缓存导致内存暴增，
        # 并且避免多 Worker 导致随机访问跳跃引发频繁换页（Chunk Thrashing），
        # 建议在使用 Chunk 策略时 num_workers=0。
        self.human_loader = DataLoader(
            human_dataset,
            batch_size= batch_size,
            shuffle=shuffle,
            drop_last=True,
            num_workers=0,
            pin_memory=True,
        )
        self.robot_loader = DataLoader(
            robot_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=True,
            num_workers=0,
            pin_memory=True,
        )
    
    def __iter__(self):
        '''
        每次 yield 一个 (x_human, x_robot) 元组。
        以较短的 loader 为主循环，避免 zip 截断问题时重复利用大数据集。
        '''
        human_iter = iter(self.human_loader)  # 创建迭代器
        robot_iter = iter(self.robot_loader)

        # 以较小数据集的batch数为准
        n_batches = min(len(self.human_loader),len(self.robot_loader))
        for _ in range(n_batches):
            try:
                x_h = next(human_iter)
            except StopIteration:  # 如果取尽了，重新开始从第一个取
                human_iter=iter(self.human_loader)
                x_h = next(human_iter)
            try:
                x_r = next(robot_iter)
            except StopIteration:
                robot_iter = iter(self.robot_loader)
                x_r = next(robot_iter)
            yield x_h, x_r

    def __len__(self) -> int:
        return min(len(self.human_loader), len(self.robot_loader))
    
# ══════════════════════════════════════════════════════════════════════════════
# 训练 / 验证单步
# ══════════════════════════════════════════════════════════════════════════════
def encoder_step(
        encoder: nn.Module,
        joint_head: nn.Module,
        x_human: torch.Tensor,
        x_robot: torch.Tensor,
        gt_theta_robot: torch.Tensor,
        w_tc: float,
        w_mmd: float,
        w_smooth: float,
        w_joint: float,
        tau: float = 0.1,
        latent_norm: bool = True,
) -> dict:
    # ---使用encoder进行编码---
    z_h = encoder(x_human)   # [B, T, D]
    z_r = encoder(x_robot)   # [B, T, D]

    # ---latent 归一化（投影到单位球）---
    # 关节头会把灵巧手 latent 拉伸/重排成姿态可分的结构，使两域出现“尺度/半径”差异，
    # 而 MMD 的 RBF 核对尺度敏感 → 对齐被破坏。把 latent 投到单位球后：
    #   1) 去掉半径自由度，MMD 只需匹配“方向分布”，更稳、更易收敛；
    #   2) 与 emencoder_probe --mode knn 的余弦相似度判据一致（训练目标=评估指标）。
    if latent_norm:
        z_h = F.normalize(z_h, dim=-1)
        z_r = F.normalize(z_r, dim=-1)
    
    # ---对编码结果进行损失计算---
    # 时序对比损失
    l_tc = (temporal_contrastive_loss(z_h, temperature=tau) + temporal_contrastive_loss(z_r, temperature=tau)) * 0.5
    # MMD 损失
    z_h_flat = z_h.reshape(-1, z_h.shape[-1])   # [B*T, D]
    z_r_flat = z_r.reshape(-1, z_r.shape[-1])
    l_mmd = mmd_loss(z_h_flat, z_r_flat)
    # 平滑损失
    l_smooth = (temporal_smoothness_loss(z_h) + temporal_smoothness_loss(z_r)) * 0.5

    # 关节重建辅助损失（路线 A）：仅作用在灵巧手 latent 上，强制其保留姿态信息
    if joint_head is not None and gt_theta_robot is not None and w_joint > 0:
        pred_sincos = joint_head(z_r)
        l_joint = joint_reconstruction_loss(pred_sincos, gt_theta_robot)
    else:
        l_joint = z_r.new_tensor(0.0)

    total = w_tc*l_tc + w_mmd*l_mmd + w_smooth*l_smooth + w_joint*l_joint

    return {
        'total': total,
        'tc': l_tc.detach(),
        'mmd': l_mmd.detach(),
        'smooth': l_smooth.detach(),
        'joint': l_joint.detach(),
    }
# ══════════════════════════════════════════════════════════════════════════════
# 主训练循环
# ══════════════════════════════════════════════════════════════════════════════
def train(args: argparse.Namespace):
    # 保存设置
    # ------------------
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_dir = Path(args.ckpt_dir) / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 启动日志记录器
    log_file = save_dir / "log.txt"
    sys.stdout = Logger(filename=str(log_file))
    print(f"[{timestamp}] 启动训练，所有文件将保存在: {save_dir}")
    
    # 初始化 Weights & Biases (W&B)
    # wandb.login(key=WANDB_API_KEY)
    wandb.init(
        project="Mymimic-emEncoder", 
        name=timestamp, 
        config=vars(args),
        dir=str(save_dir)
    )
    print(f'[日志] W&B 日志记录已启动')

    # 训练设备
    device = torch.device(
        'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    )
    print(f'[设备] {device}')
    
    # 数据集
    print('[数据]加载中......')
    print('正在扫描并划分数据集结构，请稍候...')

    # 人手数据：同目录拆分
    human_train = PointCloudDataset(
        args.human_data, args.seq_len, args.num_points, augment=True,
        val_ratio=args.val_ratio, is_val=False,
        filter_zero_windows=args.filter_zero_windows,
    )
    human_val = PointCloudDataset(
        args.human_data, args.seq_len, args.num_points, augment=False,
        val_ratio=args.val_ratio, is_val=True,
        filter_zero_windows=args.filter_zero_windows,
    )

    # 灵巧手数据：复用 PointCloudDataset 的滑动窗口/分块结构，并额外配对关节角度（路线 A）
    robot_train = PointCloudDataset(
        args.robot_data, args.seq_len, args.num_points, augment=True,
        joints_path=args.robot_joints, joint_unit=args.joint_unit,
        smooth_window=args.smooth_window, smooth_poly=args.smooth_poly,
        filter_zero_windows=args.filter_zero_windows,
    )
    robot_val = PointCloudDataset(
        args.robot_val_data, args.seq_len, args.num_points, augment=False, is_val=True,
        joints_path=args.robot_val_joints, joint_unit=args.joint_unit,
        smooth_window=args.smooth_window, smooth_poly=args.smooth_poly,
        filter_zero_windows=args.filter_zero_windows,
    )

    # 另外 shuffle 也要设为 False（或者自定义一个按 Chunk 内 shuffle 的 Sampler），
    # 我们这里通过 RandomSampler 配合 Chunk 来做，但简单起见，可以先关闭全局 shuffle 并在内部按顺序加载。
    # 更优的策略是：为了保证每个 epoch 能随机访问，可以在 epoch 开始前打乱 self.samples_meta。
    train_sampler = PairedBatchSampler(human_train, robot_train, args.batch_size, shuffle=False)
    val_sampler = PairedBatchSampler(human_val, robot_val, args.batch_size, shuffle=False)

    print(f'  人手   训练 {len(human_train)} / 验证 {len(human_val)} 条滑动窗口样本')
    print(f'  灵巧手 训练 {len(robot_train)} / 验证 {len(robot_val)} 条滑动窗口样本')
    if robot_train.hand_type is not None:
        print(
            f'  灵巧手关节格式：{robot_train.hand_type.upper()} '
            f'(J={robot_train.joint_dim}；'
            f'{"hand_qpos" if robot_train.hand_type == "l21" else "hand_joints"})'
        )
        if robot_val.hand_type is not None and robot_val.hand_type != robot_train.hand_type:
            raise ValueError(
                f"训练/验证手型不一致：train={robot_train.hand_type} "
                f"val={robot_val.hand_type}"
            )
    print(f'  注意：使用预加载 Chunk 模式以节省内存。')

    # 模型初始化
    encoder = PointCloudSequenceEncoder(
        point_feat_dim=args.point_feat_dim,
        latent_dim=args.latent_dim,
        temporal_layers=args.temporal_layers,
        temporal_heads  = args.temporal_heads,
        dropout         = args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"[模型] emEncoder 参数量：{n_params:,}")

    # 关节重建辅助头（路线 A）：仅 Stage1 训练使用，不随 encoder 部署
    _, sample_joint = robot_train[0]
    joint_dim = sample_joint.shape[-1]
    joint_head = JointReconstructionHead(
        latent_dim=args.latent_dim,
        joint_dim=joint_dim,
        dropout=args.dropout,
        head_type=args.joint_head_type,
    ).to(device)
    hand_tag = (robot_train.hand_type or "?").upper()
    print(
        f"[模型] 关节重建辅助头：latent_dim={args.latent_dim} -> joint_dim={joint_dim} "
        f"(type={args.joint_head_type}, hand={hand_tag})"
    )

    # 优化器
    optimizer = torch.optim.AdamW([
        {'params': encoder.point_encoder.parameters(), 'lr': args.lr * 0.01},
        {'params': encoder.temporal_encoder.parameters(), 'lr': args.lr},
        {'params': joint_head.parameters(), 'lr': args.lr},
    ], weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max = args.epochs,
        eta_min = args.lr * 0.01
    )

    # checkpoint 目录
    ckpt_dir = save_dir

    # 从断点恢复
    start_epoch = 0
    best_val = float('inf')
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        encoder.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_val    = ckpt.get('best_val', float('inf'))
        print(f'[恢复] 从 epoch {start_epoch} 继续训练，历史最优 val loss = {best_val:.6f}')

    # 训练的主循环
    print(f'\n[训练] 开始，共 {args.epochs} 个 epoch\n')

    for epoch in range(start_epoch, args.epochs):
        # ------------------
        # 在每个 epoch 开始前，重新打乱数据（包括 Chunk 顺序和 Chunk 内部样本顺序）
        # ------------------
        human_train._shuffle_within_chunks()
        if hasattr(robot_train, "_shuffle_within_chunks"):
            robot_train._shuffle_within_chunks()
        
        encoder.train()
        joint_head.train()
        train_metrics = {k: 0.0 for k in ['total', 'tc', 'mmd', 'smooth', 'joint']}
        t0 = time.time()

        # 训练集使用固定的温度，避免崩塌
        current_tau = 0.1

        # ── 关节损失 warmup + 线性 ramp ───────────────────────────────────────
        # 前 joint_warmup_epochs 个 epoch 完全关闭关节头（w_joint=0），
        # 让 tc/mmd/smooth 先把跨域对齐建立起来；
        # 随后用 joint_ramp_epochs 个 epoch 把关节权重从 0 线性升到目标值，
        # 避免强监督在对齐尚未稳定时就把灵巧手 latent 拉跑。
        if epoch < args.joint_warmup_epochs:
            current_w_joint = 0.0
        else:
            ramp = max(1, args.joint_ramp_epochs)
            progress = (epoch - args.joint_warmup_epochs + 1) / ramp
            current_w_joint = args.w_joint * min(1.0, progress)

        train_pbar = tqdm(train_sampler, desc=f"Epoch {epoch+1:03d}/{args.epochs} [Train]", leave=False)
        for step, (x_h, x_r) in enumerate(train_pbar):
            x_h = x_h.to(device, non_blocking=True)   # [B, T, N, 3]
            x_r_cloud, x_r_joint = x_r                 # 点云 [B, T, N, 3] / 关节 [B, T, J]
            x_r_cloud = x_r_cloud.to(device, non_blocking=True)
            x_r_joint = x_r_joint.to(device, non_blocking=True)

            losses = encoder_step(
                encoder, joint_head, x_h, x_r_cloud, x_r_joint,
                w_tc = args.w_tc,
                w_mmd = args.w_mmd,
                w_smooth = args.w_smooth,
                w_joint = current_w_joint,
                tau = current_tau,
                latent_norm = not args.no_latent_norm,
            )
            optimizer.zero_grad()
            losses['total'].backward()
            nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(joint_head.parameters()), max_norm=1.0
            )
            optimizer.step()

            for k in train_metrics:
                if k!='total':
                    train_metrics[k] += losses[k].item()
                else:
                    train_metrics[k] += losses[k].detach().item()

            # 更新进度条后缀信息
            train_pbar.set_postfix({
                'loss': f"{losses['total'].item():.4f}",
                'tc': f"{losses['tc'].item():.4f}",
                'joint': f"{losses['joint'].item():.4f}"
            })
 
        n_steps = len(train_sampler)
        for k in train_metrics:
            train_metrics[k] /= n_steps
 
        # 验证阶段
        encoder.eval()
        joint_head.eval()
        val_metrics = {k: 0.0 for k in ['total', 'tc', 'mmd', 'smooth', 'joint']}

        # 验证集使用固定的温度
        current_tau = 0.3

        val_pbar = tqdm(val_sampler, desc=f"Epoch {epoch+1:03d}/{args.epochs} [Val]", leave=False)
        with torch.no_grad():
            for x_h, x_r in val_pbar:
                x_h = x_h.to(device, non_blocking=True)
                x_r_cloud, x_r_joint = x_r
                x_r_cloud = x_r_cloud.to(device, non_blocking=True)
                x_r_joint = x_r_joint.to(device, non_blocking=True)
                losses = encoder_step(
                    encoder, joint_head, x_h, x_r_cloud, x_r_joint,
                    w_tc=args.w_tc, w_mmd=args.w_mmd, w_smooth=args.w_smooth,
                    w_joint=args.w_joint,
                    tau=current_tau,
                    latent_norm = not args.no_latent_norm,
                )
                for k in val_metrics:
                    if k != 'total':
                        val_metrics[k] += losses[k].item()
                    else:
                        val_metrics[k] += losses[k].detach().item()
                val_pbar.set_postfix({'loss': f"{losses['total'].item():.4f}"})
        
        n_val_steps = max(len(val_sampler), 1)
        for k in val_metrics:
            val_metrics[k] /= n_val_steps

        scheduler.step()

        elapsed = time.time() - t0
        print(
            f'[epoch {epoch+1:03d}/{args.epochs}] '
            f'train {train_metrics["total"]:.4f} | '
            f'val {val_metrics["total"]:.4f} | '
            f'lr {scheduler.get_last_lr()[0]:.2e} | '
            f'{elapsed:.1f}s'
        )

        # 记录到 W&B
        wandb.log({
            'Loss/train_total': train_metrics["total"],
            'Loss/train_tc': train_metrics["tc"],
            'Loss/train_mmd': train_metrics["mmd"],
            'Loss/train_smooth': train_metrics["smooth"],
            'Loss/train_joint': train_metrics["joint"],
            'Loss/val_total': val_metrics["total"],
            'Loss/val_tc': val_metrics["tc"],
            'Loss/val_mmd': val_metrics["mmd"],
            'Loss/val_smooth': val_metrics["smooth"],
            'Loss/val_joint': val_metrics["joint"],
            'LR/learning_rate': scheduler.get_last_lr()[0],
            'Sched/w_joint': current_w_joint,
            'epoch': epoch
        })

        # checkpoint保存
        # 注意：'model' 仅保存 encoder 主体，Stage2/3 加载方式无需改动；
        #       'joint_head' 单独保存，仅用于诊断/复现，不参与下游部署。
        state = {
            'epoch':     epoch,
            'model':     encoder.state_dict(),
            'joint_head': joint_head.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_val':  best_val,
            'args':      vars(args),
        }

        torch.save(state, save_dir / 'encoder_last.pt')

        # 验证 loss 改善时保存best
        if val_metrics['total'] < best_val:
            best_val = val_metrics['total']
            state['best_val'] = best_val
            torch.save(state, save_dir / 'encoder_best.pt')
            print(f'  ★ 新的最优 val loss = {best_val:.6f}，已保存 encoder_best.pt')

    print(f'\n[完成] 最优 val loss = {best_val:.6f}')
    print(f'       权重已保存至 {ckpt_dir}')
    wandb.finish()


# ══════════════════════════════════════════════════════════════════════════════
# CLI 入口
# ══════════════════════════════════════════════════════════════════════════════
 
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='阶段一：单独训练 emEncoder（跨形态动作对齐）',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
 
    # ── 数据 ─────────────────────────────────────────────────────────────────
    data = p.add_argument_group('数据')
    data.add_argument('--human_data',  type=str, default=HUMAN_PC_PATH,
                      help='人手点云数据路径（目录或 .pt 文件）')
    data.add_argument('--robot_data',  type=str, default=ROBOT_PC_PATH_TRAIN,
                      help='灵巧手点云数据路径（目录或 .pt 文件）')
    data.add_argument('--robot_val_data', type=str, default=ROBOT_PC_PATH_VAL,
                      help='灵巧手点云验证集数据路径（如果已经拆分好）')
    data.add_argument('--robot_joints', type=str, default=ROBOT_HDF5_PATH_TRAIN,
                      help='灵巧手关节 HDF5（Inspire hand_joints[T,6]；L10[T,10]；L21 hand_qpos[T,25]）')
    data.add_argument('--robot_val_joints', type=str, default=ROBOT_HDF5_PATH_VAL,
                      help='灵巧手关节验证集目录（格式同 --robot_joints）')
    data.add_argument('--joint_unit',  type=str, default='raw',
                      choices=['radians', 'degrees', 'raw', 'raw255', 'raw1000'],
                      help='关节原始单位；raw 自动按手型使用 Inspire=1000、L10/L21=255')
    data.add_argument('--smooth_window', type=int, default=5,
                      help='关节 Savitzky-Golay 滤波窗口')
    data.add_argument('--smooth_poly',   type=int, default=2,
                      help='关节 Savitzky-Golay 滤波多项式阶数')
    data.add_argument('--seq_len',     type=int, default=30,
                      help='统一序列帧数 T')
    data.add_argument('--num_points',  type=int, default=256,
                      help='每帧点数 N')
    data.add_argument('--val_ratio',   type=float, default=0.1,
                      help='验证集比例')
    data.add_argument('--filter_zero_windows', dest='filter_zero_windows',
                      action='store_true', default=True,
                      help='建立窗口索引时过滤含全零帧的窗口')
    data.add_argument('--no_filter_zero_windows', dest='filter_zero_windows',
                      action='store_false',
                      help='关闭全零帧窗口过滤（仅用于对照实验）')
 
    # ── 模型结构 ──────────────────────────────────────────────────────────────
    model = p.add_argument_group('模型')
    model.add_argument('--point_feat_dim',  type=int,   default=256)
    model.add_argument('--latent_dim',      type=int,   default=128)
    model.add_argument('--temporal_layers', type=int,   default=2)
    model.add_argument('--temporal_heads',  type=int,   default=4)
    model.add_argument('--pe_type',         type=str,   default='sinusoidal',
                       choices=['sinusoidal', 'learnable'])
    model.add_argument('--dropout',         type=float, default=0.3)
 
    # ── 训练超参 ──────────────────────────────────────────────────────────────
    train_g = p.add_argument_group('训练')
    train_g.add_argument('--epochs',       type=int,   default=100)
    train_g.add_argument('--batch_size',   type=int,   default=16)
    train_g.add_argument('--lr',           type=float, default=1e-3)
    train_g.add_argument('--weight_decay', type=float, default=1e-3)
    train_g.add_argument('--w_tc',         type=float, default=1.0,
                         help='时序对比损失权重')
    train_g.add_argument('--w_mmd',        type=float, default=0.5,
                         help='MMD 损失权重（调高以让跨域对齐占主导）')
    train_g.add_argument('--w_smooth',     type=float, default=0.01,
                         help='时序平滑损失权重')
    train_g.add_argument('--w_joint',      type=float, default=0.1,
                         help='关节重建辅助损失权重（路线 A，调低避免压制对齐）')
    train_g.add_argument('--joint_head_type', type=str, default='linear',
                         choices=['linear', 'mlp'],
                         help='关节重建辅助头结构：linear 逼姿态进 latent（推荐），mlp 表达力更强但迁移性可能变差')
    train_g.add_argument('--joint_warmup_epochs', type=int, default=20,
                         help='前 N 个 epoch 完全关闭关节损失，先建立跨域对齐')
    train_g.add_argument('--joint_ramp_epochs',   type=int, default=10,
                         help='warmup 结束后，用 N 个 epoch 把关节权重从 0 线性升到 w_joint')
    train_g.add_argument('--no_latent_norm', action='store_true',
                         help='关闭 latent 单位球归一化（默认开启），用于复现旧行为')
 
    # ── 工程 ─────────────────────────────────────────────────────────────────
    eng = p.add_argument_group('工程')
    eng.add_argument('--ckpt_dir',     type=str, default='checkpoints/emEncoder',
                     help='checkpoint 保存目录')
    eng.add_argument('--resume',       type=str, default='',
                     help='从指定 checkpoint 文件恢复训练')
    eng.add_argument('--log_interval', type=int, default=10,
                     help='每隔多少 step 打印一次训练日志')
    eng.add_argument('--cpu',          action='store_true',
                     help='强制使用 CPU（调试用）')
 
    return p.parse_args()
 
 
if __name__ == '__main__':
    train(parse_args())
