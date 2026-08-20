#!/usr/bin/env python3
"""
分析 ACT 数据集中「合手」前后各关节状态变化。

支持:
  - LinkerHand L10: qpos 16 维 = 臂6 + 手10（0~255，越小越合）
  - Inspire:        qpos 12 维 = 臂6 + 手6 （0~1000，越小越合）

用法:
  # L10 Insert_cup（默认 task）
  python inspect_data.py --task_name sim_Insert_cup

  # Inspire Insertion_Cup
  python inspect_data.py --task_name sim_Insertion_Cup \\
      --dataset_dir "/media/ub/My Book/04-Imitation Learning/Insertion_Cup"

  # 也可显式指定手型
  python inspect_data.py --dataset_dir ... --hand_type inspire
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np

ARM_NAMES = ['waist', 'shoulder', 'elbow', 'forearm_roll', 'wrist_angle', 'wrist_rotate']


@dataclass
class HandSpec:
    name: str
    hand_dim: int
    state_dim: int          # 6 + hand_dim
    hand_names: List[str]
    curl_idx: List[int]
    min_drop: float         # 合手检测最小下降幅度
    early_curl_thresh: float  # 「合手前已明显合手」判定


HAND_SPECS = {
    'l10': HandSpec(
        name='l10',
        hand_dim=10,
        state_dim=16,
        hand_names=[
            'thumb_root', 'thumb_side',
            'index_root', 'middle_root', 'ring_root', 'pinky_root',
            'index_side', 'ring_side', 'pinky_side', 'thumb_rot',
        ],
        curl_idx=[0, 2, 3, 4, 5],
        min_drop=25.0,
        early_curl_thresh=10.0,
    ),
    'inspire': HandSpec(
        name='inspire',
        hand_dim=6,
        state_dim=12,
        hand_names=[
            'finger1', 'finger2', 'finger3', 'finger4', 'finger5', 'finger6',
        ],
        # Inspire: 前 5 维为主要开合；第 6 维数据中常为 0
        curl_idx=[0, 1, 2, 3, 4],
        min_drop=100.0,          # 量程 0~1000
        early_curl_thresh=50.0,
    ),
}


def resolve_hand_spec(hand_type: str, qpos_dim: Optional[int] = None) -> HandSpec:
    """hand_type: auto | l10 | inspire"""
    ht = hand_type.lower()
    if ht == 'auto':
        if qpos_dim is None:
            raise ValueError('auto 模式需要 qpos_dim')
        if qpos_dim >= 16:
            return HAND_SPECS['l10']
        if qpos_dim == 12:
            return HAND_SPECS['inspire']
        raise ValueError(f'无法从 qpos_dim={qpos_dim} 推断手型，请用 --hand_type')
    if ht not in HAND_SPECS:
        raise KeyError(f'unknown hand_type={hand_type}, choose auto|l10|inspire')
    return HAND_SPECS[ht]


def list_episodes(dataset_dir: str) -> List[int]:
    pat = re.compile(r'^episode_(\d+)\.hdf5$')
    ids = []
    for name in os.listdir(dataset_dir):
        m = pat.match(name)
        if m:
            ids.append(int(m.group(1)))
    return sorted(ids)


def load_qpos(path: str) -> np.ndarray:
    with h5py.File(path, 'r') as f:
        if '/observations/qpos' not in f:
            raise KeyError(f'{path}: missing /observations/qpos')
        return np.asarray(f['/observations/qpos'][()], dtype=np.float64)


def finger_curl(hand: np.ndarray, curl_idx: List[int]) -> np.ndarray:
    return hand[:, curl_idx].mean(axis=1)


def smooth_1d(x: np.ndarray, k: int = 5) -> np.ndarray:
    if k <= 1 or len(x) < k:
        return x.copy()
    k = int(k) | 1
    pad = k // 2
    xp = np.pad(x, (pad, pad), mode='edge')
    ker = np.ones(k) / k
    return np.convolve(xp, ker, mode='valid')


def detect_close_onset(
    curl: np.ndarray,
    drop_frac: float = 0.25,
    smooth_k: int = 5,
    min_drop: float = 25.0,
) -> Optional[Dict]:
    """
    检测合手起点（L10/Inspire 均为数值下降=合拢）:
      阈值 = open - drop_frac * (open - min)
    """
    if len(curl) < 10:
        return None
    c = smooth_1d(curl, smooth_k)
    head = max(5, len(c) // 10)
    open_ref = float(np.median(c[:head]))
    c_min = float(c.min())
    total_drop = open_ref - c_min
    if total_drop < min_drop:
        return None

    thresh = open_ref - drop_frac * total_drop
    below = np.where(c <= thresh)[0]
    if len(below) == 0:
        return None
    t0 = int(below[0])

    dc = np.diff(c)
    t_fast = int(np.argmin(dc)) + 1 if len(dc) else t0

    return {
        't0': t0,
        't_fast': t_fast,
        'open_ref': open_ref,
        'curl_min': c_min,
        'thresh': thresh,
        'total_drop': total_drop,
        'curl_at_t0': float(c[t0]),
        'curl_smooth': c,
    }


def window_slice(T: int, t0: int, window: int) -> Tuple[slice, slice]:
    pre = slice(max(0, t0 - window), t0)
    post = slice(t0, min(T, t0 + window))
    return pre, post


def summarize_episode(
    qpos: np.ndarray,
    hand_spec: HandSpec,
    window: int,
    drop_frac: float,
) -> Optional[Dict]:
    if qpos.shape[1] < hand_spec.state_dim:
        return None
    arm = qpos[:, :6]
    hand = qpos[:, 6:6 + hand_spec.hand_dim]
    curl = finger_curl(hand, hand_spec.curl_idx)
    det = detect_close_onset(
        curl, drop_frac=drop_frac, min_drop=hand_spec.min_drop
    )
    if det is None:
        return None

    t0 = det['t0']
    T = len(qpos)
    pre, post = window_slice(T, t0, window)

    def delta(arr, sl):
        if sl.stop - sl.start < 2:
            return np.zeros(arr.shape[1])
        return arr[sl.stop - 1] - arr[sl.start]

    def abs_travel(arr, sl):
        if sl.stop - sl.start < 2:
            return np.zeros(arr.shape[1])
        return np.abs(np.diff(arr[sl], axis=0)).sum(axis=0)

    pre_arm_travel = abs_travel(arm, pre)
    post_arm_travel = abs_travel(arm, post)

    return {
        **det,
        'T': T,
        'arm_at_t0': arm[t0].copy(),
        'hand_at_t0': hand[t0].copy(),
        'pre_arm_delta': delta(arm, pre),
        'post_arm_delta': delta(arm, post),
        'pre_hand_delta': delta(hand, pre),
        'post_hand_delta': delta(hand, post),
        'pre_arm_travel': pre_arm_travel,
        'post_arm_travel': post_arm_travel,
        'pre_hand_travel': abs_travel(hand, pre),
        'post_hand_travel': abs_travel(hand, post),
        'pre_curl_delta': float(curl[pre.stop - 1] - curl[pre.start]) if pre.stop - pre.start >= 2 else 0.0,
        'post_curl_delta': float(curl[post.stop - 1] - curl[post.start]) if post.stop - post.start >= 2 else 0.0,
        'wrist_pre_travel': float(pre_arm_travel[3:6].sum()),
        'wrist_post_travel': float(post_arm_travel[3:6].sum()),
        'curl_raw': curl,
        'arm': arm,
        'hand': hand,
    }


def main():
    parser = argparse.ArgumentParser(description='Inspect grasp-close timing in ACT dataset')
    parser.add_argument('--task_name', type=str, default='sim_Insert_cup')
    parser.add_argument('--dataset_dir', type=str, default=None,
                        help='覆盖 task 配置中的 dataset_dir')
    parser.add_argument(
        '--hand_type', type=str, default='auto',
        choices=['auto', 'l10', 'inspire'],
        help='手型：auto 按 qpos 维数推断（12→inspire, 16→l10）',
    )
    parser.add_argument('--window', type=int, default=20,
                        help='合手前后分析窗口（帧）')
    parser.add_argument('--drop_frac', type=float, default=0.25,
                        help='合手起点：相对总下降量的比例阈值')
    parser.add_argument('--max_episodes', type=int, default=None)
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--plot_examples', type=int, default=6,
                        help='保存前 N 条有合手检测的示例曲线')
    args = parser.parse_args()

    if args.dataset_dir is None:
        from constants import SIM_TASK_CONFIGS
        if args.task_name not in SIM_TASK_CONFIGS:
            raise KeyError(f'unknown task {args.task_name}, set --dataset_dir')
        dataset_dir = SIM_TASK_CONFIGS[args.task_name]['dataset_dir']
    else:
        dataset_dir = args.dataset_dir

    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(dataset_dir)

    # out_dir: 若显式给了 dataset_dir，用目录名区分，避免和 L10 结果混写
    default_tag = args.task_name
    if args.dataset_dir is not None:
        default_tag = f'{args.task_name}_{os.path.basename(os.path.normpath(dataset_dir))}'
    out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'inspect_outputs',
        default_tag,
    )
    os.makedirs(out_dir, exist_ok=True)

    ids = list_episodes(dataset_dir)
    if not ids:
        raise RuntimeError(f'no episode_*.hdf5 under {dataset_dir}')
    if args.max_episodes is not None:
        ids = ids[: args.max_episodes]

    # 用第一条推断手型
    sample_q = load_qpos(os.path.join(dataset_dir, f'episode_{ids[0]}.hdf5'))
    hand_spec = resolve_hand_spec(args.hand_type, qpos_dim=sample_q.shape[1])

    print(f'task: {args.task_name}')
    print(f'dataset: {dataset_dir}')
    print(
        f'hand: {hand_spec.name}  qpos_dim={sample_q.shape[1]}  '
        f'hand_dim={hand_spec.hand_dim}  curl_idx={hand_spec.curl_idx}  '
        f'min_drop={hand_spec.min_drop}'
    )
    print(f'episodes: {len(ids)}  window=±{args.window}  drop_frac={args.drop_frac}')

    results = []
    skipped = 0
    for eid in ids:
        path = os.path.join(dataset_dir, f'episode_{eid}.hdf5')
        try:
            qpos = load_qpos(path)
            if qpos.ndim != 2 or qpos.shape[1] < hand_spec.state_dim:
                skipped += 1
                continue
            # 若手型与数据不一致则跳过
            if args.hand_type != 'auto' and qpos.shape[1] != hand_spec.state_dim:
                skipped += 1
                continue
            info = summarize_episode(qpos, hand_spec, args.window, args.drop_frac)
            if info is None:
                skipped += 1
                continue
            info['episode_id'] = eid
            results.append(info)
        except Exception as e:
            print(f'skip episode_{eid}: {e}')
            skipped += 1

    if not results:
        print('未检测到明显合手事件，请调低 --drop_frac 或检查数据。')
        return

    print(f'\n检测到合手: {len(results)} / {len(ids)}  (跳过 {skipped})')

    t0s = np.array([r['t0'] for r in results])
    Ts = np.array([r['T'] for r in results])
    print(
        f'合手起点 t0: min={t0s.min()} max={t0s.max()} '
        f'mean={t0s.mean():.1f} median={np.median(t0s):.1f}'
    )
    print(
        f't0/T: mean={np.mean(t0s / Ts):.2f} median={np.median(t0s / Ts):.2f}'
    )
    print(
        f'curl: open_ref mean={np.mean([r["open_ref"] for r in results]):.1f}  '
        f'min mean={np.mean([r["curl_min"] for r in results]):.1f}  '
        f'at_t0 mean={np.mean([r["curl_at_t0"] for r in results]):.1f}'
    )

    pre_arm_d = np.stack([r['pre_arm_delta'] for r in results], 0)
    post_arm_d = np.stack([r['post_arm_delta'] for r in results], 0)
    pre_hand_d = np.stack([r['pre_hand_delta'] for r in results], 0)
    post_hand_d = np.stack([r['post_hand_delta'] for r in results], 0)
    pre_arm_tr = np.stack([r['pre_arm_travel'] for r in results], 0)
    post_arm_tr = np.stack([r['post_arm_travel'] for r in results], 0)
    pre_hand_tr = np.stack([r['pre_hand_travel'] for r in results], 0)
    post_hand_tr = np.stack([r['post_hand_travel'] for r in results], 0)

    print(f'\n=== 臂关节：合手前/后窗口净变化 meanΔ (末-初) ===')
    print(f'{"joint":16s} {"preΔ":>10s} {"postΔ":>10s} {"pre travel":>12s} {"post travel":>12s}')
    for i, name in enumerate(ARM_NAMES):
        print(
            f'{name:16s} {pre_arm_d[:, i].mean():10.3f} {post_arm_d[:, i].mean():10.3f} '
            f'{pre_arm_tr[:, i].mean():12.3f} {post_arm_tr[:, i].mean():12.3f}'
        )

    print(f'\n=== 手关节（{hand_spec.name}）：合手前/后窗口净变化 meanΔ (末-初) ===')
    print(f'{"joint":16s} {"preΔ":>10s} {"postΔ":>10s} {"pre travel":>12s} {"post travel":>12s}')
    for i, name in enumerate(hand_spec.hand_names):
        print(
            f'{name:16s} {pre_hand_d[:, i].mean():10.3f} {post_hand_d[:, i].mean():10.3f} '
            f'{pre_hand_tr[:, i].mean():12.3f} {post_hand_tr[:, i].mean():12.3f}'
        )

    wrist_pre = np.array([r['wrist_pre_travel'] for r in results])
    wrist_post = np.array([r['wrist_post_travel'] for r in results])
    curl_pre = np.array([r['pre_curl_delta'] for r in results])
    curl_post = np.array([r['post_curl_delta'] for r in results])
    print('\n=== 阶段重叠指标（越大说明合手时腕还在猛转）===')
    print(
        f'腕相关关节(3-5)累计运动: pre={wrist_pre.mean():.2f}±{wrist_pre.std():.2f}  '
        f'post={wrist_post.mean():.2f}±{wrist_post.std():.2f}'
    )
    print(
        f'finger curl 净变化: pre={curl_pre.mean():.2f}±{curl_pre.std():.2f}  '
        f'post={curl_post.mean():.2f}±{curl_post.std():.2f}'
    )
    early_close = (curl_pre < -hand_spec.early_curl_thresh) & (wrist_pre > np.median(wrist_pre))
    print(
        f'合手前已明显合手且腕仍较活跃的 episode: '
        f'{early_close.sum()} / {len(results)} '
        f'({100.0 * early_close.mean():.1f}%)'
    )

    arm_t0 = np.stack([r['arm_at_t0'] for r in results], 0)
    hand_t0 = np.stack([r['hand_at_t0'] for r in results], 0)
    print('\n=== 合手起点 t0 处关节分布 (mean ± std) 【门控可参考】===')
    for i, name in enumerate(ARM_NAMES):
        print(f'  arm/{name:14s}: {arm_t0[:, i].mean():8.2f} ± {arm_t0[:, i].std():6.2f}')
    for i, name in enumerate(hand_spec.hand_names):
        print(f'  hand/{name:14s}: {hand_t0[:, i].mean():8.2f} ± {hand_t0[:, i].std():6.2f}')

    np.savez(
        os.path.join(out_dir, 'close_onset_summary.npz'),
        hand_type=hand_spec.name,
        episode_ids=np.array([r['episode_id'] for r in results]),
        t0=t0s,
        T=Ts,
        arm_at_t0=arm_t0,
        hand_at_t0=hand_t0,
        pre_arm_travel=pre_arm_tr,
        post_arm_travel=post_arm_tr,
        pre_hand_travel=pre_hand_tr,
        post_hand_travel=post_hand_tr,
        wrist_pre_travel=wrist_pre,
        wrist_post_travel=wrist_post,
        curl_pre_delta=curl_pre,
        curl_post_delta=curl_post,
    )

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), constrained_layout=True)
    x = np.arange(6)
    w = 0.35
    axes[0].bar(x - w / 2, pre_arm_tr.mean(0), w, label='pre (before close)', color='C0')
    axes[0].bar(x + w / 2, post_arm_tr.mean(0), w, label='post (after close)', color='C1')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(ARM_NAMES, rotation=20, ha='right')
    axes[0].set_ylabel('mean |travel|')
    axes[0].set_title(
        f'{args.task_name} [{hand_spec.name}]: arm travel ±{args.window} around close'
    )
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    xh = np.arange(hand_spec.hand_dim)
    axes[1].bar(xh - w / 2, pre_hand_tr.mean(0), w, label='pre', color='C0')
    axes[1].bar(xh + w / 2, post_hand_tr.mean(0), w, label='post', color='C1')
    axes[1].set_xticks(xh)
    axes[1].set_xticklabels(hand_spec.hand_names, rotation=30, ha='right')
    axes[1].set_ylabel('mean |travel|')
    axes[1].set_title(
        f'{args.task_name} [{hand_spec.name}]: hand travel ±{args.window} around close'
    )
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    fig.savefig(os.path.join(out_dir, 'travel_pre_post.png'), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.hist(t0s, bins=30, color='C2', edgecolor='k', alpha=0.85)
    ax.set_xlabel('close onset frame t0')
    ax.set_ylabel('count')
    ax.set_title(f'{args.task_name} [{hand_spec.name}]: grasp-close onset')
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(out_dir, 't0_hist.png'), dpi=150)
    plt.close(fig)

    n_plot = min(args.plot_examples, len(results))
    for i in range(n_plot):
        r = results[i]
        eid = r['episode_id']
        t0 = r['t0']
        curl = r['curl_raw']
        arm = r['arm']
        hand = r['hand']
        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True, constrained_layout=True)
        axes[0].plot(curl, label='finger curl mean', color='C3')
        axes[0].axvline(t0, color='k', ls='--', label=f't0={t0}')
        axes[0].axhline(r['thresh'], color='gray', ls=':', label='onset thresh')
        axes[0].set_ylabel('curl')
        axes[0].legend(loc='best')
        axes[0].set_title(f'episode_{eid} [{hand_spec.name}]: close onset')
        axes[0].grid(True, alpha=0.3)

        for j, name in enumerate(ARM_NAMES):
            axes[1].plot(arm[:, j], label=name, alpha=0.85)
        axes[1].axvline(t0, color='k', ls='--')
        axes[1].set_ylabel('arm (deg)')
        axes[1].legend(loc='upper left', ncol=3, fontsize=8)
        axes[1].grid(True, alpha=0.3)

        for j in hand_spec.curl_idx:
            axes[2].plot(hand[:, j], label=hand_spec.hand_names[j], alpha=0.85)
        axes[2].axvline(t0, color='k', ls='--')
        axes[2].set_ylabel('hand curl joints')
        axes[2].set_xlabel('frame')
        axes[2].legend(loc='best', fontsize=8)
        axes[2].grid(True, alpha=0.3)
        fig.savefig(os.path.join(out_dir, f'episode_{eid}_close.png'), dpi=120)
        plt.close(fig)

    print('\n=== 简要结论 ===')
    if wrist_pre.mean() > wrist_post.mean() * 0.8 and abs(curl_pre.mean()) > hand_spec.early_curl_thresh * 0.5:
        print(
            '- 合手前窗口内腕部仍有明显运动，且 curl 已开始下降：'
            '数据里存在一定「边转腕边合手」重叠，部署门控需更严。'
        )
    elif wrist_pre.mean() > wrist_post.mean():
        print(
            '- 合手前腕部运动大于合手后：说明合手起点大致落在「转腕刚结束/接近结束」。'
            '门控可用 t0 处臂关节/姿态分布。'
        )
    else:
        print(
            '- 合手后腕部运动仍不小：合手与后续靠近/调整重叠，门控不宜只看腕完全静止。'
        )
    print(f'\n结果已保存到: {out_dir}')
    print('  - close_onset_summary.npz')
    print('  - travel_pre_post.png')
    print('  - t0_hist.png')
    print(f'  - episode_*_close.png (x{n_plot})')


if __name__ == '__main__':
    main()
