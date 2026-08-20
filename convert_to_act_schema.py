"""
将自采 teleop HDF5 补齐为 ACT 训练所需 schema。

输入:  teleop_episode_*.hdf5 或 episode_*.hdf5
       含 observations/arm_joints (T,6) + observations/hand_joints (T,10)
         + observations/images/<cam> （默认 front_RGB + wrist_RGB）
输出:  episode_{源文件编号}.hdf5（保留源 episode 编号，不重排）
       /observations/qpos  = concat(arm, hand)  (T, 16) float32
       /action             = qpos[t+1], 末帧重复  (T, 16) float32
       RGB:   /observations/images/<cam>
       depth: /observations/depth/<cam>（camera_names 中名称含 depth 时）
       attrs['sim'] = False
"""

import argparse
import os
import re

import h5py
import numpy as np
from tqdm import tqdm

DEFAULT_CAMERA_NAMES = ['front_RGB', 'wrist_RGB']


def is_depth_name(name):
    return 'depth' in name.lower()


def list_source_episodes(src_dir):
    """返回按原始 episode 编号排序的 (idx, filename) 列表。"""
    pat = re.compile(r'^(?:teleop_)?episode_(\d+)\.hdf5$')
    items = []
    for name in os.listdir(src_dir):
        m = pat.match(name)
        if m:
            items.append((int(m.group(1)), name))
    items.sort(key=lambda x: x[0])
    return items


def parse_camera_names(s):
    """'front_RGB,wrist_RGB' -> list"""
    names = [x.strip() for x in s.split(',') if x.strip()]
    if not names:
        raise ValueError('camera_names 不能为空')
    return names


def convert_episode(src_path, dst_path, camera_names=None, compression=None):
    """compression: None / 'gzip' / 'lzf'。图像按帧 chunk，便于训练随机读。"""
    if camera_names is None:
        camera_names = list(DEFAULT_CAMERA_NAMES)

    with h5py.File(src_path, 'r') as src:
        arm = np.asarray(src['/observations/arm_joints'][()], dtype=np.float32)
        hand = np.asarray(src['/observations/hand_joints'][()], dtype=np.float32)
        if arm.ndim != 2 or arm.shape[1] != 6:
            raise ValueError(f'unexpected arm_joints shape {arm.shape} in {src_path}')
        if hand.ndim != 2 or hand.shape[1] != 10:
            raise ValueError(f'unexpected hand_joints shape {hand.shape} in {src_path}')
        if len(arm) != len(hand):
            raise ValueError(f'arm/hand length mismatch {len(arm)} vs {len(hand)} in {src_path}')

        qpos = np.concatenate([arm, hand], axis=1)  # (T, 16)
        action = np.empty_like(qpos)
        action[:-1] = qpos[1:]
        action[-1] = qpos[-1]

        observations_by_name = {}
        for cam_name in camera_names:
            if is_depth_name(cam_name):
                candidates = (
                    f'/observations/depth/{cam_name}',
                    f'/observations/images/{cam_name}',
                )
                key = next((item for item in candidates if item in src), None)
                if key is None:
                    raise KeyError(f'depth {cam_name!r} not found; tried {candidates}')
            else:
                key = f'/observations/images/{cam_name}'
                if key not in src:
                    raise KeyError(f'{key} not found in {src_path}')
            observations_by_name[cam_name] = src[key][()]

        ds_kw = {}
        if compression:
            ds_kw['compression'] = compression

        os.makedirs(os.path.dirname(dst_path) or '.', exist_ok=True)
        with h5py.File(dst_path, 'w') as dst:
            dst.attrs['sim'] = False
            for k, v in src.attrs.items():
                if k == 'sim':
                    continue
                try:
                    dst.attrs[k] = v
                except Exception:
                    pass

            obs = dst.create_group('observations')
            obs.create_dataset('qpos', data=qpos, **ds_kw)
            img_grp = obs.create_group('images')
            depth_names = [name for name in camera_names if is_depth_name(name)]
            depth_grp = obs.create_group('depth') if depth_names else None
            for cam_name, values in observations_by_name.items():
                chunks = (1,) + tuple(values.shape[1:])
                target_group = depth_grp if is_depth_name(cam_name) else img_grp
                target_group.create_dataset(
                    cam_name, data=values, chunks=chunks, **ds_kw
                )
            dst.create_dataset('action', data=action, **ds_kw)

    return len(qpos), qpos.shape[1]


def main():
    parser = argparse.ArgumentParser(description='Convert teleop HDF5 to ACT schema')
    parser.add_argument(
        '--src_dir',
        type=str,
        default='/mnt/additional/Data/DexMimic_Data/Robot_data/Insert_cup/l10',
    )
    parser.add_argument(
        '--dst_dir',
        type=str,
        default='/mnt/additional/Data/DexMimic_Data/Robot_data_ACT/Insert_cup',
    )
    parser.add_argument(
        '--camera_names',
        type=str,
        default=','.join(DEFAULT_CAMERA_NAMES),
        help=(
            '逗号分隔观测名；RGB 从 observations/images 读取，名称含 depth '
            '时从 observations/depth（或旧 images 路径）读取'
        ),
    )
    parser.add_argument(
        '--camera_name',
        type=str,
        default=None,
        help='兼容旧参数：单相机名；若设置则覆盖 --camera_names',
    )
    parser.add_argument('--overwrite', action='store_true', help='overwrite existing outputs')
    parser.add_argument(
        '--compression',
        type=str,
        default='none',
        choices=['none', 'gzip', 'lzf'],
        help='HDF5 压缩：none=不压缩(训练更快，占空间大)；gzip/lzf 省空间',
    )
    args = parser.parse_args()

    if not os.path.isdir(args.src_dir):
        raise FileNotFoundError(f'src_dir not found: {args.src_dir}')

    if args.camera_name:
        camera_names = [args.camera_name]
    else:
        camera_names = parse_camera_names(args.camera_names)

    items = list_source_episodes(args.src_dir)
    if not items:
        raise RuntimeError(f'no episode hdf5 found under {args.src_dir}')

    compression = None if args.compression == 'none' else args.compression

    os.makedirs(args.dst_dir, exist_ok=True)
    print(f'source: {args.src_dir}  ({len(items)} episodes)')
    print(f'dest:   {args.dst_dir}')
    print(f'cameras: {camera_names}')
    print(f'compression: {args.compression}')
    print(f'qpos = concat(arm_joints, hand_joints), action[t]=qpos[t+1], sim=False')

    lengths = []
    out_ids = []
    for old_idx, name in tqdm(items, desc='convert'):
        src_path = os.path.join(args.src_dir, name)
        dst_path = os.path.join(args.dst_dir, f'episode_{old_idx}.hdf5')
        if os.path.exists(dst_path) and not args.overwrite:
            try:
                with h5py.File(dst_path, 'r') as f:
                    expected = [
                        (
                            f'/observations/depth/{c}'
                            if is_depth_name(c)
                            else f'/observations/images/{c}'
                        )
                        for c in camera_names
                    ]
                    if '/action' in f and all(key in f for key in expected):
                        lengths.append(len(f['/action']))
                        out_ids.append(old_idx)
                        continue
            except Exception:
                pass
            os.remove(dst_path)
        T, D = convert_episode(
            src_path,
            dst_path,
            camera_names=camera_names,
            compression=compression,
        )
        lengths.append(T)
        out_ids.append(old_idx)

    print(f'\ndone: {len(lengths)} files -> {args.dst_dir}')
    print(f'qpos/action dim: 16')
    print(f'cameras written: {camera_names}')
    print(f'episode length: min={min(lengths)} max={max(lengths)} '
          f'mean={np.mean(lengths):.1f} median={np.median(lengths):.1f}')
    print(
        f'output naming: 保留源编号 episode_{{id}}.hdf5，'
        f'id 范围 [{min(out_ids)}, {max(out_ids)}]，共 {len(out_ids)} 个'
    )
    print(
        '提示: 若 train/val 分两次转换，请写到同一 dst_dir，'
        '并保证最终有连续的 episode_0 .. episode_{N-1} 供训练读取'
    )


if __name__ == '__main__':
    main()
