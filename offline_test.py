"""Offline evaluation for dual-decoder ACT checkpoints on recorded HDF5 episodes."""

import argparse
import json
import pickle
from pathlib import Path

import h5py
import numpy as np
import torch
from einops import rearrange
from torchvision.transforms import functional as transform_functional

from detr.main import get_args_parser
from detr.models import build_ACT_model
from detr.models.rm65_fk import RM65DifferentiableFK


DEFAULT_CKPT_PATH = (
    '/home/ub/MultiMimic/checkpoints/Peach_dual_decoder_inspire1'
)
DEFAULT_DATASET_DIR = (
    '/mnt/additional/Data/DexMimic_Data/Robot_data_ACT/Peach_in_bowl/inspire'
)
DEFAULT_CAMERA_NAMES = ['front_RGB']
EXECUTION_HORIZON = 25
ARM_JOINT_NAMES = [f'arm_joint_{index}' for index in range(1, 7)]
POSE_NAMES = ['x', 'y', 'z', 'roll', 'pitch', 'yaw']


def resolve_checkpoint(ckpt_path):
    path = Path(ckpt_path).expanduser().resolve()
    if path.is_dir():
        path = path / 'policy_best.ckpt'
    if not path.is_file():
        raise FileNotFoundError(f'checkpoint not found: {path}')
    return path


def resolve_hdf5(hdf5_path):
    """Resolve an episode number against the built-in dataset directory."""
    value = str(hdf5_path).strip()
    if value.lstrip('+-').isdigit():
        episode_id = int(value)
        if episode_id < 0:
            raise ValueError('episode number must be non-negative')
        path = Path(DEFAULT_DATASET_DIR) / f'episode_{episode_id}.hdf5'
    else:
        path = Path(value).expanduser()
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f'HDF5 episode not found: {path}')
    return path


def infer_model_config(state_dict, camera_names):
    qkey = 'model.query_embed.weight'
    arm_key = 'model.arm_action_head.weight'
    hand_key = 'model.hand_action_head.weight'
    missing = [key for key in (qkey, arm_key, hand_key) if key not in state_dict]
    if missing:
        raise KeyError(
            'checkpoint is not a dual-decoder ACT checkpoint; missing: '
            + ', '.join(missing)
        )

    keys = state_dict.keys()
    if any('.body.patch_embed.' in key for key in keys):
        backbone = 'dinov2_vits14'
    elif any('.body.conv1.' in key for key in keys):
        backbone = 'resnet18'
    else:
        raise KeyError('cannot infer image backbone from checkpoint keys')

    if any(key.startswith('model.depth_encoder.') for key in keys):
        raise NotImplementedError('offline_test.py currently expects an RGB-only checkpoint')

    arm_dim = int(state_dict[arm_key].shape[0])
    hand_dim = int(state_dict[hand_key].shape[0])
    if arm_dim != 6:
        raise ValueError(f'expected six arm outputs, got {arm_dim}')
    return {
        'lr': 1e-5,
        'num_queries': int(state_dict[qkey].shape[0]),
        'kl_weight': 10,
        'hidden_dim': int(state_dict[qkey].shape[1]),
        'dim_feedforward': 3200,
        'lr_backbone': 1e-5,
        'backbone': backbone,
        'enc_layers': 4,
        'dec_layers': 7,
        'nheads': 8,
        'camera_names': list(camera_names),
        'state_dim': arm_dim + hand_dim,
    }


def load_policy_and_stats(ckpt_file, camera_names, device):
    state_dict = torch.load(ckpt_file, map_location='cpu', weights_only=True)
    config = infer_model_config(state_dict, camera_names)
    model_args = get_args_parser().parse_args([])
    for key, value in config.items():
        setattr(model_args, key, value)
    model = build_ACT_model(model_args)
    model_state = {
        key.removeprefix('model.'): value
        for key, value in state_dict.items()
        if key.startswith('model.')
    }
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()

    stats_path = ckpt_file.parent / 'dataset_stats.pkl'
    if not stats_path.is_file():
        raise FileNotFoundError(f'dataset statistics not found: {stats_path}')
    with stats_path.open('rb') as stream:
        stats = pickle.load(stream)
    for key in ('qpos_mean', 'qpos_std', 'action_mean', 'action_std'):
        stats[key] = np.asarray(stats[key], dtype=np.float32)
        if stats[key].shape != (config['state_dim'],):
            raise ValueError(
                f'{key} shape={stats[key].shape}, expected '
                f'({config["state_dim"]},)'
            )
    return model, config, stats


def normalize_frame_index(frame, episode_len):
    resolved = frame if frame >= 0 else episode_len + frame
    if resolved < 0 or resolved >= episode_len:
        raise IndexError(
            f'obs frame {frame} resolves to {resolved}, but episode length is '
            f'{episode_len}'
        )
    return resolved


def read_episode_metadata(hdf5_path, camera_names, state_dim):
    with h5py.File(hdf5_path, 'r') as root:
        if '/observations/qpos' not in root or '/action' not in root:
            raise KeyError('HDF5 must contain /observations/qpos and /action')
        if '/action_wrist_pose_base' not in root:
            raise KeyError(
                'HDF5 must contain /action_wrist_pose_base for FK pose comparison'
            )
        qpos_shape = root['/observations/qpos'].shape
        action_shape = root['/action'].shape
        if len(qpos_shape) != 2 or qpos_shape[1] != state_dim:
            raise ValueError(f'qpos shape={qpos_shape}, expected (T, {state_dim})')
        if action_shape != qpos_shape:
            raise ValueError(f'action shape={action_shape} != qpos shape={qpos_shape}')
        if root['/action_wrist_pose_base'].shape != (qpos_shape[0], 6):
            raise ValueError(
                'action_wrist_pose_base shape='
                f'{root["/action_wrist_pose_base"].shape}, expected '
                f'({qpos_shape[0]}, 6)'
            )
        for name in camera_names:
            key = f'/observations/images/{name}'
            if key not in root:
                raise KeyError(f'{key} not found in {hdf5_path}')
            if len(root[key]) != qpos_shape[0]:
                raise ValueError(f'{key} length does not match qpos length')
        is_sim = bool(root.attrs.get('sim', False))
    return qpos_shape[0], is_sim


def read_observation(root, frame, camera_names, stats, device):
    qpos = np.asarray(root['/observations/qpos'][frame], dtype=np.float32)
    qpos_normalized = (qpos - stats['qpos_mean']) / stats['qpos_std']
    qpos_tensor = torch.from_numpy(qpos_normalized).unsqueeze(0).to(device)

    images = [
        np.asarray(root[f'/observations/images/{name}'][frame])
        for name in camera_names
    ]
    image_tensor = np.stack(
        [rearrange(image, 'h w c -> c h w') for image in images], axis=0
    )
    image_tensor = torch.from_numpy(image_tensor / 255.0).float()
    image_tensor = image_tensor.unsqueeze(0).to(device)
    return qpos, qpos_tensor, image_tensor


def predict_chunk(policy, qpos_tensor, image_tensor, stats):
    image_tensor = transform_functional.normalize(
        image_tensor,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    with torch.inference_mode():
        normalized = policy(qpos_tensor, image_tensor, None)[0][0]
    prediction = normalized.detach().cpu().numpy()
    return prediction * stats['action_std'] + stats['action_mean']


def compute_statistics(gt, pred, arm_dim=6):
    error = pred - gt
    abs_error = np.abs(error)
    squared_error = np.square(error)
    hand_slice = slice(arm_dim, gt.shape[1])

    def group_stats(values, squares):
        return {
            'mae': float(values.mean()),
            'rmse': float(np.sqrt(squares.mean())),
            'max_abs_error': float(values.max()),
        }

    return {
        'num_actions': int(gt.shape[0]),
        'state_dim': int(gt.shape[1]),
        'overall': group_stats(abs_error, squared_error),
        'arm': group_stats(abs_error[:, :arm_dim], squared_error[:, :arm_dim]),
        'hand': group_stats(abs_error[:, hand_slice], squared_error[:, hand_slice]),
        'per_dimension_mae': abs_error.mean(axis=0).tolist(),
        'per_dimension_rmse': np.sqrt(squared_error.mean(axis=0)).tolist(),
        'per_chunk_offset_mae': abs_error.mean(axis=1).tolist(),
    }


def rotation_matrix_to_euler_xyz(rotation):
    """Rz(yaw) @ Ry(pitch) @ Rx(roll) to intrinsic XYZ Euler radians."""
    sy = torch.sqrt(rotation[..., 0, 0].square() + rotation[..., 1, 0].square())
    singular = sy < 1e-6
    roll = torch.atan2(rotation[..., 2, 1], rotation[..., 2, 2])
    pitch = torch.atan2(-rotation[..., 2, 0], sy)
    yaw = torch.atan2(rotation[..., 1, 0], rotation[..., 0, 0])
    singular_roll = torch.atan2(-rotation[..., 1, 2], rotation[..., 1, 1])
    roll = torch.where(singular, singular_roll, roll)
    yaw = torch.where(singular, torch.zeros_like(yaw), yaw)
    return torch.stack((roll, pitch, yaw), dim=-1)


def fk_pose_from_actions(fk, actions, device):
    joints = torch.from_numpy(actions[:, :6]).float().to(device)
    with torch.inference_mode():
        transform = fk(joints)
        xyz = transform[..., :3, 3]
        euler = rotation_matrix_to_euler_xyz(transform[..., :3, :3])
        pose = torch.cat((xyz, euler), dim=-1)
    return pose.cpu().numpy()


def wrap_radians(values):
    return (values + np.pi) % (2.0 * np.pi) - np.pi


def compute_pose_statistics(gt_pose, pred_pose):
    position_error = pred_pose[:, :3] - gt_pose[:, :3]
    euler_error = wrap_radians(pred_pose[:, 3:] - gt_pose[:, 3:])
    position_distance = np.linalg.norm(position_error, axis=1)
    rotation_l2 = np.linalg.norm(euler_error, axis=1)
    return {
        'position_units': 'meters',
        'rotation_units': 'radians',
        'position': {
            'mae_per_axis': np.abs(position_error).mean(axis=0).tolist(),
            'mean_euclidean_error': float(position_distance.mean()),
            'rmse_euclidean': float(np.sqrt(np.square(position_distance).mean())),
            'max_euclidean_error': float(position_distance.max()),
        },
        'rotation_euler_wrapped': {
            'mae_per_axis': np.abs(euler_error).mean(axis=0).tolist(),
            'mean_l2_error': float(rotation_l2.mean()),
            'rmse_l2': float(np.sqrt(np.square(rotation_l2).mean())),
            'max_l2_error': float(rotation_l2.max()),
        },
    }


def named_side_by_side(names, gt, pred):
    """JSON key/value comparison: ``gt=<value>`` on left, prediction on right."""
    return {
        name: {f'gt={float(gt[index]):.9g}': float(pred[index])}
        for index, name in enumerate(names)
    }


def compare_at_frame(
    root,
    obs_frame,
    is_sim,
    policy,
    config,
    stats,
    camera_names,
    device,
    fk,
):
    qpos, qpos_tensor, image_tensor = read_observation(
        root, obs_frame, camera_names, stats, device
    )
    pred = predict_chunk(policy, qpos_tensor, image_tensor, stats)

    # This mirrors EpisodicDataset: real-robot actions begin one index earlier.
    action_start = obs_frame if is_sim else max(0, obs_frame - 1)
    action_stop = min(action_start + config['num_queries'], len(root['/action']))
    gt = np.asarray(root['/action'][action_start:action_stop], dtype=np.float32)
    gt_pose = np.asarray(
        root['/action_wrist_pose_base'][action_start:action_stop],
        dtype=np.float32,
    )
    pred = pred[:len(gt)]
    if len(gt) == 0:
        raise ValueError(f'no future actions available for observation frame {obs_frame}')

    pred_pose = fk_pose_from_actions(fk, pred, device)
    joint_names = ARM_JOINT_NAMES + [
        f'hand_joint_{index}' for index in range(1, gt.shape[1] - 6 + 1)
    ]

    comparisons = []
    for offset, (gt_action, pred_action, gt_pose_value, pred_pose_value) in enumerate(
        zip(gt, pred, gt_pose, pred_pose)
    ):
        comparisons.append({
            'chunk_offset': offset,
            'dataset_action_frame': action_start + offset,
            'joint_comparison': named_side_by_side(
                joint_names, gt_action, pred_action
            ),
            'fk_pose_comparison': named_side_by_side(
                POSE_NAMES, gt_pose_value, pred_pose_value
            ),
        })

    return {
        'obs_frame': int(obs_frame),
        'qpos': qpos.tolist(),
        'comparison_format': {
            'left_json_key': 'gt=<ground-truth value>',
            'right_json_value': 'predicted value',
        },
        'joint_units': {
            'arm_joint_1..6': 'degrees',
            'hand_joint_1..N': 'device units',
        },
        'fk_pose_units': {
            'x_y_z': 'meters',
            'roll_pitch_yaw': 'radians (intrinsic XYZ)',
        },
        'gt_action_start_frame': int(action_start),
        'predicted_chunk_size': int(config['num_queries']),
        'compared_chunk_size': len(gt),
        'actions': comparisons,
        'joint_statistics': compute_statistics(gt, pred),
        'fk_pose_statistics': compute_pose_statistics(gt_pose, pred_pose),
    }, gt, pred, gt_pose, pred_pose


def aggregate_global_statistics(all_gt, all_pred, all_gt_pose, all_pred_pose):
    gt = np.concatenate(all_gt, axis=0)
    pred = np.concatenate(all_pred, axis=0)
    gt_pose = np.concatenate(all_gt_pose, axis=0)
    pred_pose = np.concatenate(all_pred_pose, axis=0)
    return {
        'num_prediction_frames': len(all_gt),
        'joint_statistics': compute_statistics(gt, pred),
        'fk_pose_statistics': compute_pose_statistics(gt_pose, pred_pose),
    }


def output_path_for(ckpt_file, hdf5_path, global_mode, obs_frame):
    stem = hdf5_path.stem
    suffix = 'global' if global_mode else f'frame_{obs_frame}'
    return ckpt_file.parent / f'offline_test_{stem}_{suffix}.json'


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run a dual-decoder ACT checkpoint on recorded observations.'
    )
    parser.add_argument(
        '--ckpt_path', default=DEFAULT_CKPT_PATH,
        help='checkpoint directory or a specific .ckpt file',
    )
    parser.add_argument(
        '--obs_frame', type=int, default=-50,
        help='single-mode observation frame; negative values index from the end',
    )
    parser.add_argument(
        '--global', dest='global_mode', action='store_true',
        help='predict throughout the episode; --obs_frame is ignored',
    )
    parser.add_argument(
        '--hdf5_path', default='0',
        help=(
            f'episode number under {DEFAULT_DATASET_DIR}, or an explicit HDF5 path'
        ),
    )
    parser.add_argument(
        '--device', default='cuda' if torch.cuda.is_available() else 'cpu',
        help='PyTorch device (default: cuda when available)',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ckpt_file = resolve_checkpoint(args.ckpt_path)
    hdf5_path = resolve_hdf5(args.hdf5_path)
    device = torch.device(args.device)
    camera_names = list(DEFAULT_CAMERA_NAMES)

    print(f'checkpoint: {ckpt_file}')
    print(f'episode:    {hdf5_path}')
    print(f'device:     {device}')
    policy, config, stats = load_policy_and_stats(
        ckpt_file, camera_names, device
    )
    episode_len, is_sim = read_episode_metadata(
        hdf5_path, camera_names, config['state_dim']
    )
    fk = RM65DifferentiableFK().to(device)
    fk.eval()

    with h5py.File(hdf5_path, 'r') as root:
        if args.global_mode:
            query_frequency = min(EXECUTION_HORIZON, config['num_queries'])
            frames = list(range(0, episode_len, query_frequency))
        else:
            query_frequency = None
            frames = [normalize_frame_index(args.obs_frame, episode_len)]

        results = []
        all_gt = []
        all_pred = []
        all_gt_pose = []
        all_pred_pose = []
        for index, frame in enumerate(frames, start=1):
            print(f'predicting frame {frame} ({index}/{len(frames)})')
            result, gt, pred, gt_pose, pred_pose = compare_at_frame(
                root, frame, is_sim, policy, config, stats,
                camera_names, device, fk,
            )
            results.append(result)
            all_gt.append(gt)
            all_pred.append(pred)
            all_gt_pose.append(gt_pose)
            all_pred_pose.append(pred_pose)

    report = {
        'mode': 'global' if args.global_mode else 'single_frame',
        'checkpoint': str(ckpt_file),
        'hdf5': str(hdf5_path),
        'camera_names': camera_names,
        'device': str(device),
        'episode_length': episode_len,
        'is_sim': is_sim,
        'state_dim': config['state_dim'],
        'chunk_size': config['num_queries'],
        'action_alignment': (
            'action_start=obs_frame' if is_sim
            else 'action_start=max(0, obs_frame-1), matching training'
        ),
        'query_frequency': query_frequency,
        'results': results,
    }
    if args.global_mode:
        report['global_statistics'] = aggregate_global_statistics(
            all_gt, all_pred, all_gt_pose, all_pred_pose
        )

    output_path = output_path_for(
        ckpt_file, hdf5_path, args.global_mode,
        results[0]['obs_frame'],
    )
    with output_path.open('w', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    print(f'report saved: {output_path}')


if __name__ == '__main__':
    main()
