import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader

from depth_utils import is_depth_name, normalize_depth

import IPython
e = IPython.embed

# class EpisodicDataset(torch.utils.data.Dataset):
#     def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats):
#         super(EpisodicDataset).__init__()
#         self.episode_ids = episode_ids
#         self.dataset_dir = dataset_dir
#         self.camera_names = camera_names
#         self.norm_stats = norm_stats
#         self.is_sim = None
#         self.__getitem__(0) # initialize self.is_sim

#     def __len__(self):
#         return len(self.episode_ids)

#     def __getitem__(self, index):
#         sample_full_episode = False # hardcode

#         episode_id = self.episode_ids[index]
#         dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
#         with h5py.File(dataset_path, 'r') as root:
#             is_sim = root.attrs['sim']
#             original_action_shape = root['/action'].shape
#             episode_len = original_action_shape[0]
#             if sample_full_episode:
#                 start_ts = 0
#             else:
#                 start_ts = np.random.choice(episode_len)
#             # get observation at start_ts only
#             qpos = root['/observations/qpos'][start_ts]
#             qvel = root['/observations/qvel'][start_ts]
#             image_dict = dict()
#             for cam_name in self.camera_names:
#                 image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]
#             # get all actions after and including start_ts
#             if is_sim:
#                 action = root['/action'][start_ts:]
#                 action_len = episode_len - start_ts
#             else:
#                 action = root['/action'][max(0, start_ts - 1):] # hack, to make timesteps more aligned
#                 action_len = episode_len - max(0, start_ts - 1) # hack, to make timesteps more aligned

#         self.is_sim = is_sim
#         padded_action = np.zeros(original_action_shape, dtype=np.float32)
#         padded_action[:action_len] = action
#         is_pad = np.zeros(episode_len)
#         is_pad[action_len:] = 1

#         # new axis for different cameras
#         all_cam_images = []
#         for cam_name in self.camera_names:
#             all_cam_images.append(image_dict[cam_name])
#         all_cam_images = np.stack(all_cam_images, axis=0)

#         # construct observations
#         image_data = torch.from_numpy(all_cam_images)
#         qpos_data = torch.from_numpy(qpos).float()
#         action_data = torch.from_numpy(padded_action).float()
#         is_pad = torch.from_numpy(is_pad).bool()

#         # channel last
#         image_data = torch.einsum('k h w c -> k c h w', image_data)

#         # normalize image and change dtype to float
#         image_data = image_data / 255.0
#         action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
#         qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

#         return image_data, qpos_data, action_data, is_pad

# 可选的截断版本 - 如果不想填充太多零的话
class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats, fixed_len=300):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None
        self.fixed_len = fixed_len  # 固定长度，可以设为你数据的中位数长度
        
        self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False # hardcode

        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            is_sim = root.attrs['sim']
            original_action_shape = root['/action'].shape
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
            # get observation at start_ts only
            qpos = root['/observations/qpos'][start_ts]
            # qvel = root['/observations/qvel'][start_ts]
            image_dict = dict()
            for cam_name in self.camera_names:
                if is_depth_name(cam_name):
                    depth_keys = (
                        f'/observations/depth/{cam_name}',
                        f'/observations/images/{cam_name}',
                    )
                    depth_key = next((key for key in depth_keys if key in root), None)
                    if depth_key is None:
                        raise KeyError(
                            f'depth {cam_name!r} not found; tried {depth_keys}'
                        )
                    image_dict[cam_name] = root[depth_key][start_ts]
                else:
                    image_dict[cam_name] = root[
                        f'/observations/images/{cam_name}'
                    ][start_ts]
            # get all actions after and including start_ts
            if is_sim:
                action = root['/action'][start_ts:]
                action_len = episode_len - start_ts
            else:
                action = root['/action'][max(0, start_ts - 1):] # hack, to make timesteps more aligned
                action_len = episode_len - max(0, start_ts - 1) # hack, to make timesteps more aligned

        self.is_sim = is_sim
        
        # 截断或填充到固定长度
        padded_action = np.zeros((self.fixed_len, original_action_shape[1]), dtype=np.float32)
        actual_len = min(action_len, self.fixed_len)
        padded_action[:actual_len] = action[:actual_len]
        
        is_pad = np.zeros(self.fixed_len)
        is_pad[actual_len:] = 1

        # RGB-only keeps the original stacked tensor path byte-for-byte. When a
        # depth modality is requested, use a dict so RGB [3,H,W] and depth
        # [1,H,W] can coexist without fake/repeated channels.
        has_depth = any(is_depth_name(name) for name in self.camera_names)
        if has_depth:
            image_data = {}
            for cam_name in self.camera_names:
                value = image_dict[cam_name]
                if is_depth_name(cam_name):
                    value = normalize_depth(value)[None, ...]
                    image_data[cam_name] = torch.from_numpy(value).float()
                else:
                    value = np.asarray(value)
                    value = np.einsum('h w c -> c h w', value)
                    image_data[cam_name] = torch.from_numpy(value).float() / 255.0
        else:
            all_cam_images = np.stack(
                [image_dict[cam_name] for cam_name in self.camera_names], axis=0
            )
            image_data = torch.from_numpy(all_cam_images)
            image_data = torch.einsum('k h w c -> k c h w', image_data)
            image_data = image_data / 255.0

        # construct proprioception/action observations
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        return image_data, qpos_data, action_data, is_pad

# def get_norm_stats(dataset_dir, num_episodes):
#     all_qpos_data = []
#     all_action_data = []
#     for episode_idx in range(num_episodes):
#         dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
#         with h5py.File(dataset_path, 'r') as root:
#             qpos = root['/observations/qpos'][()]
#             qvel = root['/observations/qvel'][()]
#             action = root['/action'][()]
#         all_qpos_data.append(torch.from_numpy(qpos))
#         all_action_data.append(torch.from_numpy(action))
#     all_qpos_data = torch.stack(all_qpos_data)
#     all_action_data = torch.stack(all_action_data)
#     all_action_data = all_action_data

#     # normalize action data
#     action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
#     action_std = all_action_data.std(dim=[0, 1], keepdim=True)
#     action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

#     # normalize qpos data
#     qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
#     qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
#     qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping

#     stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
#              "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
#              "example_qpos": qpos}

#     return stats

def get_norm_stats(dataset_dir, num_episodes):
    all_qpos_data = []
    all_action_data = []
    
    # 第一步：找到所有episode中的最大长度
    max_len = 0
    episode_lengths = []
    
    print("正在检查episode长度...")
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        if not os.path.exists(dataset_path):
            print(f"警告: 文件 {dataset_path} 不存在，跳过...")
            continue
            
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['/observations/qpos'][()]
            episode_len = len(qpos)
            episode_lengths.append(episode_len)
            max_len = max(max_len, episode_len)
            print(f"Episode {episode_idx}: {episode_len} timesteps")
    
    print(f"\n找到的最大episode长度: {max_len}")
    print(f"最小episode长度: {min(episode_lengths)}")
    print(f"平均episode长度: {np.mean(episode_lengths):.1f}")
    
    # 第二步：加载数据并填充到相同长度
    print("\n正在加载和填充数据...")
    for episode_idx in range(num_episodes):
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
        if not os.path.exists(dataset_path):
            continue
            
        with h5py.File(dataset_path, 'r') as root:
            qpos = root['/observations/qpos'][()]
            # qvel = root['/observations/qvel'][()]
            action = root['/action'][()]
            
            current_len = len(qpos)
            
            # 如果当前episode长度小于最大长度，进行填充
            if current_len < max_len:
                pad_len = max_len - current_len
                print(f"Episode {episode_idx}: 填充 {pad_len} 个timesteps")
                
                # 使用最后一个值进行填充 (edge padding)
                qpos_padded = np.pad(qpos, ((0, pad_len), (0, 0)), mode='edge')
                action_padded = np.pad(action, ((0, pad_len), (0, 0)), mode='edge')
                
                # 或者使用零填充（根据需要选择）
                # qpos_padded = np.pad(qpos, ((0, pad_len), (0, 0)), mode='constant', constant_values=0)
                # action_padded = np.pad(action, ((0, pad_len), (0, 0)), mode='constant', constant_values=0)
                
            else:
                qpos_padded = qpos
                action_padded = action
            
            all_qpos_data.append(torch.from_numpy(qpos_padded))
            all_action_data.append(torch.from_numpy(action_padded))
    
    # 现在所有张量都有相同的长度，可以安全地stack
    all_qpos_data = torch.stack(all_qpos_data)
    all_action_data = torch.stack(all_action_data)
    
    print(f"最终数据形状:")
    print(f"qpos shape: {all_qpos_data.shape}")
    print(f"action shape: {all_action_data.shape}")

    # normalize action data
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

    # normalize qpos data
    qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": qpos}

    return stats


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val):
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    train_ratio = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
