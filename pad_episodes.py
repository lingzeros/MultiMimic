#!/usr/bin/env python3
"""
智能版：先检查数据结构，然后填充到500
"""

import h5py
import numpy as np
import os
import shutil
from tqdm import tqdm

def inspect_hdf5_structure(filepath):
    """检查HDF5文件结构"""
    with h5py.File(filepath, 'r') as f:
        print(f"检查文件: {os.path.basename(filepath)}")
        
        # 检查action
        action_shape = f['/action'].shape
        print(f"  Action shape: {action_shape}")
        
        # 检查qpos, qvel
        qpos_shape = f['/observations/qpos'].shape
        qvel_shape = f['/observations/qvel'].shape
        print(f"  QPos shape: {qpos_shape}")
        print(f"  QVel shape: {qvel_shape}")
        
        # 检查图像
        print("  Images:")
        for cam_name in f['/observations/images'].keys():
            img_shape = f[f'/observations/images/{cam_name}'].shape
            print(f"    {cam_name}: {img_shape}")
        
        # 检查属性
        print(f"  Attributes: {dict(f.attrs)}")
        print()
        
        return {
            'action_shape': action_shape,
            'qpos_shape': qpos_shape,
            'qvel_shape': qvel_shape,
            'image_shapes': {cam: f[f'/observations/images/{cam}'].shape 
                           for cam in f['/observations/images'].keys()},
            'attrs': dict(f.attrs)
        }

def smart_pad_or_truncate(data, target_len, data_name=""):
    """智能填充，根据数据维度自动处理"""
    original_len = len(data)
    
    if original_len == target_len:
        return data
    
    print(f"    {data_name}: {data.shape} ({original_len} -> {target_len})")
    
    if original_len < target_len:
        pad_len = target_len - original_len
        
        if data.ndim == 1:  # 1D数据
            padded_data = np.pad(data, (0, pad_len), mode='edge')
        elif data.ndim == 2:  # 2D数据 (时间, 特征)
            padded_data = np.pad(data, ((0, pad_len), (0, 0)), mode='edge')
        elif data.ndim == 3:  # 3D图像 (时间, H, W) 
            last_frame = data[-1:] 
            padded_frames = np.repeat(last_frame, pad_len, axis=0)
            padded_data = np.concatenate([data, padded_frames], axis=0)
        elif data.ndim == 4:  # 4D图像 (时间, H, W, C)
            last_frame = data[-1:] 
            padded_frames = np.repeat(last_frame, pad_len, axis=0)
            padded_data = np.concatenate([data, padded_frames], axis=0)
        else:
            print(f"      警告: 未知维度 {data.ndim}, 使用通用填充")
            # 通用填充：只在第一个维度填充
            pad_width = [(0, pad_len)] + [(0, 0)] * (data.ndim - 1)
            padded_data = np.pad(data, pad_width, mode='edge')
    else:
        # 截断
        padded_data = data[:target_len]
    
    return padded_data

def smart_pad_episode(input_file, output_file, target_length=500):
    """智能填充episode"""
    with h5py.File(input_file, 'r') as src:
        # 读取所有数据
        qpos = src['/observations/qpos'][()]
        qvel = src['/observations/qvel'][()]
        action = src['/action'][()]
        
        # 读取图像数据
        image_data = {}
        for cam_name in src['/observations/images'].keys():
            image_data[cam_name] = src[f'/observations/images/{cam_name}'][()]
        
        # 读取属性
        attrs = dict(src.attrs)
    
    print(f"  处理 {os.path.basename(input_file)}:")
    
    # 智能填充所有数据
    padded_qpos = smart_pad_or_truncate(qpos, target_length, "qpos")
    padded_qvel = smart_pad_or_truncate(qvel, target_length, "qvel") 
    padded_action = smart_pad_or_truncate(action, target_length, "action")
    
    padded_images = {}
    for cam_name, img_data in image_data.items():
        padded_images[cam_name] = smart_pad_or_truncate(img_data, target_length, f"image_{cam_name}")
    
    # 写入新文件
    with h5py.File(output_file, 'w') as dst:
        # 复制属性
        for attr_name, attr_value in attrs.items():
            dst.attrs[attr_name] = attr_value
        
        # 创建组和数据集
        obs_group = dst.create_group('observations')
        images_group = obs_group.create_group('images')
        
        # 保存数据
        obs_group.create_dataset('qpos', data=padded_qpos, compression='gzip')
        obs_group.create_dataset('qvel', data=padded_qvel, compression='gzip')
        dst.create_dataset('action', data=padded_action, compression='gzip')
        
        # 保存图像（压缩以节省空间）
        for cam_name, img_data in padded_images.items():
            images_group.create_dataset(cam_name, data=img_data, compression='gzip')

def main():
    dataset_dir = "/home/ub/act/dataset/Pick_Place_Square_backup"
    output_dir = "/home/ub/act/dataset/Pick_Place_Square_padded"
    target_length = 500
    
    print("="*60)
    print("智能Episode填充脚本")
    print("="*60)
    print(f"输入目录: {dataset_dir}")
    print(f"输出目录: {output_dir}")
    print(f"目标长度: {target_length}")
    print("="*60)
    
    # 检查输入目录
    if not os.path.exists(dataset_dir):
        print(f"❌ 输入目录不存在: {dataset_dir}")
        return
    
    # 获取文件列表
    episode_files = [f for f in os.listdir(dataset_dir) 
                    if f.startswith('episode_') and f.endswith('.hdf5')]
    episode_files.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
    
    if not episode_files:
        print("❌ 没有找到episode文件")
        return
    
    print(f"找到 {len(episode_files)} 个episode文件")
    
    # 检查第一个文件的结构
    print("\n📋 检查数据结构:")
    first_file = os.path.join(dataset_dir, episode_files[0])
    structure = inspect_hdf5_structure(first_file)
    
    # 确认
    response = input("确认执行填充? (y/N): ").lower().strip()
    if response not in ['y', 'yes']:
        print("操作已取消")
        return
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 处理每个文件
        for filename in tqdm(episode_files, desc="填充episodes"):
            input_path = os.path.join(dataset_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            smart_pad_episode(input_path, output_path, target_length)
        
        print("\n✅ 所有episode填充完成！")
        print(f"📁 输出目录: {output_dir}")
        
        # 验证结果
        print("\n📋 验证结果:")
        for filename in episode_files[:3]:
            filepath = os.path.join(output_dir, filename)
            with h5py.File(filepath, 'r') as f:
                action_len = len(f['/action'])
                qpos_len = len(f['/observations/qpos'])
                print(f"  {filename}: action={action_len}, qpos={qpos_len}")
        
        print(f"\n🎉 完成！现在可以将 {output_dir} 重命名为原目录名使用")
        print(f"   mv {output_dir} /home/ub/act/dataset/Pick_Place_Square")
                
    except Exception as e:
        print(f"❌ 出现错误: {e}")
        raise

if __name__ == "__main__":
    main()