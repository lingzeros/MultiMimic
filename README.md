# MultiMimic

精简后的仓库保留两条模型链路：

- RM65 + L10/Inspire 双 decoder ACT：训练、数据转换、离线测试和真机部署。
- `pointcloud_embedding`：编码器、解码器、训练脚本及权重。

## 主要文件

- `imitate_episodes.py`：双 decoder ACT 训练入口。
- `policy.py`：ACT policy、关节损失及可选 FK 位姿监督。
- `detr/`：双 decoder ACT 模型、Transformer、backbone 和 RM65 FK。
- `convert_to_act_schema.py`：将 L10/Inspire HDF5 转换为 ACT schema。
- `offline_test.py`：checkpoint 单帧/全局离线预测及 FK 对比。
- `eval.py`：RM65 + L10/Inspire 真机部署。
- `constants.py`：当前数据集任务配置。
- `utils.py`、`depth_utils.py`：数据加载与预处理。
- `dinov2/`：可选 DINOv2 图像 backbone。
- `linker_hand_python_sdk/`：L10 部署 SDK。
- `pointcloud_embedding/`：点云表征模型完整内容。
- `checkpoints/`：当前训练权重与统计文件（被 Git 忽略）。

## ACT 训练

```bash
python imitate_episodes.py \
  --task_name sim_Peach_in_bowl_inspire \
  --ckpt_dir checkpoints/Peach_dual_decoder_inspire \
  --policy_class ACT --backbone resnet18 \
  --kl_weight 10 --chunk_size 50 --hidden_dim 512 \
  --dim_feedforward 3200 --batch_size 8 \
  --num_epochs 5000 --lr 1e-5 --seed 0
```

启用 FK 位姿监督时增加：

```bash
--fk_pose_weight 1.0 --fk_rotation_weight 1.0
```

## 测试与部署

```bash
# 离线单帧测试
python offline_test.py --hdf5_path 0 --obs_frame -50

# 离线全局测试
python offline_test.py --hdf5_path 0 --global

# 真机部署（先在 eval.py 配置 checkpoint、相机和手型）
python eval.py
```

RM65、RealSense、CAN/Modbus 等硬件依赖需在部署环境中单独安装和配置。
