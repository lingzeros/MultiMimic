import torch
import numpy as np
import os
import pickle
import argparse
# 设置matplotlib后端为非交互式
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from constants import SIM_TASK_CONFIGS
from utils import load_data # data functions
from utils import compute_dict_mean, set_seed, detach_dict # helper functions
from policy import ACTPolicy, CNNMLPPolicy

def main(args):
    set_seed(1)
    # command line parameters
    ckpt_dir = args['ckpt_dir']
    policy_class = args['policy_class']
    task_name = args['task_name']
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_epochs = args['num_epochs']

    # get task parameters
    task_config = SIM_TASK_CONFIGS[task_name]
    dataset_dir = task_config['dataset_dir']
    num_episodes = task_config['num_episodes']
    camera_names = task_config['camera_names']

    # A task-specific dimension is authoritative (e.g. Inspire: arm 6 + hand 6
    # = 12). Other tasks retain the CLI override for backward compatibility.
    state_dim = task_config.get('state_dim', args['state_dim'])
    lr_backbone = 1e-5
    backbone = args['backbone']
    if policy_class == 'ACT':
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {'lr': args['lr'],
                         'num_queries': args['chunk_size'],
                         'kl_weight': args['kl_weight'],
                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],
                         'lr_backbone': lr_backbone,
                         'backbone': backbone,
                         'enc_layers': enc_layers,
                         'dec_layers': dec_layers,
                         'nheads': nheads,
                         'camera_names': camera_names,
                         'state_dim': state_dim,
                         'fk_pose_weight': args['fk_pose_weight'],
                         'fk_rotation_weight': args['fk_rotation_weight'],
                         'rm65_urdf': args['rm65_urdf'],
                         }
    elif policy_class == 'CNNMLP':
        policy_config = {'lr': args['lr'], 'lr_backbone': lr_backbone, 'backbone' : backbone, 'num_queries': 1,
                         'camera_names': camera_names, 'state_dim': state_dim,}
    else:
        raise NotImplementedError

    print(f'state_dim={state_dim}  camera_names={camera_names}')

    config = {
        'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        'policy_class': policy_class,
        'policy_config': policy_config,
        'seed': args['seed'],
    }

    use_fk_pose = policy_class == 'ACT' and args['fk_pose_weight'] > 0
    train_dataloader, val_dataloader, stats, _ = load_data(
        dataset_dir,
        num_episodes,
        camera_names,
        batch_size_train,
        batch_size_val,
        use_fk_pose=use_fk_pose,
        state_dim=state_dim,
    )
    if use_fk_pose:
        policy_config.update({
            'action_mean': stats['action_mean'],
            'action_std': stats['action_std'],
            'wrist_pose_std': stats['wrist_pose_std'],
        })

    # save dataset stats
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    # save best checkpoint
    ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}')


def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'CNNMLP':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def forward_pass(data, policy):
    if len(data) == 5:
        image_data, qpos_data, action_data, is_pad, pose_gt = data
    elif len(data) == 4:
        image_data, qpos_data, action_data, is_pad = data
        pose_gt = None
    else:
        raise ValueError(f'Unexpected training batch with {len(data)} elements')
    if isinstance(image_data, dict):
        image_data = {
            name: tensor.cuda(non_blocking=True)
            for name, tensor in image_data.items()
        }
    else:
        image_data = image_data.cuda(non_blocking=True)
    qpos_data = qpos_data.cuda(non_blocking=True)
    action_data = action_data.cuda(non_blocking=True)
    is_pad = is_pad.cuda(non_blocking=True)
    if pose_gt is not None:
        pose_gt = pose_gt.cuda(non_blocking=True)
    return policy(
        qpos_data, image_data, action_data, is_pad, pose_gt=pose_gt,
    )


def train_bc(train_dataloader, val_dataloader, config):
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']

    set_seed(seed)

    policy = make_policy(policy_class, policy_config)
    policy.cuda()
    optimizer = make_optimizer(policy_class, policy)

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    epoch_bar = tqdm(
        range(num_epochs),
        desc='Training',
        dynamic_ncols=True,
        leave=True,
    )
    for epoch in epoch_bar:
        # validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, policy)
                epoch_dicts.append(forward_dict)
            val_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(val_summary)

            epoch_val_loss = val_summary['loss']
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))

        # training
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy)
            # backward
            loss = forward_dict['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
        train_summary = compute_dict_mean(
            train_history[(batch_idx + 1) * epoch:(batch_idx + 1) * (epoch + 1)]
        )

        # Keep all epoch metrics on one terminal line. tqdm updates this postfix in place.
        progress_metrics = {
            'val_loss': f'{epoch_val_loss.item():.4f}',
            'train_loss': f'{train_summary["loss"].item():.4f}',
        }
        progress_metrics.update({
            f'val_{key}': f'{value.item():.3f}'
            for key, value in val_summary.items()
            if key != 'loss'
        })
        progress_metrics.update({
            f'train_{key}': f'{value.item():.3f}'
            for key, value in train_summary.items()
            if key != 'loss'
        })
        epoch_bar.set_postfix(progress_metrics)

        if epoch % 100 == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
            torch.save(policy.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save(policy.state_dict(), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished: seed={seed}, best_val_loss={min_val_loss:.6f}, epoch={best_epoch}')

    # save training curves
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    return best_ckpt_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(np.linspace(0, num_epochs-1, len(train_history)), train_values, label='train')
        plt.plot(np.linspace(0, num_epochs-1, len(validation_history)), val_values, label='validation')
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
        plt.close()  # 添加这一行


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', default ='/home/ub/act/checkpoints', action='store', type=str, help='ckpt_dir', required=False)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=False)
    parser.add_argument('--task_name', default= 'Pick_Place_Square', action='store', type=str, help='task_name', required=False)
    parser.add_argument('--batch_size', default= 8, action='store', type=int, help='batch_size', required=False)
    parser.add_argument('--seed', default= 0, action='store', type=int, help='seed', required=False)
    parser.add_argument('--num_epochs', default= 2000, action='store', type=int, help='num_epochs', required=False)
    parser.add_argument('--lr', default= 1e-5, action='store', type=float, help='lr', required=False)
    parser.add_argument(
        '--backbone',
        default='resnet18',
        choices=['resnet18', 'dinov2_vits14'],
        help='image observation backbone',
    )

    # for ACT
    parser.add_argument('--kl_weight', default= 10, action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', default= 50, action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', default= 512, action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', default= 3200, action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument(
        '--fk_pose_weight',
        default=0.0,
        action='store',
        type=float,
        help=(
            'RM65 FK wrist-pose auxiliary loss weight; 0 disables it and '
            'preserves the original training path'
        ),
    )
    parser.add_argument(
        '--fk_rotation_weight',
        default=1.0,
        action='store',
        type=float,
        help='rotation term weight inside the FK pose loss',
    )
    parser.add_argument(
        '--rm65_urdf',
        default=(
            '/home/ub/snap/Projects/DRO-Grasp-master/'
            'data/data_urdf/robot/rm65b/RM65-B.urdf'
        ),
        action='store',
        type=str,
        help='RM65-B URDF used by differentiable FK supervision',
    )
    parser.add_argument(
        '--state_dim',
        default=16,
        action='store',
        type=int,
        help=(
            'qpos/action 维度：RM65+L10=16，RM65+Inspire=12，ALOHA=14；'
            '若任务配置包含 state_dim，则任务配置优先'
        ),
        required=False,
    )

    main(vars(parser.parse_args()))
