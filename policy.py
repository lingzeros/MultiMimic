import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms

from detr.main import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer
from detr.models.rm65_fk import (
    DEFAULT_RM65_URDF,
    RM65DifferentiableFK,
    masked_fk_pose_loss,
)

class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']
        self.fk_pose_weight = float(args_override.get('fk_pose_weight', 0.0))
        self.fk_rotation_weight = float(
            args_override.get('fk_rotation_weight', 1.0)
        )
        self.fk = None
        if self.fk_pose_weight > 0:
            required_stats = ('action_mean', 'action_std', 'wrist_pose_std')
            missing = [key for key in required_stats if key not in args_override]
            if missing:
                raise ValueError(
                    'FK pose supervision requires dataset statistics: '
                    + ', '.join(missing)
                )
            action_mean = np.asarray(args_override['action_mean'], dtype=np.float32)
            action_std = np.asarray(args_override['action_std'], dtype=np.float32)
            wrist_pose_std = np.asarray(
                args_override['wrist_pose_std'], dtype=np.float32,
            )
            if action_mean.shape[0] < 6 or action_std.shape[0] < 6:
                raise ValueError('FK pose supervision requires six arm action dimensions')
            if wrist_pose_std.shape != (6,):
                raise ValueError(
                    f'wrist_pose_std must have shape (6,), got {wrist_pose_std.shape}'
                )
            self.register_buffer(
                'fk_action_mean', torch.from_numpy(action_mean), persistent=False,
            )
            self.register_buffer(
                'fk_action_std', torch.from_numpy(action_std), persistent=False,
            )
            self.register_buffer(
                'fk_pose_xyz_std',
                torch.from_numpy(wrist_pose_std[:3]),
                persistent=False,
            )
            self.fk = RM65DifferentiableFK(
                urdf_path=args_override.get('rm65_urdf', DEFAULT_RM65_URDF),
            )
        print(f'KL Weight {self.kl_weight}')
        if self.fk is not None:
            print(
                'FK pose supervision enabled: '
                f'weight={self.fk_pose_weight}, '
                f'rotation_weight={self.fk_rotation_weight}'
            )

    def __call__(self, qpos, image, actions=None, is_pad=None, pose_gt=None):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]
            if pose_gt is not None:
                pose_gt = pose_gt[:, :self.model.num_queries]

            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            valid = (~is_pad).unsqueeze(-1).to(dtype=all_l1.dtype)
            l1 = (all_l1 * valid).mean()
            valid_queries = valid.sum().clamp_min(1.0)
            arm_l1 = (
                (all_l1[..., :6] * valid).sum()
                / (valid_queries * 6)
            )
            hand_dim = all_l1.shape[-1] - 6
            hand_l1 = (
                (all_l1[..., 6:] * valid).sum()
                / (valid_queries * hand_dim)
            )
            loss_dict['l1'] = l1
            loss_dict['arm_l1'] = arm_l1
            loss_dict['hand_l1'] = hand_l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            if self.fk is not None:
                if pose_gt is None:
                    raise ValueError(
                        'FK pose supervision is enabled but pose_gt was not provided'
                    )
                fk_pose, fk_position, fk_rotation = masked_fk_pose_loss(
                    a_hat[..., :6],
                    pose_gt,
                    is_pad,
                    fk=self.fk,
                    action_mean=self.fk_action_mean,
                    action_std=self.fk_action_std,
                    pose_xyz_std=self.fk_pose_xyz_std,
                    rotation_weight=self.fk_rotation_weight,
                )
                loss_dict['fk_pose'] = fk_pose
                loss_dict['fk_position'] = fk_position
                loss_dict['fk_rotation'] = fk_rotation
                loss_dict['loss'] = (
                    loss_dict['loss'] + self.fk_pose_weight * fk_pose
                )
            return loss_dict
        else: # inference time
            a_hat, _, (_, _) = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


class CNNMLPPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model # decoder
        self.optimizer = optimizer

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None # TODO
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        if isinstance(image, dict):
            raise NotImplementedError("depth/dict observations are not supported by CNNMLP")
        image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, 0]
            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)
            loss_dict = dict()
            loss_dict['mse'] = mse
            loss_dict['loss'] = loss_dict['mse']
            return loss_dict
        else: # inference time
            a_hat = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld
