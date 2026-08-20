# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from ..util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other policy_models than torchvision.policy_models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        # for name, parameter in backbone.named_parameters(): # only train later layers # TODO do we want this?
        #     if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
        #         parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor):
        xs = self.body(tensor)
        return xs
        # out: Dict[str, NestedTensor] = {}
        # for name, x in xs.items():
        #     m = tensor_list.mask
        #     assert m is not None
        #     mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
        #     out[name] = NestedTensor(x, mask)
        # return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d) # pretrained # TODO do we want frozen batch_norm??
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class DINOv2Backbone(nn.Module):
    """Expose DINOv2 patch tokens as an ACT-compatible 2-D feature map."""

    def __init__(
        self,
        repo_path: Path,
        weights_path: Path,
        freeze: bool = True,
        input_size=(210, 280),
    ):
        super().__init__()
        if not repo_path.is_dir():
            raise FileNotFoundError(f"DINOv2 repository not found: {repo_path}")
        if not weights_path.is_file():
            raise FileNotFoundError(f"DINOv2 weights not found: {weights_path}")

        self.body = torch.hub.load(
            str(repo_path),
            "dinov2_vits14",
            source="local",
            pretrained=False,
        )
        state_dict = torch.load(
            weights_path,
            map_location="cpu",
            weights_only=True,
        )
        self.body.load_state_dict(state_dict, strict=True)
        self.num_channels = 384
        self.freeze = freeze
        self.input_size = input_size

        if self.freeze:
            self.body.requires_grad_(False)
            self.body.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze:
            # Keep the frozen pretrained encoder deterministic when policy.train()
            # recursively changes all child modules to training mode.
            self.body.eval()
        return self

    def forward(self, tensor):
        # DINOv2-S/14 requires both spatial dimensions to be divisible by 14.
        # 210x280 preserves the 4:3 aspect ratio of the 480x640 ACT images and
        # yields a 15x20 feature map, matching ResNet18's output token count.
        if tensor.shape[-2:] != self.input_size:
            tensor = F.interpolate(
                tensor,
                size=self.input_size,
                mode="bilinear",
                align_corners=False,
            )

        def extract_feature_map():
            return self.body.get_intermediate_layers(
                tensor,
                n=1,
                reshape=True,
                norm=True,
            )[0]

        if self.freeze:
            with torch.no_grad():
                feature = extract_feature_map()
        else:
            feature = extract_feature_map()

        return OrderedDict({"0": feature})


class DepthEncoder(nn.Module):
    """Trainable encoder for a normalized single-channel 480x640 depth map.

    Five stride-2 stages produce a 15x20 feature map, matching both the
    ResNet18 output at 480x640 and the DINOv2-S/14 output at 210x280.
    GroupNorm is used because ACT commonly trains with small batch sizes.
    """

    def __init__(self):
        super().__init__()
        channels = (1, 32, 64, 128, 192, 256)
        layers = []
        for index, (in_channels, out_channels) in enumerate(
            zip(channels[:-1], channels[1:])
        ):
            kernel_size = 7 if index == 0 else 3
            padding = 3 if index == 0 else 1
            layers.extend(
                [
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=kernel_size,
                        stride=2,
                        padding=padding,
                        bias=False,
                    ),
                    nn.GroupNorm(min(8, out_channels), out_channels),
                    nn.GELU(),
                ]
            )
        self.body = nn.Sequential(*layers)
        self.num_channels = channels[-1]

    def forward(self, tensor):
        if tensor.ndim != 4 or tensor.shape[1] != 1:
            raise ValueError(
                f"depth encoder expects [B,1,H,W], got {tuple(tensor.shape)}"
            )
        return OrderedDict({"0": self.body(tensor)})


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    if args.backbone == "dinov2_vits14":
        project_root = Path(__file__).resolve().parents[2]
        backbone = DINOv2Backbone(
            repo_path=project_root / "dinov2",
            weights_path=(
                project_root
                / "dinov2"
                / "checkpoints"
                / "dinov2_vits14_pretrain.pth"
            ),
            freeze=True,
        )
    else:
        train_backbone = args.lr_backbone > 0
        return_interm_layers = args.masks
        backbone = Backbone(
            args.backbone,
            train_backbone,
            return_interm_layers,
            args.dilation,
        )
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model


def build_depth_encoder(args):
    """Build the optional depth encoder with ACT-compatible position encoding."""
    encoder = DepthEncoder()
    model = Joiner(encoder, build_position_encoding(args))
    model.num_channels = encoder.num_channels
    return model
