"""Differentiable RM65 forward kinematics and auxiliary pose loss.

The implementation parses the RM65-B URDF once at policy construction time.
Joint inputs are degrees and the output frame matches the recorded
``wrist_pose_base`` labels: RM65 base frame with a 180-degree Z correction.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn


DEFAULT_RM65_URDF = (
    "/home/ub/snap/Projects/DRO-Grasp-master/"
    "data/data_urdf/robot/rm65b/RM65-B.urdf"
)


def _parse_vector(text: str, default: Sequence[float]) -> np.ndarray:
    if not text:
        return np.asarray(default, dtype=np.float64)
    values = [float(value) for value in text.split()]
    if len(values) != 3:
        raise ValueError(f"URDF vector must contain three values, got {text!r}")
    return np.asarray(values, dtype=np.float64)


def _rotation_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray(((1, 0, 0), (0, c, -s), (0, s, c)), dtype=np.float64)


def _rotation_y(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray(((c, 0, s), (0, 1, 0), (-s, 0, c)), dtype=np.float64)


def _rotation_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray(((c, -s, 0), (s, c, 0), (0, 0, 1)), dtype=np.float64)


def _origin_transform(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = (
        _rotation_z(float(yaw))
        @ _rotation_y(float(pitch))
        @ _rotation_x(float(roll))
    )
    transform[:3, 3] = xyz
    return transform


def _parse_chain(urdf_path: Path, base_link: str, end_link: str):
    root = ET.parse(urdf_path).getroot()
    child_to_joint = {}
    for element in root.findall("joint"):
        parent = element.find("parent")
        child = element.find("child")
        if parent is None or child is None:
            continue
        origin = element.find("origin")
        axis = element.find("axis")
        joint = {
            "type": element.attrib.get("type", ""),
            "parent": parent.attrib["link"],
            "child": child.attrib["link"],
            "xyz": _parse_vector(
                "" if origin is None else origin.attrib.get("xyz", ""),
                (0.0, 0.0, 0.0),
            ),
            "rpy": _parse_vector(
                "" if origin is None else origin.attrib.get("rpy", ""),
                (0.0, 0.0, 0.0),
            ),
            "axis": _parse_vector(
                "" if axis is None else axis.attrib.get("xyz", ""),
                (0.0, 0.0, 1.0),
            ),
        }
        child_to_joint[joint["child"]] = joint

    chain = []
    current = end_link
    while current != base_link:
        if current not in child_to_joint:
            raise ValueError(
                f"Cannot construct URDF chain {base_link!r} -> {end_link!r}; "
                f"no joint leads to {current!r}"
            )
        joint = child_to_joint[current]
        chain.append(joint)
        current = joint["parent"]
    chain.reverse()
    return chain


def _axis_angle_rotation(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    axis = axis / axis.norm().clamp_min(1e-12)
    x, y, z = axis.unbind()
    zero = x * 0.0
    skew = torch.stack(
        (
            torch.stack((zero, -z, y)),
            torch.stack((z, zero, -x)),
            torch.stack((-y, x, zero)),
        )
    )
    eye = torch.eye(3, device=angle.device, dtype=angle.dtype)
    sin = angle.sin()[..., None, None]
    cos = angle.cos()[..., None, None]
    return eye + sin * skew + (1.0 - cos) * (skew @ skew)


def _homogeneous_rotation(rotation: torch.Tensor) -> torch.Tensor:
    shape = rotation.shape[:-2]
    transform = torch.zeros(*shape, 4, 4, device=rotation.device, dtype=rotation.dtype)
    transform[..., :3, :3] = rotation
    transform[..., 3, 3] = 1.0
    return transform


def euler_xyz_intrinsic_to_matrix(euler: torch.Tensor) -> torch.Tensor:
    """Convert intrinsic XYZ Euler angles in radians to rotation matrices."""
    roll, pitch, yaw = euler.unbind(dim=-1)
    cr, sr = roll.cos(), roll.sin()
    cp, sp = pitch.cos(), pitch.sin()
    cy, sy = yaw.cos(), yaw.sin()
    one = torch.ones_like(roll)
    zero = torch.zeros_like(roll)
    rx = torch.stack(
        (
            torch.stack((one, zero, zero), dim=-1),
            torch.stack((zero, cr, -sr), dim=-1),
            torch.stack((zero, sr, cr), dim=-1),
        ),
        dim=-2,
    )
    ry = torch.stack(
        (
            torch.stack((cp, zero, sp), dim=-1),
            torch.stack((zero, one, zero), dim=-1),
            torch.stack((-sp, zero, cp), dim=-1),
        ),
        dim=-2,
    )
    rz = torch.stack(
        (
            torch.stack((cy, -sy, zero), dim=-1),
            torch.stack((sy, cy, zero), dim=-1),
            torch.stack((zero, zero, one), dim=-1),
        ),
        dim=-2,
    )
    return rz @ ry @ rx


class RM65DifferentiableFK(nn.Module):
    """RM65 joint degrees ``[..., 6]`` to base-frame transform ``[..., 4, 4]``."""

    def __init__(
        self,
        urdf_path: str = DEFAULT_RM65_URDF,
        base_link: str = "base_link",
        end_link: str = "link_6",
        base_fix_z_deg: float = 180.0,
    ):
        super().__init__()
        path = Path(urdf_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"RM65 URDF not found: {path}")
        chain = _parse_chain(path, base_link, end_link)

        origins = []
        axes = []
        moving = []
        for joint in chain:
            joint_type = joint["type"]
            if joint_type not in ("fixed", "revolute", "continuous"):
                raise ValueError(f"Unsupported RM65 URDF joint type: {joint_type!r}")
            origins.append(_origin_transform(joint["xyz"], joint["rpy"]))
            axes.append(joint["axis"])
            moving.append(joint_type in ("revolute", "continuous"))

        if sum(moving) != 6:
            raise ValueError(f"RM65 FK expects 6 moving joints, URDF chain has {sum(moving)}")

        base_fix = np.eye(4, dtype=np.float64)
        base_fix[:3, :3] = _rotation_z(math.radians(float(base_fix_z_deg)))
        self.register_buffer(
            "origins", torch.as_tensor(np.stack(origins), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "axes", torch.as_tensor(np.stack(axes), dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "base_fix", torch.as_tensor(base_fix, dtype=torch.float32),
            persistent=False,
        )
        self.moving = tuple(moving)
        self.urdf_path = str(path)

    def forward(self, joint_degrees: torch.Tensor) -> torch.Tensor:
        if joint_degrees.shape[-1] < 6:
            raise ValueError(
                f"RM65 FK requires at least 6 joint values, got {joint_degrees.shape}"
            )
        if not torch.is_floating_point(joint_degrees):
            raise TypeError("RM65 joint tensor must use a floating dtype")

        input_dtype = joint_degrees.dtype
        calc_dtype = (
            torch.float32
            if input_dtype in (torch.float16, torch.bfloat16)
            else input_dtype
        )
        joints = joint_degrees[..., :6].to(dtype=calc_dtype)
        output_shape = joints.shape[:-1]
        joints = joints.reshape(-1, 6) * (math.pi / 180.0)
        count = joints.shape[0]
        transform = torch.eye(
            4, device=joints.device, dtype=joints.dtype,
        ).unsqueeze(0).expand(count, -1, -1)

        joint_index = 0
        for chain_index, is_moving in enumerate(self.moving):
            origin = self.origins[chain_index].to(device=joints.device, dtype=joints.dtype)
            transform = transform @ origin
            if is_moving:
                axis = self.axes[chain_index].to(device=joints.device, dtype=joints.dtype)
                rotation = _axis_angle_rotation(axis, joints[:, joint_index])
                transform = transform @ _homogeneous_rotation(rotation)
                joint_index += 1

        base_fix = self.base_fix.to(device=joints.device, dtype=joints.dtype)
        transform = base_fix @ transform
        return transform.reshape(*output_shape, 4, 4)


def masked_fk_pose_loss(
    pred_joint_normalized: torch.Tensor,
    target_pose: torch.Tensor,
    is_pad: torch.Tensor,
    *,
    fk: RM65DifferentiableFK,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    pose_xyz_std: torch.Tensor,
    rotation_weight: float = 1.0,
):
    """Pose loss from normalized predicted joints and raw GT ``[xyz, Euler XYZ]``.

    XYZ error is standardized by dataset position scale. Rotation uses smooth
    squared chordal distance on SO(3), avoiding Euler wrap discontinuities.
    """
    if pred_joint_normalized.shape[-1] < 6 or target_pose.shape[-1] < 6:
        raise ValueError("FK pose loss requires six joint and six pose values")
    if pred_joint_normalized.shape[:2] != target_pose.shape[:2]:
        raise ValueError(
            "FK pose prediction/target batch shape mismatch: "
            f"{pred_joint_normalized.shape} vs {target_pose.shape}"
        )

    pred = pred_joint_normalized[..., :6].float()
    target = target_pose[..., :6].float()
    mean = action_mean.to(device=pred.device, dtype=pred.dtype)[..., :6]
    std = action_std.to(device=pred.device, dtype=pred.dtype)[..., :6]
    xyz_std = pose_xyz_std.to(device=pred.device, dtype=pred.dtype)[..., :3]

    pred_joint_degrees = pred * std + mean
    pred_transform = fk(pred_joint_degrees)
    pred_xyz = pred_transform[..., :3, 3]
    pred_rotation = pred_transform[..., :3, :3]
    target_rotation = euler_xyz_intrinsic_to_matrix(target[..., 3:6])

    xyz_per_query = (
        (pred_xyz - target[..., :3]).abs() / xyz_std.clamp_min(1e-4)
    ).mean(dim=-1)
    rotation_per_query = 0.5 * (
        pred_rotation - target_rotation
    ).square().sum(dim=(-2, -1))

    if is_pad.ndim == 3:
        is_pad = is_pad.squeeze(-1)
    valid = (~is_pad).to(device=pred.device, dtype=pred.dtype)
    denominator = valid.sum().clamp_min(1.0)
    xyz_loss = (xyz_per_query * valid).sum() / denominator
    rotation_loss = (rotation_per_query * valid).sum() / denominator
    pose_loss = xyz_loss + float(rotation_weight) * rotation_loss
    return pose_loss, xyz_loss, rotation_loss
