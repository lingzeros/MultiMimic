#!/usr/bin/env python3
"""RM65 六关节角离线正运动学（FK）demo。

该实现与 Mymimic ``mimic_utils/arm_wrist_pose.py`` 的默认逻辑一致：

* 从 RM65-B URDF 解析 ``base_link -> link_6``；
* 输入六个关节角，单位为度；
* 在机械臂 base 系应用绕 Z 轴 180 度的修正；
* 输出 ``[x, y, z, rx, ry, rz]``，位置单位为米，姿态为 intrinsic XYZ
  欧拉角（弧度）。

脚本只做离线数学计算，不连接机械臂。

示例：
    python rm65_fk_demo.py --joints -6.94 12.31 88.53 13.85 34.45 -21.31
"""

from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


DEFAULT_URDF_PATH = Path(
    "/home/ub/snap/Projects/DRO-Grasp-master/"
    "data/data_urdf/robot/rm65b/RM65-B.urdf"
)
DEFAULT_BASE_FIX_Z_DEG = 180.0


def _rotation_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
        dtype=np.float64,
    )


def _rotation_y(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array(
        [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
        dtype=np.float64,
    )


def _rotation_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _transform_from_xyz_rpy(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    """URDF origin: fixed-axis RPY, equivalent to Rz(yaw) Ry(pitch) Rx(roll)."""
    roll, pitch, yaw = np.asarray(rpy, dtype=np.float64)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = (
        _rotation_z(float(yaw))
        @ _rotation_y(float(pitch))
        @ _rotation_x(float(roll))
    )
    transform[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return transform


def _axis_angle_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)

    x, y, z = axis / norm
    c, s = math.cos(angle), math.sin(angle)
    one_minus_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_minus_c, x * y * one_minus_c - z * s,
             x * z * one_minus_c + y * s],
            [y * x * one_minus_c + z * s, c + y * y * one_minus_c,
             y * z * one_minus_c - x * s],
            [z * x * one_minus_c - y * s, z * y * one_minus_c + x * s,
             c + z * z * one_minus_c],
        ],
        dtype=np.float64,
    )


def _matrix_to_euler_xyz_intrinsic(rotation: np.ndarray) -> np.ndarray:
    """旋转矩阵转 intrinsic XYZ 欧拉角，返回弧度。"""
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    sin_pitch = float(np.clip(-rotation[2, 0], -1.0, 1.0))
    pitch = math.asin(sin_pitch)
    cos_pitch = math.cos(pitch)

    if abs(cos_pitch) > 1e-8:
        roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
        yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    else:
        roll = 0.0
        yaw = math.atan2(-float(rotation[0, 1]), float(rotation[1, 1]))
    return np.array([roll, pitch, yaw], dtype=np.float64)


def _parse_vector(text: str, default: Sequence[float]) -> np.ndarray:
    if not text:
        return np.asarray(default, dtype=np.float64)
    values = [float(value) for value in text.split()]
    if len(values) != 3:
        raise ValueError(f"URDF 三维向量格式错误: {text!r}")
    return np.asarray(values, dtype=np.float64)


@lru_cache(maxsize=4)
def _parse_urdf_chain(
    urdf_path: str,
    base_link: str = "base_link",
    end_link: str = "link_6",
) -> Tuple[Dict[str, object], ...]:
    root = ET.parse(urdf_path).getroot()
    child_to_joint: Dict[str, Dict[str, object]] = {}

    for element in root.findall("joint"):
        parent_element = element.find("parent")
        child_element = element.find("child")
        if parent_element is None or child_element is None:
            continue

        origin_element = element.find("origin")
        axis_element = element.find("axis")
        joint = {
            "type": element.attrib.get("type", ""),
            "parent": parent_element.attrib["link"],
            "child": child_element.attrib["link"],
            "xyz": _parse_vector(
                "" if origin_element is None else origin_element.attrib.get("xyz", ""),
                (0.0, 0.0, 0.0),
            ),
            "rpy": _parse_vector(
                "" if origin_element is None else origin_element.attrib.get("rpy", ""),
                (0.0, 0.0, 0.0),
            ),
            "axis": _parse_vector(
                "" if axis_element is None else axis_element.attrib.get("xyz", ""),
                (0.0, 0.0, 1.0),
            ),
        }
        child_to_joint[str(joint["child"])] = joint

    chain: List[Dict[str, object]] = []
    current_link = end_link
    while current_link != base_link:
        if current_link not in child_to_joint:
            raise ValueError(
                f"URDF 中找不到连接到 {current_link!r} 的关节，"
                f"无法构造 {base_link!r} -> {end_link!r} 运动链"
            )
        joint = child_to_joint[current_link]
        chain.append(joint)
        current_link = str(joint["parent"])

    chain.reverse()
    return tuple(chain)


def forward_kinematics(
    joint_degrees: Sequence[float],
    urdf_path: Path = DEFAULT_URDF_PATH,
    base_fix_z_degrees: float = DEFAULT_BASE_FIX_Z_DEG,
) -> Tuple[np.ndarray, np.ndarray]:
    """将 RM65 六关节角转换为 base 系腕部位姿。

    Args:
        joint_degrees: 六个关节角，单位为度。
        urdf_path: RM65-B URDF 路径。
        base_fix_z_degrees: 与训练标注一致的 base Z 轴修正角，默认 180 度。

    Returns:
        pose6: ``[x, y, z, rx, ry, rz]``，xyz 为米，欧拉角为弧度。
        transform: 对应的 4x4 ``T_base_wrist`` 齐次变换矩阵。
    """
    joints = np.asarray(joint_degrees, dtype=np.float64).reshape(-1)
    if joints.size != 6:
        raise ValueError(f"必须输入 6 个关节角，当前输入 {joints.size} 个")
    if not np.all(np.isfinite(joints)):
        raise ValueError("关节角包含 NaN 或 Inf")

    urdf_path = Path(urdf_path).expanduser().resolve()
    if not urdf_path.is_file():
        raise FileNotFoundError(f"找不到 RM65-B URDF: {urdf_path}")

    chain = _parse_urdf_chain(str(urdf_path))
    joint_radians = np.deg2rad(joints)
    transform = np.eye(4, dtype=np.float64)
    joint_index = 0

    for joint in chain:
        transform = transform @ _transform_from_xyz_rpy(
            np.asarray(joint["xyz"]), np.asarray(joint["rpy"])
        )
        joint_type = str(joint["type"])

        if joint_type in ("revolute", "continuous"):
            if joint_index >= joint_radians.size:
                raise ValueError("URDF 运动链所需关节数超过输入关节数")
            joint_transform = np.eye(4, dtype=np.float64)
            joint_transform[:3, :3] = _axis_angle_rotation(
                np.asarray(joint["axis"]), joint_radians[joint_index]
            )
            transform = transform @ joint_transform
            joint_index += 1
        elif joint_type == "prismatic":
            raise ValueError("当前 RM65 demo 不支持移动关节")
        elif joint_type != "fixed":
            raise ValueError(f"不支持的 URDF 关节类型: {joint_type!r}")

    if joint_index != 6:
        raise ValueError(f"URDF 运动链使用了 {joint_index} 个旋转关节，预期为 6 个")

    base_fix = np.eye(4, dtype=np.float64)
    base_fix[:3, :3] = _rotation_z(math.radians(base_fix_z_degrees))
    transform = base_fix @ transform

    pose = np.empty(6, dtype=np.float64)
    pose[:3] = transform[:3, 3]
    pose[3:] = _matrix_to_euler_xyz_intrinsic(transform[:3, :3])
    return pose, transform


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="将 RM65 六个关节角（度）离线 FK 为 base 系末端位姿"
    )
    parser.add_argument(
        "--joints",
        type=float,
        nargs=6,
        required=True,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        help="六个关节角，单位为度",
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        default=DEFAULT_URDF_PATH,
        help=f"RM65-B URDF 路径（默认: {DEFAULT_URDF_PATH}）",
    )
    parser.add_argument(
        "--base-fix-z",
        type=float,
        default=DEFAULT_BASE_FIX_Z_DEG,
        help="base 坐标系绕 Z 轴修正角，单位为度（默认: 180）",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    pose, transform = forward_kinematics(
        args.joints,
        urdf_path=args.urdf,
        base_fix_z_degrees=args.base_fix_z,
    )

    np.set_printoptions(precision=6, suppress=True)
    print(f"joint_deg       = {np.asarray(args.joints)}")
    print(f"position_m      = {pose[:3]}")
    print(f"position_mm     = {pose[:3] * 1000.0}")
    print(f"euler_xyz_rad   = {pose[3:]}")
    print(f"euler_xyz_deg   = {np.rad2deg(pose[3:])}")
    print("T_base_wrist =")
    print(transform)


if __name__ == "__main__":
    main()
