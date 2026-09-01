#!/usr/bin/env python3
"""Compare Human IK trajectories with Robot RM65 demonstrations."""

import argparse
import glob
import json
import math
import os
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import torch

from detr.models.rm65_fk import (
    DEFAULT_RM65_URDF,
    RM65DifferentiableFK,
    euler_xyz_intrinsic_to_matrix,
)


DEFAULT_HUMAN_GLOB = (
    "/mnt/additional/Data/DexMimic_Data/Human_data/Peach_in_bowl/"
    "episode_*.hdf5"
)
DEFAULT_ROBOT_GLOB = (
    "/mnt/additional/Data/DexMimic_Data/Robot_data_ACT/Peach_in_bowl/inspire/"
    "episode_*.hdf5"
)
JOINT_NAMES = [f"joint_{index}" for index in range(1, 7)]


def scalar_summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "p90": float(np.percentile(values, 90)),
        "max": float(np.max(values)),
    }


def per_joint_summary(values):
    values = np.asarray(values, dtype=np.float64)
    result = {}
    for index, name in enumerate(JOINT_NAMES):
        column = values[:, index]
        result[name] = {
            "mean": float(np.mean(column)),
            "std": float(np.std(column)),
            "p05": float(np.percentile(column, 5)),
            "p50": float(np.percentile(column, 50)),
            "p90": float(np.percentile(column, 90)),
            "p95": float(np.percentile(column, 95)),
            "min": float(np.min(column)),
            "max": float(np.max(column)),
        }
    return result


def read_episodes(paths, include_pose=False):
    joints = []
    velocities = []
    accelerations = []
    action_joints = []
    action_poses = []
    lengths = []
    sample_rates = []

    for path in paths:
        with h5py.File(path, "r") as root:
            if "/observations/qpos" not in root:
                raise KeyError(f"/observations/qpos missing in {path}")
            qpos = np.asarray(root["/observations/qpos"], dtype=np.float64)[:, :6]
            if qpos.ndim != 2 or qpos.shape[1] != 6:
                raise ValueError(f"unexpected qpos shape {qpos.shape} in {path}")
            hz = float(root.attrs.get("sample_hz", root.attrs.get("camera_fps", 30)))
            if hz <= 0:
                raise ValueError(f"invalid sample rate {hz} in {path}")

            joints.append(qpos)
            lengths.append(len(qpos))
            sample_rates.append(hz)
            if len(qpos) >= 2:
                velocities.append(np.diff(qpos, axis=0) * hz)
            if len(qpos) >= 3:
                accelerations.append(np.diff(qpos, n=2, axis=0) * hz * hz)

            if include_pose:
                for key in ("/action", "/action_wrist_pose_base"):
                    if key not in root:
                        raise KeyError(f"{key} missing in {path}")
                action = np.asarray(root["/action"], dtype=np.float64)[:, :6]
                pose = np.asarray(root["/action_wrist_pose_base"], dtype=np.float64)
                if action.shape != (len(qpos), 6) or pose.shape != (len(qpos), 6):
                    raise ValueError(
                        f"action/pose shape mismatch in {path}: "
                        f"qpos={qpos.shape}, action={action.shape}, pose={pose.shape}"
                    )
                action_joints.append(action)
                action_poses.append(pose)

    return {
        "joints": np.concatenate(joints),
        "velocity": np.concatenate(velocities),
        "acceleration": np.concatenate(accelerations),
        "action_joints": np.concatenate(action_joints) if action_joints else None,
        "action_poses": np.concatenate(action_poses) if action_poses else None,
        "lengths": lengths,
        "sample_rates": sample_rates,
    }


def read_joint_limits(urdf_path):
    root = ET.parse(urdf_path).getroot()
    limits = []
    for joint in root.findall("joint"):
        if joint.attrib.get("type") != "revolute":
            continue
        limit = joint.find("limit")
        if limit is None:
            continue
        limits.append(
            (
                math.degrees(float(limit.attrib["lower"])),
                math.degrees(float(limit.attrib["upper"])),
            )
        )
    if len(limits) != 6:
        raise ValueError(f"expected six revolute joint limits, found {len(limits)}")
    return np.asarray(limits, dtype=np.float64)


def fk_errors(action_joints, target_pose, urdf_path, batch_size=4096):
    fk = RM65DifferentiableFK(urdf_path=urdf_path).eval()
    position_errors = []
    rotation_errors = []
    with torch.inference_mode():
        for start in range(0, len(action_joints), batch_size):
            joints = torch.from_numpy(action_joints[start:start + batch_size]).float()
            pose = torch.from_numpy(target_pose[start:start + batch_size]).float()
            transform = fk(joints)
            position = torch.linalg.vector_norm(
                transform[:, :3, 3] - pose[:, :3], dim=-1,
            )
            target_rotation = euler_xyz_intrinsic_to_matrix(pose[:, 3:6])
            relative = transform[:, :3, :3].transpose(-1, -2) @ target_rotation
            cosine = ((relative.diagonal(dim1=-2, dim2=-1).sum(-1) - 1.0) / 2.0)
            angle = torch.acos(cosine.clamp(-1.0, 1.0)) * (180.0 / math.pi)
            position_errors.append(position.cpu().numpy())
            rotation_errors.append(angle.cpu().numpy())
    return np.concatenate(position_errors), np.concatenate(rotation_errors)


def motion_summary(values, unit):
    absolute = np.abs(values)
    vector_norm = np.linalg.norm(values, axis=1)
    result = {
        "unit": unit,
        "absolute_all_joints": scalar_summary(absolute.reshape(-1)),
        "vector_l2": scalar_summary(vector_norm),
        "per_joint": {},
    }
    for index, name in enumerate(JOINT_NAMES):
        result["per_joint"][name] = scalar_summary(absolute[:, index])
    return result


def limit_summary(joints, limits):
    below = joints < limits[:, 0]
    above = joints > limits[:, 1]
    violation = below | above
    per_joint = {}
    for index, name in enumerate(JOINT_NAMES):
        per_joint[name] = {
            "lower_deg": float(limits[index, 0]),
            "upper_deg": float(limits[index, 1]),
            "below_count": int(below[:, index].sum()),
            "above_count": int(above[:, index].sum()),
            "violation_rate": float(violation[:, index].mean()),
        }
    return {
        "scalar_violation_count": int(violation.sum()),
        "scalar_total": int(violation.size),
        "scalar_violation_rate": float(violation.mean()),
        "frames_with_any_violation": int(violation.any(axis=1).sum()),
        "frame_total": int(len(joints)),
        "frame_violation_rate": float(violation.any(axis=1).mean()),
        "per_joint": per_joint,
    }


def distribution_comparison(human, robot):
    human_stats = per_joint_summary(human)
    robot_stats = per_joint_summary(robot)
    comparison = {}
    quantiles = np.linspace(0.0, 1.0, 1001)
    for index, name in enumerate(JOINT_NAMES):
        h = human[:, index]
        r = robot[:, index]
        pooled_std = math.sqrt((float(np.var(h)) + float(np.var(r))) / 2.0)
        comparison[name] = {
            "human": human_stats[name],
            "robot": robot_stats[name],
            "mean_difference_human_minus_robot_deg": float(np.mean(h) - np.mean(r)),
            "standardized_mean_difference": (
                float((np.mean(h) - np.mean(r)) / pooled_std)
                if pooled_std > 0 else None
            ),
            "quantile_distance_mean_deg": float(
                np.mean(np.abs(np.quantile(h, quantiles) - np.quantile(r, quantiles)))
            ),
            "human_outside_robot_p05_p95_rate": float(
                np.mean((h < np.percentile(r, 5)) | (h > np.percentile(r, 95)))
            ),
        }
    return comparison


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--human_glob", default=DEFAULT_HUMAN_GLOB)
    parser.add_argument("--robot_glob", default=DEFAULT_ROBOT_GLOB)
    parser.add_argument("--urdf", default=DEFAULT_RM65_URDF)
    parser.add_argument("--output", default="human_robot_arm_statistics.json")
    args = parser.parse_args()

    human_paths = sorted(glob.glob(args.human_glob))
    robot_paths = sorted(glob.glob(args.robot_glob))
    if not human_paths or not robot_paths:
        raise RuntimeError(
            f"no input episodes: human={len(human_paths)}, robot={len(robot_paths)}"
        )

    human = read_episodes(human_paths, include_pose=True)
    robot = read_episodes(robot_paths, include_pose=False)
    position_error, rotation_error = fk_errors(
        human["action_joints"], human["action_poses"], args.urdf,
    )
    limits = read_joint_limits(args.urdf)

    report = {
        "metadata": {
            "human_glob": args.human_glob,
            "robot_glob": args.robot_glob,
            "rm65_urdf": os.path.abspath(args.urdf),
            "human_episode_count": len(human_paths),
            "robot_episode_count": len(robot_paths),
            "human_frame_count": int(len(human["joints"])),
            "robot_frame_count": int(len(robot["joints"])),
            "human_sample_rates_hz": sorted(set(human["sample_rates"])),
            "robot_sample_rates_hz": sorted(set(robot["sample_rates"])),
            "rotation_error_definition": "SO(3) geodesic angle",
            "velocity_definition": "absolute first difference multiplied by per-episode Hz",
            "acceleration_definition": "absolute second difference multiplied by per-episode Hz squared",
        },
        "human_fk_error": {
            "position": {"unit": "m", **scalar_summary(position_error)},
            "rotation": {"unit": "degree", **scalar_summary(rotation_error)},
        },
        "human_joint_velocity": motion_summary(human["velocity"], "degree/second"),
        "human_joint_acceleration": motion_summary(
            human["acceleration"], "degree/second^2",
        ),
        "human_joint_limit_violations": limit_summary(human["joints"], limits),
        "human_vs_robot_arm_joint_distribution": {
            "unit": "degree",
            "per_joint": distribution_comparison(human["joints"], robot["joints"]),
        },
    }

    output_path = os.path.abspath(args.output)
    with open(output_path, "w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(output_path)


if __name__ == "__main__":
    main()
