#!/usr/bin/env python3
"""
DexMimic 真机推理部署脚本。

模块划分（初始化 / 数据管线 / 推理 相互隔离）：
  - SharedDataBus      : 多进程共享字典，定义键名与读写接口
  - ArmController      : 机械臂初始化 + 逆解 CANFD 控制循环
  - HandController     : 灵巧手初始化 + finger_move 控制循环
  - CameraSystem       : RealSense front 相机（仅 RGB 供模型）
  - HandPointCloudGenerator : DRO-Grasp URDF 从关节角合成手部点云
  - ModelEngine        : DexMimic 权重加载 + forward_eval
  - ObservationBuilder : 传感器数据 → 模型 batch
  - ActionDispatcher   : 模型输出 → 共享总线指令
  - InferenceLoop      : 主进程推理调度（相机 + 模型 + 下发）

用法：
  python inference.py --ckpt checkpoints/DexMimic/.../dexmimic_best.pt
  python inference.py --dry-run --ckpt ...   # 不连接机械臂/灵巧手
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# torch.cuda 初始化时会触发 pynvml 弃用 FutureWarning；须在 import torch 之前过滤
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*pynvml.*",
)
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    )

import cv2
import numpy as np
import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "model"))
sys.path.insert(0, str(_ROOT / "external"))

import model.DexMimic  # noqa: F401  注册 dexmimic algo
from model.dexmimic_config import DexMimicConfig
from external.egomimic.algo import algo_factory

from constants import (
    ARM_IP,
    ARM_EULER_FLAG,
    ARM_TOOL_FRAME,
    ARM_WORK_FRAME,
    DEFAULT_CAN_INTERFACE,
    DEFAULT_HAND_JOINT,
    DEFAULT_HAND_TYPE,
    DEFAULT_SPEED,
    DEFAULT_TORQUE,
    DRO_GRASP_ROOT,
    IMAGE_FPS,
    IMAGE_SIZE,
    TEST_CKPT_PATH_DEC,
    TEST_CKPT_PATH_ENC,
    DEXMIMIC_PATH,
)

from mimic_utils.arm_wrist_pose import wrist_pose_base_from_arm_joints

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None

from mimic_utils.hand_joint_utils import (
    apply_l25_reserved_joints,
    joints_rad_to_raw255,
)
from mimic_utils.hand_pointcloud_gen import HandPointCloudGenerator, DEFAULT_DRO_GRASP_ROOT

try:
    from scipy.spatial.transform import Rotation
except ImportError:
    Rotation = None


# ══════════════════════════════════════════════════════════════════════════════
# 工具函数（点云 / 关节映射，与训练保持一致）
# ══════════════════════════════════════════════════════════════════════════════
def resample_points(cloud: torch.Tensor, num_points: int) -> torch.Tensor:
    """cloud: [T, N, 3] -> [T, num_points, 3]（推理固定步长采样）"""
    n = cloud.shape[1]
    if n > num_points:
        idx = torch.linspace(0, n - 1, steps=num_points).long()
        cloud = cloud[:, idx, :]
    elif n < num_points:
        idx = torch.arange(num_points) % n
        cloud = cloud[:, idx, :]
    return cloud


def normalize_cloud(cloud: torch.Tensor) -> torch.Tensor:
    """全局序列归一化，与 mimic_train / emEncoder 一致。"""
    centroid = cloud.view(-1, 3).mean(dim=0, keepdim=True)
    cloud = cloud - centroid
    dist = torch.sqrt(torch.sum(cloud ** 2, dim=-1))
    max_dist = torch.clamp(torch.max(dist), min=1e-6)
    return cloud / max_dist


def resolve_hand_decode_mode(
    hand_decode: str,
    train_args: Optional[dict],
    mlp_loaded: bool,
) -> str:
    """解析手部解码路径；与训练 direct_joint_lambda 对齐。"""
    if hand_decode != "auto":
        return hand_decode
    if train_args and train_args.get("direct_joint_lambda", 1.0) == 0:
        return "em_decoder"
    return "mlp_hand" if mlp_loaded else "em_decoder"


def euler_to_pose7(xyz_euler: np.ndarray) -> np.ndarray:
    """6D (xyz + euler_xyz) → 7D (xyz + quat_xyzw)，供 RM 逆解（euler=0 模式）。"""
    xyz = xyz_euler[:3]
    eul = xyz_euler[3:6]
    if Rotation is None:
        raise ImportError("需要 scipy 以转换欧拉角 → 四元数")
    quat = Rotation.from_euler("xyz", eul).as_quat()
    return np.concatenate([xyz, quat]).astype(np.float64)


# ══════════════════════════════════════════════════════════════════════════════
# 1. 共享数据总线（进程间指令 / 状态传输）
# ══════════════════════════════════════════════════════════════════════════════
class SharedDataBus:
    """
    主进程 ↔ 臂/手子进程 的共享字典封装。

    键名约定：
      wrist_pose_base : [6] 当前末端位姿（臂进程写，主进程读）
      hand_qpos_state : [25] 当前灵巧手驱动值 0-255（手进程写，主进程读）
      arm_pose_cmd    : [6] 目标末端位姿（主进程写，臂进程读）
      hand_qpos_cmd   : [25] 灵巧手 0-255（主进程写，手进程读）
      EXIT            : bool  退出标志
    """

    KEY_WRIST_POSE = "wrist_pose_base"
    KEY_ARM_CMD = "arm_pose_cmd"
    KEY_HAND_CMD = "hand_qpos_cmd"
    KEY_HAND_STATE = "hand_qpos_state"
    KEY_EXIT = "EXIT"

    def __init__(self, manager_dict: Dict[str, Any]):
        self._d = manager_dict

    @classmethod
    def create(cls, manager) -> "SharedDataBus":
        d = manager.dict()
        d[cls.KEY_WRIST_POSE] = None
        d[cls.KEY_ARM_CMD] = None
        d[cls.KEY_HAND_CMD] = None
        d[cls.KEY_HAND_STATE] = None
        d[cls.KEY_EXIT] = False
        return cls(d)

    @property
    def raw(self) -> Dict[str, Any]:
        return self._d

    def request_exit(self):
        self._d[self.KEY_EXIT] = True

    def should_exit(self) -> bool:
        return bool(self._d.get(self.KEY_EXIT, False))

    def publish_arm_cmd(self, pose6: np.ndarray):
        self._d[self.KEY_ARM_CMD] = np.asarray(pose6, dtype=np.float64).copy()

    def publish_hand_cmd(self, qpos255: list):
        self._d[self.KEY_HAND_CMD] = list(qpos255)

    def read_arm_cmd(self) -> Optional[np.ndarray]:
        v = self._d.get(self.KEY_ARM_CMD)
        return None if v is None else np.asarray(v, dtype=np.float64)

    def read_hand_cmd(self) -> Optional[list]:
        return self._d.get(self.KEY_HAND_CMD)

    def publish_wrist_pose(self, pose6: np.ndarray):
        self._d[self.KEY_WRIST_POSE] = np.asarray(pose6, dtype=np.float64).copy()

    def read_wrist_pose(self) -> Optional[np.ndarray]:
        v = self._d.get(self.KEY_WRIST_POSE)
        return None if v is None else np.asarray(v, dtype=np.float64)

    def publish_hand_state(self, qpos255: list):
        self._d[self.KEY_HAND_STATE] = list(qpos255)

    def read_hand_state(self) -> Optional[np.ndarray]:
        v = self._d.get(self.KEY_HAND_STATE)
        return None if v is None else np.asarray(v, dtype=np.float64)


# ══════════════════════════════════════════════════════════════════════════════
# 2. 机械臂控制器（初始化 + 执行循环）
# ══════════════════════════════════════════════════════════════════════════════
class ArmController:
    """RM65 机械臂：连接、坐标系、逆解 CANFD 透传。"""

    def __init__(
        self,
        ip: str = ARM_IP,
        work_frame: str = ARM_WORK_FRAME,
        tool_frame: str = ARM_TOOL_FRAME,
        init_joints: Optional[list] = None,
        sim: bool = False,
        dro_grasp_root: str = DRO_GRASP_ROOT,
    ):
        self.ip = ip
        self.work_frame = work_frame
        self.tool_frame = tool_frame
        self.init_joints = init_joints or [-10, 25, 70, 70, 27, -20]
        self.sim = sim
        self.dro_grasp_root = dro_grasp_root
        self._arm = None
        self._algo = None

    def connect(self):
        robot_path = os.environ.get(
            "ROBOT_CONTROL_PATH",
            "/home/ub/TeleOperation_DataCollection/Robot_control",
        )
        if robot_path not in sys.path:
            sys.path.append(robot_path)
        from Robotic_Arm.rm_robot_interface import (  # type: ignore
            Algo,
            RoboticArm,
            rm_force_type_e,
            rm_inverse_kinematics_params_t,
            rm_robot_arm_model_e,
            rm_thread_mode_e,
        )

        arm_model = rm_robot_arm_model_e.RM_MODEL_RM_65_E
        force_type = rm_force_type_e.RM_MODEL_RM_B_E
        self._algo = Algo(arm_model, force_type)
        self._algo.rm_algo_set_redundant_parameter_traversal_mode(False)
        self._arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        handle = self._arm.rm_create_robot_arm(self.ip, 8080)
        print(f"[Arm] 已连接 handle.id={handle.id}")

        ret_w = self._arm.rm_change_work_frame(self.work_frame)
        ret_t = self._arm.rm_change_tool_frame(self.tool_frame)
        print(f"[Arm] work_frame={self.work_frame!r} ret={ret_w}")
        print(f"[Arm] tool_frame={self.tool_frame!r} ret={ret_t}")
        self._arm.rm_set_arm_run_mode(0 if self.sim else 1)
        self._arm.rm_movej(self.init_joints, 60, 0, 0, 1)
        time.sleep(1.0)
        self._ik_params_cls = rm_inverse_kinematics_params_t

    def disconnect(self):
        if self._arm is not None:
            self._arm.rm_delete_robot_arm()
            try:
                from Robotic_Arm.rm_robot_interface import RoboticArm  # type: ignore
                RoboticArm.rm_destroy()
            except Exception:
                pass
            self._arm = None

    @staticmethod
    def _joint_diff(target, current):
        return np.asarray(target, dtype=np.float64) - np.asarray(current, dtype=np.float64)

    @classmethod
    def _pose_follow(cls, q_out, c_q):
        """单步限幅跟随（与 robotpipelinesample/arm_move.py PoseFollow 一致）。"""
        tar_q = np.asarray(q_out, dtype=np.float64).copy()
        c_q = np.asarray(c_q, dtype=np.float64)
        diff = tar_q - c_q
        index = np.abs(diff) >= 10
        if np.any(index):
            tar_q[index] = (c_q + 9.5 * (diff / np.abs(diff)))[index]
        return tar_q

    def get_wrist_pose_base(self) -> np.ndarray:
        """从关节角 FK 得到 wrist_pose_base（与训练标注同一 base 系）。"""
        current_q = self._arm.rm_get_joint_degree()[1]
        return wrist_pose_base_from_arm_joints(current_q, dro_root=self.dro_grasp_root)

    def move_to_pose6(self, pose6: np.ndarray, euler_flag: int = ARM_EULER_FLAG) -> bool:
        """6D wrist_pose_base → 逆解 → CANFD 透传。"""
        pose6 = np.asarray(pose6, dtype=np.float64).reshape(-1)[:6]
        if euler_flag == 0:
            pose_input = euler_to_pose7(pose6)
        else:
            pose_input = pose6
        current_q = self._arm.rm_get_joint_degree()[1]
        params = self._ik_params_cls(current_q, pose_input, euler_flag)
        q_out = self._algo.rm_algo_inverse_kinematics(params)
        if q_out[0] != 0:
            print(f"[Arm] 逆解失败 code={q_out[0]} target={pose6} "
                  f"(work={self.work_frame} tool={self.tool_frame} euler={euler_flag})")
            return False
        target_q = np.asarray(q_out[1], dtype=np.float64)
        current_q = np.asarray(current_q, dtype=np.float64)
        if target_q.shape != current_q.shape:
            n = min(target_q.size, current_q.size)
            print(f"[Arm] 警告：关节维度不一致 target={target_q.shape} current={current_q.shape}，截断为 {n}")
            target_q = target_q[:n]
            current_q = current_q[:n]
        if np.max(np.abs(target_q - current_q)) >= 10:
            target_q = self._pose_follow(target_q, current_q)
        self._arm.rm_movej_canfd(target_q.tolist(), False)
        return True

    @staticmethod
    def run_loop(bus: SharedDataBus, cfg: "InferenceConfig"):
        """子进程：轮询 arm_pose_cmd 并执行，同时回写 wrist_pose_base。"""
        ctrl = ArmController(
            ip=cfg.arm_ip,
            work_frame=cfg.work_frame,
            tool_frame=cfg.tool_frame,
            sim=cfg.sim,
            dro_grasp_root=cfg.dro_grasp_root,
        )
        try:
            ctrl.connect()
            dt = 1.0 / cfg.control_hz
            while not bus.should_exit():
                try:
                    bus.publish_wrist_pose(ctrl.get_wrist_pose_base())
                except Exception as exc:
                    print(f"[Arm] 读取位姿失败: {exc}")

                cmd = bus.read_arm_cmd()
                if cmd is not None:
                    ctrl.move_to_pose6(cmd, euler_flag=cfg.arm_euler_flag)
                time.sleep(dt)
        finally:
            ctrl.disconnect()
            print("[Arm] 进程已退出")


# ══════════════════════════════════════════════════════════════════════════════
# 3. 灵巧手控制器（初始化 + 执行循环）
# ══════════════════════════════════════════════════════════════════════════════
class HandController:
    """LinkerHand L21：CAN 初始化 + finger_move。"""

    def __init__(
        self,
        hand_joint: str = DEFAULT_HAND_JOINT,
        hand_type: str = DEFAULT_HAND_TYPE,
        can: str = DEFAULT_CAN_INTERFACE,
        speed: Optional[list] = None,
        torque: Optional[list] = None,
    ):
        self.hand_joint = hand_joint
        self.hand_type = hand_type
        self.can = can
        self.speed = speed or [DEFAULT_SPEED, 220, 220, 220, 220]
        self.torque = torque or [DEFAULT_TORQUE] * 5
        self._hand = None

    def connect(self):
        from linker_hand_python_sdk.LinkerHand.linker_hand_api import LinkerHandApi  # type: ignore

        self._hand = LinkerHandApi(
            hand_joint=self.hand_joint, hand_type=self.hand_type, can=self.can
        )
        self._hand.set_speed(speed=self.speed)
        self._hand.set_torque(torque=self.torque)
        init_pose = [128] * 25
        self._hand.finger_move(pose=init_pose)
        time.sleep(0.5)
        self._last_qpos = init_pose
        print("[Hand] LinkerHand 初始化完成")

    def move(self, qpos255: list):
        self._hand.finger_move(pose=qpos255)
        self._last_qpos = list(qpos255)

    @staticmethod
    def run_loop(bus: SharedDataBus, cfg: "InferenceConfig"):
        """子进程：轮询 hand_qpos_cmd 并执行，同时回写 hand_qpos_state。"""
        ctrl = HandController(can=cfg.can_interface)
        try:
            ctrl.connect()
            bus.publish_hand_state(ctrl._last_qpos)
            dt = 1.0 / cfg.control_hz
            while not bus.should_exit():
                cmd = bus.read_hand_cmd()
                if cmd is not None:
                    try:
                        ctrl.move(cmd)
                        bus.publish_hand_state(ctrl._last_qpos)
                    except Exception as exc:
                        print(f"[Hand] finger_move 失败: {exc}")
                time.sleep(dt)
        finally:
            print("[Hand] 进程已退出")


# ══════════════════════════════════════════════════════════════════════════════
# 4. 相机系统（仅 front RGB，供 DexMimic 视觉输入）
# ══════════════════════════════════════════════════════════════════════════════
FRONT_CAMERA_ID = "216322073710"


@dataclass
class CameraFrame:
    front_rgb: np.ndarray          # [H, W, 3] uint8 BGR


class CameraSystem:
    """RealSense 单路 front 相机，仅采集 RGB。"""

    def __init__(
        self,
        device_id: str = FRONT_CAMERA_ID,
        width: int = IMAGE_SIZE["width"],
        height: int = IMAGE_SIZE["height"],
        fps: int = IMAGE_FPS,
    ):
        if rs is None:
            raise ImportError("需要 pyrealsense2：pip install pyrealsense2")
        self.device_id = device_id
        self.width = width
        self.height = height
        self.fps = fps
        self._pipeline = None

    def start(self):
        cfg = rs.config()
        cfg.enable_device(self.device_id)
        cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        self._pipeline = rs.pipeline()
        self._pipeline.start(cfg)
        print(f"[Camera] front RGB={self.device_id}")

    def stop(self):
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None

    def grab(self) -> CameraFrame:
        frames = self._pipeline.wait_for_frames()
        color = frames.get_color_frame()
        if not color:
            raise RuntimeError("相机帧不完整")
        return CameraFrame(front_rgb=np.asanyarray(color.get_data()))


# ══════════════════════════════════════════════════════════════════════════════
# 5. 模型引擎（加载 + 推理）
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class PoseStats:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def load(
        cls,
        ckpt: dict,
        pose_stats_path: Optional[str],
        robot_arm_supervision: str = "joint",
    ) -> "PoseStats":
        blob = None
        if "pose_stats" in ckpt:
            blob = ckpt["pose_stats"]
        else:
            candidates = []
            if pose_stats_path:
                candidates.append(Path(pose_stats_path))
            ckpt_dir = ckpt.get("_ckpt_dir")
            if ckpt_dir:
                candidates.append(Path(ckpt_dir) / "pose_stats.json")
            for path in candidates:
                if path.exists():
                    with open(path, "r", encoding="utf-8") as f:
                        blob = json.load(f)
                    break
        if blob is None:
            raise FileNotFoundError("checkpoint 中无 pose_stats，请通过 --pose-stats 指定")
        if robot_arm_supervision == "joint" and "joint" in blob:
            s = blob["joint"]
        elif "shared" in blob:
            s = blob["shared"]
        elif "wrist" in blob:
            s = blob["wrist"]
        else:
            raise FileNotFoundError("pose_stats 中缺少 joint/shared/wrist 条目")
        return cls(mean=np.asarray(s["mean"], dtype=np.float32),
                   std=np.asarray(s["std"], dtype=np.float32))


class ModelEngine:
    """DexMimic 权重加载与 forward_eval 封装。"""

    def __init__(self, cfg: "InferenceConfig", device: torch.device):
        self.cfg = cfg
        self.device = device
        self.algo = None
        self.pose_stats: Optional[PoseStats] = None
        self.seq_len = cfg.seq_len

    def _build_config(self) -> DexMimicConfig:
        c = DexMimicConfig()
        with c.values_unlocked():
            c.train.seq_length = self.cfg.seq_len
            c.train.seq_length_hand = self.cfg.seq_len
            c.algo.act.window_size = self.cfg.window_size
            c.algo.act.joint_dim = self.cfg.joint_dim
            c.algo.act.point_feat_dim = self.cfg.point_feat_dim
            c.algo.act.em_latent_dim = self.cfg.em_latent_dim
            c.algo.act.kl_weight = self.cfg.kl_weight
            c.algo.act.img_token_downsample = self.cfg.img_token_downsample
            c.algo.act.dropout = self.cfg.dropout
            c.algo.act.cold_start_prob = 0.0
            c.algo.act.freeze_em = True
            c.algo.act.em_encoder_ckpt = self.cfg.encoder_ckpt
            c.algo.act.em_decoder_ckpt = self.cfg.decoder_ckpt
        c.lock()
        return c

    def load(self):
        ckpt_path = Path(self.cfg.ckpt)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"权重不存在: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        ckpt["_ckpt_dir"] = str(ckpt_path.parent)

        train_args = ckpt.get("args", {})
        for k in ("seq_len", "window_size", "joint_dim", "arm_pose_dim",
                  "num_points", "point_feat_dim", "em_latent_dim",
                  "robot_arm_supervision", "img_channel_order"):
            if k in train_args:
                setattr(self.cfg, k, train_args[k])

        dex_cfg = self._build_config()
        obs_shapes = {
            "arm_pose": (self.cfg.arm_pose_dim,),
            "front_RGB": (3, self.cfg.img_size[0], self.cfg.img_size[1]),
        }
        self.algo = algo_factory(
            algo_name="dexmimic",
            config=dex_cfg,
            obs_key_shapes=obs_shapes,
            ac_dim=self.cfg.arm_pose_dim,
            device=self.device,
        )
        # mimic_train 保存的是 unwrap 后的 policy.state_dict()（无 "policy." 前缀）
        model_sd = ckpt["model"]
        if any(k.startswith("policy.") for k in model_sd):
            self.algo.nets.load_state_dict(model_sd)
        else:
            self.algo.nets["policy"].load_state_dict(model_sd, strict=False)
        self.algo.nets.eval()
        self.algo._mlp_hand_loaded = "mlp_hand.0.weight" in model_sd
        resolved = resolve_hand_decode_mode(
            self.cfg.hand_decode, ckpt.get("args"), self.algo._mlp_hand_loaded,
        )
        self.algo.hand_decode_mode = resolved
        self.algo.nets["policy"].reset_cache()
        self.pose_stats = PoseStats.load(
            ckpt, self.cfg.pose_stats,
            robot_arm_supervision=getattr(self.cfg, "robot_arm_supervision", "joint"),
        )
        self.seq_len = self.algo.chunk_size
        mlp = "yes" if self.algo._mlp_hand_loaded else "no"
        print(f"[Model] 已加载 {ckpt_path.name} | chunk={self.seq_len} "
              f"W={self.algo.window_size} J={self.algo.joint_dim} | "
              f"mlp_hand={mlp} | hand_decode={resolved} | "
              f"arm_supervision={getattr(self.cfg, 'robot_arm_supervision', 'joint')}")

    @torch.no_grad()
    def predict(self, batch: dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        返回 denorm 后的 (pred_arm [T,6], pred_joints [T,J])。
        """
        pb = self.algo.process_batch_for_training(batch, self.algo.ac_key_robot)
        preds = self.algo.forward_eval(pb, unnorm_stats=None)

        pred_arm = preds["actions_arm"][0].cpu().numpy()
        pred_joints = preds["actions_hand"][0].cpu().numpy()

        if self.pose_stats is not None:
            pred_arm = pred_arm * self.pose_stats.std + self.pose_stats.mean
        pred_joints = apply_l25_reserved_joints(pred_joints)
        return pred_arm, pred_joints


# ══════════════════════════════════════════════════════════════════════════════
# 6. 观测构建 & 7. 动作下发（数据管线）
# ══════════════════════════════════════════════════════════════════════════════
class ObservationBuilder:
    """将相机帧 + 本体感知 → DexMimic batch。"""

    def __init__(self, cfg: "InferenceConfig", pose_stats: PoseStats):
        self.cfg = cfg
        self.pose_stats = pose_stats

    def normalize_pose(self, pose6: np.ndarray) -> np.ndarray:
        return (pose6 - self.pose_stats.mean) / self.pose_stats.std

    def _format_hand_cloud(self, hand_cloud: torch.Tensor) -> torch.Tensor:
        """统一为 [B, W, N, 3]；在线推理 W=window_size。"""
        if hand_cloud.ndim == 2:
            # [N, 3] → [1, 1, N, 3]
            return hand_cloud.unsqueeze(0).unsqueeze(0)
        if hand_cloud.ndim == 3:
            if hand_cloud.shape[-1] != 3:
                raise ValueError(f"hand_cloud 末维应为 3，实际 {hand_cloud.shape}")
            # [1, N, 3] → [1, 1, N, 3]；[W, N, 3] → [1, W, N, 3]
            return hand_cloud.unsqueeze(0)
        if hand_cloud.ndim == 4:
            return hand_cloud
        raise ValueError(f"hand_cloud 形状不支持: {tuple(hand_cloud.shape)}")

    def build(
        self,
        front_rgb: np.ndarray,
        hand_cloud: torch.Tensor,
        wrist_pose6: np.ndarray,
    ) -> dict:
        """构造 batch_size=1 的推理 batch。"""
        img = torch.from_numpy(front_rgb).float().permute(2, 0, 1) / 255.0
        if getattr(self.cfg, "img_channel_order", "bgr") == "rgb":
            img = img.flip(0)          # 相机帧为 BGR → 翻转为 RGB（与训练一致）
        h, w = self.cfg.img_size
        if img.shape[1] != h or img.shape[2] != w:
            img = F.interpolate(img.unsqueeze(0), size=(h, w),
                                mode="bilinear", align_corners=False)[0]

        pose_norm = self.normalize_pose(wrist_pose6.astype(np.float32))
        T = self.cfg.seq_len

        batch = {
            "obs": {
                "front_RGB": img.unsqueeze(0).unsqueeze(0),       # [1,1,3,H,W]
                "arm_pose": torch.from_numpy(pose_norm).float().reshape(1, 1, -1),
                "pad_mask": torch.ones(1, T, 1),
            },
            "type": torch.zeros(1, T, 1),
            "hand_point_cloud": self._format_hand_cloud(hand_cloud),
        }
        return batch


class ActionDispatcher:
    """将 chunk 中的单步动作写入共享总线。"""

    def __init__(self, bus: SharedDataBus):
        self.bus = bus

    def dispatch(self, arm_pose6: np.ndarray, hand_joints_rad: np.ndarray):
        self.bus.publish_arm_cmd(arm_pose6)
        self.bus.publish_hand_cmd(joints_rad_to_raw255(hand_joints_rad))


# ══════════════════════════════════════════════════════════════════════════════
# 8. 推理主循环（调度：感知 → 推理 → 下发）
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class InferenceConfig:
    ckpt: str = ""
    pose_stats: Optional[str] = None
    encoder_ckpt: str = TEST_CKPT_PATH_ENC
    decoder_ckpt: str = TEST_CKPT_PATH_DEC

    arm_ip: str = ARM_IP
    work_frame: str = ARM_WORK_FRAME
    tool_frame: str = ARM_TOOL_FRAME
    arm_euler_flag: int = ARM_EULER_FLAG
    sim: bool = False
    can_interface: str = DEFAULT_CAN_INTERFACE

    front_camera_id: str = FRONT_CAMERA_ID
    dro_grasp_root: str = DEFAULT_DRO_GRASP_ROOT
    hand_robot_name: str = "linker21_right"

    seq_len: int = 20
    window_size: int = 20
    joint_dim: int = 25
    arm_pose_dim: int = 6
    num_points: int = 256
    point_feat_dim: int = 256
    em_latent_dim: int = 128
    kl_weight: float = 20.0
    img_token_downsample: int = 1
    dropout: float = 0.1
    img_size: Tuple[int, int] = (240, 320)
    robot_arm_supervision: str = "joint"
    # 送入模型的通道顺序；从 ckpt 训练参数读取（旧 ckpt=bgr，新 ckpt=rgb）
    img_channel_order: str = "bgr"

    control_hz: float = 30.0
    replan_steps: int = 5          # 每个 chunk 执行步数，然后重新推理
    dry_run: bool = False
    no_camera: bool = False

    # 手部关节解码：默认 em_decoder（与 direct_joint_lambda=0 训练一致）
    hand_decode: str = "em_decoder"


class InferenceLoop:
    def __init__(
        self,
        cfg: InferenceConfig,
        bus: SharedDataBus,
        model: ModelEngine,
        camera: Optional[CameraSystem],
        hand_pc_gen: Optional[HandPointCloudGenerator],
    ):
        self.cfg = cfg
        self.bus = bus
        self.model = model
        self.camera = camera
        self.hand_pc_gen = hand_pc_gen
        self.obs_builder = ObservationBuilder(cfg, model.pose_stats)
        self.dispatcher = ActionDispatcher(bus)

        self._chunk_arm: Optional[np.ndarray] = None
        self._chunk_hand: Optional[np.ndarray] = None
        self._chunk_idx = 0
        self._default_hand_qpos = np.full(25, 128, dtype=np.float64)
        # 原始点云历史（未归一化）；与 mimic_train 一致，在 W 帧窗口上做一次全局 normalize
        self._pc_raw_history: list[torch.Tensor] = []

    def _current_hand_qpos(self) -> np.ndarray:
        state = self.bus.read_hand_state()
        if state is not None:
            return state
        return self._default_hand_qpos.copy()

    def _append_raw_hand_cloud(self, wrist_pose6: np.ndarray) -> None:
        """每控制步追加一帧原始点云（与训练数据生成一致，不做单帧归一化）。"""
        if self.hand_pc_gen is None:
            cloud = torch.zeros(self.cfg.num_points, 3)
        else:
            cloud = self.hand_pc_gen.generate(
                self._current_hand_qpos(),
                wrist_pose_base=wrist_pose6,
                normalize=False,
            )
        self._pc_raw_history.append(cloud)
        # 仅保留最近 2W 帧，避免无限增长
        max_keep = self.cfg.window_size * 2
        if len(self._pc_raw_history) > max_keep:
            self._pc_raw_history = self._pc_raw_history[-max_keep:]

    def _build_hand_cloud_window(self) -> torch.Tensor:
        """
        构造 [W, N, 3] 手部点云窗口（末帧为当前帧）。
        训练时 mimic_train 对 buf_cloud[W+T] 做全局 normalize 后取前 W 帧；
        推理复现同一口径。
        """
        W, N = self.cfg.window_size, self.cfg.num_points
        if not self._pc_raw_history:
            return torch.zeros(W, N, 3)

        hist = self._pc_raw_history
        if len(hist) < W:
            padded = [hist[0]] * (W - len(hist)) + hist
        else:
            padded = hist[-W:]
        window = torch.stack(padded, dim=0)
        return normalize_cloud(window)

    def _need_replan(self) -> bool:
        if self._chunk_arm is None:
            return True
        return self._chunk_idx >= min(self.cfg.replan_steps, len(self._chunk_arm))

    def _replan(self, frame: CameraFrame, wrist_pose6: np.ndarray):
        hand_cloud = self._build_hand_cloud_window()
        batch = self.obs_builder.build(frame.front_rgb, hand_cloud, wrist_pose6)
        pred_arm, pred_hand = self.model.predict(batch)
        self._chunk_arm = pred_arm
        self._chunk_hand = pred_hand
        self._chunk_idx = 0
        print(f"[Infer] 新 chunk | arm[0]={pred_arm[0]} hand[0][:3]={pred_hand[0][:3]}")

    def _fallback_pose(self) -> np.ndarray:
        p = self.bus.read_wrist_pose()
        if p is not None:
            return p
        return np.zeros(self.cfg.arm_pose_dim, dtype=np.float64)

    def run(self):
        dt = 1.0 / self.cfg.control_hz
        print(f"[Loop] 控制频率 {self.cfg.control_hz} Hz | replan_steps={self.cfg.replan_steps}")
        self._pc_raw_history.clear()
        if self.model.algo is not None:
            self.model.algo.reset()

        if not self.cfg.dry_run:
            for _ in range(50):
                if self.bus.read_wrist_pose() is not None:
                    break
                time.sleep(0.1)
            else:
                print("[Loop] 警告：未收到 wrist_pose_base，将使用零位姿作为 proprio")

        try:
            while not self.bus.should_exit():
                t0 = time.time()

                if self.camera is not None:
                    frame = self.camera.grab()
                    wrist_pose6 = self.bus.read_wrist_pose()
                    if wrist_pose6 is None:
                        wrist_pose6 = self._fallback_pose()

                    self._append_raw_hand_cloud(wrist_pose6)
                    if self._need_replan():
                        self._replan(frame, wrist_pose6)

                    arm_step = self._chunk_arm[self._chunk_idx]
                    hand_step = self._chunk_hand[self._chunk_idx]
                    self.dispatcher.dispatch(arm_step, hand_step)
                    self._chunk_idx += 1

                    if not self.cfg.dry_run:
                        cv2.imshow("front", frame.front_rgb)
                        if cv2.waitKey(1) == ord("q"):
                            self.bus.request_exit()
                else:
                    # 无相机 dry-run：用 DRO 合成点云或零占位
                    wrist_pose6 = self._fallback_pose()
                    self._append_raw_hand_cloud(wrist_pose6)
                    hand_cloud = self._build_hand_cloud_window()
                    batch = self.obs_builder.build(
                        np.zeros((480, 640, 3), np.uint8),
                        hand_cloud,
                        wrist_pose6,
                    )
                    pred_arm, pred_hand = self.model.predict(batch)
                    print(f"[DryRun] 推理成功 | pred_arm {pred_arm.shape} "
                          f"pred_hand {pred_hand.shape}")
                    self.bus.request_exit()

                elapsed = time.time() - t0
                time.sleep(max(0.0, dt - elapsed))
        finally:
            cv2.destroyAllWindows()
            print("[Loop] 推理循环结束")


# ══════════════════════════════════════════════════════════════════════════════
# 9. 入口：进程编排
# ══════════════════════════════════════════════════════════════════════════════
def parse_args() -> InferenceConfig:
    p = argparse.ArgumentParser(description="DexMimic 真机推理")
    p.add_argument("--ckpt", type=str, default=DEXMIMIC_PATH, help="dexmimic_best.pt 路径")
    p.add_argument("--pose-stats", type=str, default=None,
                   help="pose_stats.json（若 ckpt 内未包含）")
    p.add_argument("--encoder-ckpt", type=str, default=TEST_CKPT_PATH_ENC)
    p.add_argument("--decoder-ckpt", type=str, default=TEST_CKPT_PATH_DEC)
    p.add_argument("--dry-run", action="store_true", help="不启动力/臂子进程")
    p.add_argument("--no-camera", action="store_true", help="跳过相机（仅测模型加载）")
    p.add_argument("--sim", action="store_true", help="机械臂仿真模式")
    p.add_argument("--replan-steps", type=int, default=5)
    p.add_argument("--control-hz", type=float, default=30.0)
    p.add_argument("--arm-ip", type=str, default=ARM_IP)
    p.add_argument("--work-frame", type=str, default=ARM_WORK_FRAME,
                   help="工作坐标系（训练 wrist_pose_base 用 RM65；VisionPro 遥操 sample 用 visionpro）")
    p.add_argument("--tool-frame", type=str, default=ARM_TOOL_FRAME,
                   help="工具坐标系（训练用 Arm_Tip；VisionPro 遥操 sample 用 vr_right）")
    p.add_argument("--arm-euler-flag", type=int, default=ARM_EULER_FLAG, choices=[0, 1],
                   help="逆解姿态格式：0=四元数(7D)，1=欧拉角(6D，与 wrist_pose_base 一致)")
    p.add_argument("--can", type=str, default=DEFAULT_CAN_INTERFACE)
    p.add_argument("--dro-grasp-root", type=str, default=DEFAULT_DRO_GRASP_ROOT,
                   help="DRO-Grasp-master 仓库路径（手部 URDF 点云生成）")
    p.add_argument("--hand-robot-name", type=str, default="linker21_right",
                   help="DRO-Grasp 手模型名称")
    p.add_argument(
        "--hand-decode",
        type=str,
        default="em_decoder",
        choices=("auto", "mlp_hand", "em_decoder"),
        help="手部关节解码：默认 em_decoder；auto 时若训练 direct_joint_lambda=0 则用 em_decoder",
    )
    args = p.parse_args()
    return InferenceConfig(
        ckpt=args.ckpt,
        pose_stats=args.pose_stats,
        encoder_ckpt=args.encoder_ckpt,
        decoder_ckpt=args.decoder_ckpt,
        dry_run=args.dry_run,
        no_camera=args.no_camera,
        sim=args.sim,
        replan_steps=args.replan_steps,
        control_hz=args.control_hz,
        arm_ip=args.arm_ip,
        work_frame=args.work_frame,
        tool_frame=args.tool_frame,
        arm_euler_flag=args.arm_euler_flag,
        can_interface=args.can,
        dro_grasp_root=args.dro_grasp_root,
        hand_robot_name=args.hand_robot_name,
        hand_decode=args.hand_decode,
    )


def main():
    multiprocessing.set_start_method("spawn", force=True)
    cfg = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manager = multiprocessing.Manager()
    bus = SharedDataBus.create(manager)

    # ---- 模型初始化（主进程）----
    model = ModelEngine(cfg, device)
    model.load()

    # ---- 手部点云生成器（DRO-Grasp URDF）----
    hand_pc_gen = HandPointCloudGenerator(
        dro_root=cfg.dro_grasp_root,
        robot_name=cfg.hand_robot_name,
        num_points=cfg.num_points,
    )

    # ---- 相机初始化（主进程，仅 RGB）----
    camera = None
    if not cfg.no_camera:
        try:
            camera = CameraSystem(cfg.front_camera_id)
            camera.start()
        except Exception as exc:
            print(f"[Camera] 初始化失败: {exc}")
            if not cfg.dry_run:
                raise

    procs = []
    if not cfg.dry_run:
        arm_p = multiprocessing.Process(
            target=ArmController.run_loop, args=(bus, cfg), daemon=True
        )
        hand_p = multiprocessing.Process(
            target=HandController.run_loop, args=(bus, cfg), daemon=True
        )
        arm_p.start()
        hand_p.start()
        procs.extend([arm_p, hand_p])
        time.sleep(2.0)  # 等待臂/手初始化

    loop = InferenceLoop(cfg, bus, model, camera, hand_pc_gen)
    try:
        loop.run()
    except KeyboardInterrupt:
        print("\n[Main] 收到中断")
    finally:
        bus.request_exit()
        if camera is not None:
            camera.stop()
        for proc in procs:
            proc.join(timeout=2.0)
            if proc.is_alive():
                proc.terminate()
        print("[Main] 已安全退出")


if __name__ == "__main__":
    main()
