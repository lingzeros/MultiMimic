# eval.py
"""RM65 + L10/Inspire 灵巧手真机部署脚本（ACT）。"""
import os
import sys
import time
import pickle

import cv2
import numpy as np
import torch
from einops import rearrange

from depth_utils import is_depth_name, normalize_depth

# LinkerHand SDK（必须把 LinkerHand/ 放在 sys.path 最前，
# 否则会命中本仓库根目录的 utils.py，导致 SDK 内 `from utils.mapping` 失败）
_ROOT = os.path.dirname(os.path.abspath(__file__))
_LINKER_HAND_DIR = os.path.join(_ROOT, 'linker_hand_python_sdk', 'LinkerHand')
_LINKER_SDK_DIR = os.path.join(_ROOT, 'linker_hand_python_sdk')

CKPT_DIR = "/home/ub/MultiMimic/checkpoints/Peach_dual_decoder_inspire1"
CKPT_TYPE = 'policy_best.ckpt'
# CKPT_TYPE = 'policy_epoch_4800_seed_0.ckpt'
# camera_names 是 depth encoder 的唯一开关；
# 带 depth 的 checkpoint必须选择 front_depth/dual_depth，
# RGB-only checkpoint 选择 front/dual
CAMERA_MODE = 'front'
# CAMERA_MODE = 'dual'
# CAMERA_MODE = 'front_depth'
# CAMERA_MODE = 'dual_depth'

# 灵巧手类型：'l10'（LinkerHand 10维）或 'inspire'（因时6维）
ROBOT_HAND = 'inspire'

def _import_linker_hand_api():
    """导入 LinkerHandApi，规避与 act/utils.py 的命名冲突。"""
    # 若已误加载为单文件模块，清掉后再导入 SDK 的 utils 包
    mod = sys.modules.get('utils')
    if mod is not None and not hasattr(mod, '__path__'):
        del sys.modules['utils']
        for name in list(sys.modules):
            if name.startswith('utils.'):
                del sys.modules[name]

    if _LINKER_HAND_DIR not in sys.path:
        sys.path.insert(0, _LINKER_HAND_DIR)
    if _LINKER_SDK_DIR not in sys.path:
        sys.path.insert(0, _LINKER_SDK_DIR)

    from linker_hand_python_sdk.LinkerHand.linker_hand_api import LinkerHandApi
    return LinkerHandApi


from Robotic_Arm.rm_robot_interface import *
import pyrealsense2 as rs

from policy import ACTPolicy, CNNMLPPolicy

# ─── 硬件默认配置 ───────────────────────────────────────────────
ARM_IP = '192.168.1.18'
ARM_PORT = 8080
HAND_JOINT = 'L10'          # LinkerHand 型号，仅 L10 模式使用
HAND_TYPE = 'right'         # left / right，仅 L10 模式使用
CAN_INTERFACE = 'can0'
HAND_SPEED = [180, 250, 250, 250, 250]
HAND_TORQUE = [255, 255, 255, 255, 255]

HAND_CONFIGS = {
    'l10': {
        'dim': 10,
        'min': 0.0,
        'max': 255.0,
        'max_step': 45.0,
        # LinkerHand SDK open_palm 示例
        'init_pose': [255.0, 70.0, 255.0, 255.0, 255.0,
                      255.0, 255.0, 255.0, 255.0, 255.0],
    },
    'inspire': {
        'dim': 6,
        'min': 0.0,
        'max': 1000.0,
        'max_step': 150.0,
        # 与 Inspire 训练数据每条 episode 的初始 hand_joints 一致
        'init_pose': [1000.0, 1000.0, 1000.0, 1000.0, 1000.0, 0.0],
    },
}
if ROBOT_HAND not in HAND_CONFIGS:
    raise ValueError(f'ROBOT_HAND 必须是 {list(HAND_CONFIGS)}, 实际为 {ROBOT_HAND!r}')
HAND_CONFIG = HAND_CONFIGS[ROBOT_HAND]
HAND_DIM = HAND_CONFIG['dim']
STATE_DIM = 6 + HAND_DIM
HAND_INIT_POSE = list(HAND_CONFIG['init_pose'])
# ARM_HOME = [-8, 12, 98, 59, 19, -20]
ARM_HOME = [-2.4, -9.4, 100.3, 1.3, 43.5, -1.9]

# 部署时通过 --camera-mode 选择；默认单前视相机以匹配 Peach 训练配置。
CAMERA_MODES = {
    'front': ['front_RGB'],
    'dual': ['front_RGB', 'wrist_RGB'],
    'front_depth': ['front_RGB', 'front_depth'],
    'dual_depth': ['front_RGB', 'wrist_RGB', 'front_depth'],
}
DEFAULT_CAMERA_NAMES = CAMERA_MODES['front']
# 采集时 HDF5 attrs 中的序列号（与 Orange/Peach 自采一致）
FRONT_CAMERA_SERIAL = '216322073710'
WRIST_CAMERA_SERIAL = '260322279022'
CAMERA_SERIALS = {
    'front_RGB': FRONT_CAMERA_SERIAL,
    'wrist_RGB': WRIST_CAMERA_SERIAL,
}

# 机械臂/灵巧手单次指令限幅
ARM_MAX_STEP = [5, 5, 5, 5, 5, 5]
HAND_MAX_STEP = HAND_CONFIG['max_step']


class L10HandController:
    """LinkerHand L10：CAN 初始化 + finger_move（与 mimic_inference.HandController 同驱动）。"""

    def __init__(
        self,
        hand_joint=HAND_JOINT,
        hand_type=HAND_TYPE,
        can=CAN_INTERFACE,
        speed=None,
        torque=None,
    ):
        self.hand_joint = hand_joint
        self.hand_type = hand_type
        self.can = can
        self.speed = speed or list(HAND_SPEED)
        self.torque = torque or list(HAND_TORQUE)
        self._hand = None
        self._last_qpos = list(HAND_INIT_POSE)

    def connect(self, init_pose=None):
        # 必须走 _import_linker_hand_api：先把 LinkerHand/ 放到 path 最前并清掉
        # 被 act/utils.py 占用的 utils，否则 SDK 内 `from utils.mapping` 会失败
        LinkerHandApi = _import_linker_hand_api()

        self._hand = LinkerHandApi(
            hand_joint=self.hand_joint,
            hand_type=self.hand_type,
            can=self.can,
        )
        self._hand.set_speed(speed=self.speed)
        self._hand.set_torque(torque=self.torque)
        pose = list(init_pose) if init_pose is not None else list(HAND_INIT_POSE)
        self.move(pose)
        time.sleep(0.5)
        print(f'✓ LinkerHand {self.hand_type} {self.hand_joint} 初始化完成 (can={self.can})')

    def move(self, qpos, force=False):
        """下发 10 维关节指令，数值范围 0~255。"""
        pose = [float(np.clip(x, 0, 255)) for x in qpos]
        if len(pose) != HAND_DIM:
            raise ValueError(f'L10 pose 需要 {HAND_DIM} 维，实际 {len(pose)}')
        self._hand.finger_move(pose=pose)
        self._last_qpos = pose

    def get_state(self):
        """读取手部状态；失败则回退到上次指令。"""
        try:
            state = self._hand.get_state()
            if state is not None and len(state) >= HAND_DIM:
                return [float(x) for x in state[:HAND_DIM]]
        except Exception:
            pass
        return list(self._last_qpos)


class InspireHandController:
    """通过 RM65 末端 Modbus 控制6维 Inspire 灵巧手。

    采集程序没有读取手指位置反馈，因此这里同样使用最近一次成功下发的
    指令作为 hand state，以保持训练和部署的观测语义一致。
    """

    def __init__(self, arm, command_interval=3):
        self._arm = arm
        self._last_qpos = list(HAND_INIT_POSE)
        self._command_interval = command_interval
        self._move_count = 0
        self._modbus_open = False

    def connect(self, init_pose=None):
        # 上一次进程若异常退出，RM65 控制器端的末端 RS485 可能仍处于
        # Modbus RTU 模式。先尝试关闭旧模式，再重新初始化。
        try:
            self._arm.rm_close_modbus_mode(1)
            time.sleep(0.5)
        except Exception as exc:
            print(f'⚠ 清理旧Inspire Modbus状态失败，将继续尝试初始化: {exc}')

        result = self._arm.rm_set_modbus_mode(1, 115200, 2)
        if result not in (0, None):
            raise RuntimeError(f'rm_set_modbus_mode失败: {result}')
        self._modbus_open = True
        time.sleep(0.5)
        try:
            result = self._arm.rm_set_voltage(3, True)
        except TypeError:
            # 兼容不带 block 参数的 RM API 版本。
            result = self._arm.rm_set_voltage(3)
        if result not in (0, None):
            raise RuntimeError(f'rm_set_voltage失败: {result}')
        time.sleep(0.5)

        speed_force = [2, 232] * HAND_DIM
        params = rm_peripheral_read_write_params_t(1, 1522, 1, HAND_DIM)
        result = self._arm.rm_write_registers(params, speed_force)
        if result not in (0, None):
            raise RuntimeError(f'Inspire速度/力阈值设置失败: {result}')

        pose = list(init_pose) if init_pose is not None else list(HAND_INIT_POSE)
        self.move(pose, force=True)
        time.sleep(0.5)
        print('✓ Inspire 灵巧手初始化完成 (RM65 Modbus, 6 DoF)')

    def move(self, qpos, force=False):
        if len(qpos) != HAND_DIM:
            raise ValueError(f'Inspire pose需要{HAND_DIM}维，实际{len(qpos)}')
        pose = [
            int(round(np.clip(x, HAND_CONFIG['min'], HAND_CONFIG['max'])))
            for x in qpos
        ]
        self._move_count += 1
        if not force and self._move_count % self._command_interval != 0:
            return
        result = self._arm.rm_set_hand_angle(pose, False, 0)
        if result not in (0, None):
            raise RuntimeError(f'rm_set_hand_angle失败: {result}')
        self._last_qpos = [float(x) for x in pose]

    def get_state(self):
        return list(self._last_qpos)

    def close(self):
        """释放RM65末端RS485的Modbus模式，供下次运行重新初始化。"""
        if not self._modbus_open:
            return
        result = self._arm.rm_close_modbus_mode(1)
        if result not in (0, None):
            print(f'⚠ 关闭Inspire Modbus返回: {result}')
        else:
            print('✓ Inspire Modbus已关闭')
        self._modbus_open = False


def safe_action_filter(
    action,
    current_qpos,
    arm_max_step=None,
    hand_max_step=HAND_MAX_STEP,
):
    """安全动作过滤：臂/手每关节限步长。"""
    if arm_max_step is None:
        arm_max_step = ARM_MAX_STEP
    action = np.array(action, dtype=np.float64)
    current_qpos = np.array(current_qpos, dtype=np.float64)

    for i in range(6):
        diff = action[i] - current_qpos[i]
        if abs(diff) > arm_max_step[i]:
            action[i] = current_qpos[i] + np.sign(diff) * arm_max_step[i]

    hand = np.clip(
        action[6:6 + HAND_DIM], HAND_CONFIG['min'], HAND_CONFIG['max']
    )
    hand_obs = current_qpos[6:6 + HAND_DIM]

    for i in range(HAND_DIM):
        diff = hand[i] - hand_obs[i]
        if abs(diff) > hand_max_step:
            hand[i] = hand_obs[i] + np.sign(diff) * hand_max_step
    hand = np.clip(hand, HAND_CONFIG['min'], HAND_CONFIG['max'])
    action[6:6 + HAND_DIM] = hand
    return action


def canfd_judge(target_angle, current_q):
    return np.max(np.abs(np.array(target_angle) - np.array(current_q))) < 10


def pose_follow(q_out, c_q):
    tar_q = np.array(q_out, dtype=np.float64)
    joint_diff = np.array(q_out, dtype=np.float64) - np.array(c_q, dtype=np.float64)
    index = np.abs(joint_diff) >= 10
    tar_q[index] = (np.array(c_q) + 9.5 * (joint_diff / np.abs(joint_diff)))[index]
    return tar_q


def init_robot():
    """初始化 RM65 机械臂（不含手）。"""
    print('=== 初始化 RM65 机械臂 ===')
    try:
        left_arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
        handle = left_arm.rm_create_robot_arm(ARM_IP, ARM_PORT)
        print(f'✓ 机械臂连接成功，handle id: {handle.id}')

        left_arm.rm_change_work_frame('RM65')
        left_arm.rm_change_tool_frame('Arm_Tip')
        left_arm.rm_set_arm_run_mode(1)
        return left_arm
    except Exception as e:
        print(f'✗ 机械臂初始化失败: {e}')
        return None


def init_hand(left_arm):
    """根据 ROBOT_HAND 初始化 L10 或 Inspire。"""
    print(f'=== 初始化灵巧手: {ROBOT_HAND} ===')
    hand = None
    try:
        if ROBOT_HAND == 'l10':
            hand = L10HandController()
        else:
            hand = InspireHandController(left_arm)
        hand.connect()
        return hand
    except Exception as e:
        print(f'✗ {ROBOT_HAND} 初始化失败: {e}')
        if hand is not None and hasattr(hand, 'close'):
            try:
                hand.close()
            except Exception as close_exc:
                print(f'⚠ 初始化失败后的Modbus清理也失败: {close_exc}')
        return None


def _camera_source(name):
    """front_RGB/front_depth -> front; wrist_RGB/wrist_depth -> wrist."""
    for suffix in ('_RGB', '_depth'):
        if name.endswith(suffix):
            return name[:-len(suffix)]
    raise ValueError(f'无法识别相机观测名称: {name}')


def _start_realsense(serial, enable_depth=False):
    """启动 RealSense；需要时将原始 Z16 深度对齐到彩色图。"""
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
    if enable_depth:
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
    pipeline = rs.pipeline()
    pipeline.start(cfg)
    align = rs.align(rs.stream.color) if enable_depth else None
    return pipeline, align


def init_cameras(camera_names=None, serials=None):
    """
    初始化多路 RealSense。RGB/depth 按物理设备去重。
    返回 {front/wrist: (pipeline, align)}；失败返回 None。
    """
    if camera_names is None:
        camera_names = list(DEFAULT_CAMERA_NAMES)
    if serials is None:
        serials = CAMERA_SERIALS

    print('=== 初始化 RealSense 相机 ===')
    cams = {}
    try:
        sources = {}
        for name in camera_names:
            sources.setdefault(_camera_source(name), []).append(name)

        for source, observation_names in sources.items():
            serial = serials.get(f'{source}_RGB') or serials.get(source)
            if not serial:
                raise KeyError(f'未配置相机序列号: {source}')
            enable_depth = any(is_depth_name(name) for name in observation_names)
            print(f'正在启动 {source} (serial={serial}, depth={enable_depth})...')
            try:
                pipeline, align = _start_realsense(serial, enable_depth)
            except Exception as exc:
                raise RuntimeError(
                    f'{source} 启动失败 (serial={serial}): {exc}'
                ) from exc
            cams[source] = (pipeline, align)
            print(f'✓ {source} 初始化成功 (serial={serial})')
        return cams
    except Exception as e:
        print(f'✗ 相机初始化失败: {e}')
        for pipeline, _ in cams.values():
            try:
                pipeline.stop()
            except Exception:
                pass
        return None


def stop_cameras(cams):
    if not cams:
        return
    for name, (pipeline, _) in cams.items():
        try:
            pipeline.stop()
        except Exception:
            pass


def cleanup_resources(left_arm=None, hand=None, cams=None):
    """按相机→灵巧手Modbus→机械臂的顺序释放所有部署资源。"""
    stop_cameras(cams)
    if hand is not None and hasattr(hand, 'close'):
        try:
            hand.close()
        except Exception as exc:
            print(f'⚠ 关闭灵巧手通信失败: {exc}')
    if left_arm is not None:
        try:
            left_arm.rm_delete_robot_arm()
        except Exception as exc:
            print(f'⚠ 断开机械臂失败: {exc}')
        try:
            RoboticArm.rm_destroy()
        except Exception as exc:
            print(f'⚠ 销毁机械臂SDK失败: {exc}')
    cv2.destroyAllWindows()


def _infer_act_shape_from_ckpt(state_dict):
    """从 ACT checkpoint 权重形状推断 chunk_size(=num_queries) 与 state_dim。"""
    qkey = 'model.query_embed.weight'
    arm_key = 'model.arm_action_head.weight'
    hand_key = 'model.hand_action_head.weight'
    if qkey not in state_dict:
        raise KeyError(f'checkpoint 缺少 {qkey}，无法推断 chunk_size')
    num_queries = int(state_dict[qkey].shape[0])
    if arm_key not in state_dict or hand_key not in state_dict:
        raise KeyError('checkpoint 不是 arm/hand 双 decoder ACT 模型')
    state_dim = int(state_dict[arm_key].shape[0] + state_dict[hand_key].shape[0])
    return num_queries, state_dim


def _infer_backbone_from_ckpt(state_dict):
    """Infer which supported visual backbone produced an ACT checkpoint."""
    keys = state_dict.keys()
    if any(".body.patch_embed." in key for key in keys):
        return "dinov2_vits14"
    if any(".body.conv1." in key for key in keys):
        return "resnet18"
    raise KeyError("无法从 checkpoint 参数名判断 backbone 类型")


def _infer_depth_from_ckpt(state_dict):
    """Depth checkpoints contain a separately trained depth encoder."""
    return any(key.startswith('model.depth_encoder.') for key in state_dict)


def load_policy(ckpt_dir, policy_class, camera_names=None):
    """加载 ACT / CNNMLP 策略与 dataset_stats。

    ACT 的 chunk_size(=num_queries) 从 ckpt 的 query_embed 形状自动读取，
    无需与训练时手动保持一致。
    """
    if camera_names is None:
        camera_names = list(DEFAULT_CAMERA_NAMES)
    try:
        ckpt_path = os.path.join(ckpt_dir, CKPT_TYPE)
        if not os.path.exists(ckpt_path):
            print(f'✗ 找不到模型文件: {ckpt_path}')
            return None, None, None

        state_dict = torch.load(
            ckpt_path,
            map_location='cpu',
            weights_only=True,
        )
        backbone = _infer_backbone_from_ckpt(state_dict)
        checkpoint_uses_depth = _infer_depth_from_ckpt(state_dict)
        requested_depth = any(is_depth_name(name) for name in camera_names)
        if checkpoint_uses_depth != requested_depth:
            raise ValueError(
                'checkpoint 与 camera_names 的 depth 配置不一致: '
                f'checkpoint_uses_depth={checkpoint_uses_depth}, '
                f'camera_names={camera_names}'
            )

        if policy_class == 'ACT':
            num_queries, ckpt_state_dim = _infer_act_shape_from_ckpt(state_dict)
            if ckpt_state_dim != STATE_DIM:
                print(
                    f'⚠ ckpt state_dim={ckpt_state_dim} 与部署 STATE_DIM={STATE_DIM} 不一致，'
                    f'以 ckpt 为准构建网络'
                )
            policy_config = {
                'lr': 1e-5,
                'num_queries': num_queries,
                'kl_weight': 10,
                'hidden_dim': 512,
                'dim_feedforward': 3200,
                'lr_backbone': 1e-5,
                'backbone': backbone,
                'enc_layers': 4,
                'dec_layers': 7,
                'nheads': 8,
                'camera_names': list(camera_names),
                'state_dim': ckpt_state_dim,
            }
            policy = ACTPolicy(policy_config)
        elif policy_class == 'CNNMLP':
            policy_config = {
                'lr': 1e-4,
                'lr_backbone': 1e-5,
                'backbone': backbone,
                'num_queries': 1,
                'camera_names': list(camera_names),
                'state_dim': STATE_DIM,
            }
            policy = CNNMLPPolicy(policy_config)
        else:
            raise NotImplementedError

        policy.load_state_dict(state_dict)
        policy.cuda()
        policy.eval()
        print(f'✓ 策略加载成功: {policy_class}  '
              f'(chunk_size/num_queries={policy_config["num_queries"]}, '
              f'state_dim={policy_config["state_dim"]}, backbone={backbone}, '
              f'cameras={camera_names})')

        stats_path = os.path.join(ckpt_dir, 'dataset_stats.pkl')
        if os.path.exists(stats_path):
            with open(stats_path, 'rb') as f:
                stats = pickle.load(f)
        else:
            stats = {
                'qpos_mean': np.zeros(STATE_DIM),
                'qpos_std': np.ones(STATE_DIM),
                'action_mean': np.zeros(STATE_DIM),
                'action_std': np.ones(STATE_DIM),
            }
            print('⚠ 未找到 dataset_stats.pkl，使用单位归一化')

        for key in ('qpos_mean', 'qpos_std', 'action_mean', 'action_std'):
            if np.asarray(stats[key]).shape != (policy_config['state_dim'],):
                raise ValueError(
                    f'{key} shape={np.asarray(stats[key]).shape} 与 checkpoint '
                    f'state_dim={policy_config["state_dim"]} 不一致'
                )

        return policy, policy_config, stats
    except Exception as e:
        print(f'✗ 策略加载失败: {e}')
        return None, None, None


def _read_realsense(pipeline, align, need_depth=False):
    """读取一次 frameset，返回 RGB、BGR 和对齐后的 uint16 depth。"""
    blank_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    blank_depth = np.zeros((480, 640), dtype=np.uint16)
    try:
        frames = pipeline.wait_for_frames(1000)
        if align is not None:
            frames = align.process(frames)
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame() if need_depth else None
        if not color_frame:
            print(f"\nRealsense no color frame got")
            return blank_rgb, blank_rgb, blank_depth
        if need_depth and not depth_frame:
            print(f"\nRealsense no depth frame got")
            return blank_rgb, blank_rgb, blank_depth
        bgr = np.asanyarray(color_frame.get_data()).copy()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        depth = blank_depth
        
        if depth_frame:
            depth = np.asanyarray(depth_frame.get_data()).astype(
                np.uint16, copy=True
            )
        return rgb, bgr, depth

    except Exception as exc:
        print(f'\nRealsense RuntimeError: {exc}')
        return blank_rgb, blank_rgb, blank_depth
    except Exception as exc:
        print(f'\nRealSense未知异常: {type(exc).__name__}: {exc}')
        return blank_rgb, blank_rgb, blank_depth


def get_observation(left_arm, cams, hand, camera_names=None):
    """观测：臂/手状态以及 camera_names 指定的 RGB/depth。"""
    if camera_names is None:
        camera_names = list(DEFAULT_CAMERA_NAMES)
    obs = {'qpos': None, 'images': {}}

    try:
        joint_positions = np.array(left_arm.rm_get_joint_degree()[1], dtype=np.float64)
        hand_state = np.array(hand.get_state(), dtype=np.float64)
        obs['qpos'] = np.concatenate([joint_positions, hand_state])
    except Exception:
        obs['qpos'] = np.zeros(STATE_DIM)

    observations_by_source = {}
    for name in camera_names:
        observations_by_source.setdefault(_camera_source(name), []).append(name)

    for source, observation_names in observations_by_source.items():
        pipeline, align = cams[source]
        need_depth = any(is_depth_name(name) for name in observation_names)
        rgb, bgr, depth = _read_realsense(pipeline, align, need_depth)
        for name in observation_names:
            obs['images'][name] = depth if is_depth_name(name) else rgb
        if source == 'front' or '_bgr_display' not in obs['images']:
            obs['images']['_bgr_display'] = bgr

    return obs


def images_to_tensor(obs_images, camera_names=None):
    """构造模型输入；depth 保持 [B,1,H,W]，不复制为伪 RGB。"""
    if camera_names is None:
        camera_names = list(DEFAULT_CAMERA_NAMES)
    if any(is_depth_name(name) for name in camera_names):
        observations = {}
        for name in camera_names:
            if is_depth_name(name):
                value = normalize_depth(obs_images[name])[None, ...]
            else:
                value = rearrange(obs_images[name], 'h w c -> c h w') / 255.0
            observations[name] = (
                torch.from_numpy(np.asarray(value)).float().cuda().unsqueeze(0)
            )
        return observations

    cams = [
        rearrange(obs_images[name], 'h w c -> c h w')
        for name in camera_names
    ]
    stacked = np.stack(cams, axis=0)
    return torch.from_numpy(stacked / 255.0).float().cuda().unsqueeze(0)


def execute_action(left_arm, hand, action):
    """执行机械臂及当前配置的灵巧手动作。"""
    try:
        joint_action = action[:6]
        hand_action = action[6:6 + HAND_DIM]

        current_q = left_arm.rm_get_joint_degree()[1]
        if canfd_judge(joint_action, current_q):
            left_arm.rm_movej_canfd(joint_action.tolist(), False)
        else:
            tar_q = pose_follow(joint_action, current_q)
            left_arm.rm_movej_canfd(tar_q.tolist(), False)

        hand.move(hand_action.tolist())
        return True
    except Exception as e:
        print(f'动作执行失败: {e}')
        return False


def main():
    print(f'=== RM65 + {ROBOT_HAND} ACT 部署 ===')
    ckpt_dir = CKPT_DIR
    # if not os.path.exists(ckpt_dir):
    #     print(f'错误: 目录不存在 {ckpt_dir}')
    #     return
    print(f'ckpt_dir:{ckpt_dir}')
    policy_class = 'ACT'
    # policy_class = input('请输入策略类型 (ACT/CNNMLP): ').strip()
    # if policy_class not in ['ACT', 'CNNMLP']:
    #     print('错误: 策略类型必须是 ACT 或 CNNMLP')
    #     return

    # max_steps = int(input('请输入最大执行步数 (默认1000): ').strip() or '1000')
    max_steps = 210
    # 与 imitate_episodes.eval_bc 一致：对重叠 chunk 做指数加权平滑
    temporal_agg = False

    camera_mode = CAMERA_MODE

    left_arm = init_robot()
    if left_arm is None:
        return

    hand = init_hand(left_arm)
    if hand is None:
        cleanup_resources(left_arm=left_arm)
        return

    camera_names = list(CAMERA_MODES[camera_mode])
    print(f'camera_mode={camera_mode}, camera_names={camera_names}')
    cams = init_cameras(camera_names)
    if cams is None:
        cleanup_resources(left_arm=left_arm, hand=hand)
        return

    policy, policy_config, stats = load_policy(ckpt_dir, policy_class, camera_names)
    if policy is None:
        cleanup_resources(left_arm=left_arm, hand=hand, cams=cams)
        return
    ckpt_state_dim = policy_config['state_dim']
    if ckpt_state_dim != STATE_DIM:
        print(
            f'✗ checkpoint state_dim={ckpt_state_dim} 与 ROBOT_HAND={ROBOT_HAND} '
            f'所需 state_dim={STATE_DIM} 不一致'
        )
        cleanup_resources(left_arm=left_arm, hand=hand, cams=cams)
        return

    # 与 imitate_episodes 一致：temporal_agg 时每步重查询；否则间隔 = chunk_size
    query_frequency = policy_config['num_queries'] if policy_class == 'ACT' else 1
    num_queries = policy_config['num_queries']
    execution_horizon = 20
    if  policy_class == 'ACT':
        if temporal_agg:
            query_frequency = 1
        else:
            query_frequency = min(execution_horizon, num_queries)
    else:
        query_frequency = 1

    preprocess = lambda qpos: (qpos - stats['qpos_mean']) / stats['qpos_std']
    postprocess = lambda action: action * stats['action_std'] + stats['action_mean']

    print('\n=== 系统初始化完成 ===')
    print(f'cameras={camera_names}')
    print(
        f'hand={ROBOT_HAND}, dim={HAND_DIM}, '
        f'range=[{HAND_CONFIG["min"]}, {HAND_CONFIG["max"]}], '
        f'max_step={HAND_MAX_STEP}'
    )
    input('按Enter开始执行...')

    print('重置机械臂与灵巧手到初始位置...')
    left_arm.rm_movej(ARM_HOME, 60, 0, 0, 1)
    hand.move(HAND_INIT_POSE, force=True)
    time.sleep(3)

    try:
        print(f'\n=== 开始执行策略 (最大{max_steps}步) ===')
        print(
            f"按 'q' 退出, 按 'r' 重置机械臂 | "
            f"query_frequency={query_frequency} temporal_agg={temporal_agg}"
        )
        all_actions = None
        all_time_actions = None
        if temporal_agg and policy_class == 'ACT':
            all_time_actions = torch.zeros(
                [max_steps, max_steps + num_queries, ckpt_state_dim]
            ).cuda()

        for step in range(max_steps):
            print(f'\r执行步数: {step + 1}/{max_steps}', end='')

            obs = get_observation(left_arm, cams, hand, camera_names)
            if obs['qpos'] is None:
                print('\n观察获取失败')
                break

            qpos_processed = preprocess(obs['qpos'])
            qpos_tensor = torch.from_numpy(qpos_processed).float().cuda().unsqueeze(0)
            curr_image = images_to_tensor(obs['images'], camera_names)

            with torch.no_grad():
                if policy_class == 'ACT':
                    if step % query_frequency == 0:
                        all_actions = policy(qpos_tensor, curr_image)
                    if temporal_agg:
                        all_time_actions[[step], step:step + num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, step]
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        raw_action = all_actions[:, step % query_frequency]
                else:
                    raw_action = policy(qpos_tensor, curr_image)

            action = postprocess(raw_action.squeeze(0).cpu().numpy())
            safe_action = safe_action_filter(action, obs['qpos'])

            raw_diff = action[:6] - obs['qpos'][:6]
            sent_diff = safe_action[:6] - obs['qpos'][:6]

            print(
                f"\nstep={step + 1} "
                f"qpos={np.round(obs['qpos'][:6], 2)} "
                f"target={np.round(action[:6], 2)} "
                f"raw_diff={np.round(raw_diff, 2)} "
                f"sent_diff={np.round(sent_diff, 2)} "
                f"clipped={np.linalg.norm(action[:6] - safe_action[:6]):.3f}"
            )


            if not execute_action(left_arm, hand, safe_action):
                print('\n动作执行失败')
                break

            # 拼接 front | wrist 方便监控
            front_bgr = obs['images'].get(
                '_bgr_display',
                cv2.cvtColor(obs['images'][camera_names[0]], cv2.COLOR_RGB2BGR),
            )
            wrist_bgr = cv2.cvtColor(obs['images']['wrist_RGB'], cv2.COLOR_RGB2BGR) \
                if 'wrist_RGB' in obs['images'] else front_bgr
            display_img = np.concatenate([front_bgr, wrist_bgr], axis=1)
            cv2.putText(display_img, f'Step: {step + 1}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(display_img, "Press 'q' to quit, 'r' to reset", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow(
                f'RM65 + {ROBOT_HAND} ACT Deployment (front | wrist)',
                display_img,
            )

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print('\n用户退出')
                break
            elif key == ord('r'):
                print('\n重置机械臂...')
                left_arm.rm_movej(ARM_HOME, 60, 0, 0, 1)
                hand.move(HAND_INIT_POSE, force=True)
                time.sleep(2)
                # 重置后清空 temporal 缓冲，避免旧预测污染
                if all_time_actions is not None:
                    all_time_actions.zero_()
                all_actions = None

            time.sleep(0.03)

        print(f'\n执行完成，总步数: {step + 1}')

    except KeyboardInterrupt:
        print('\n用户中断')
    except Exception as e:
        print(f'\n执行出错: {e}')
    finally:
        cleanup_resources(left_arm=left_arm, hand=hand, cams=cams)
        print('✓ 资源清理完成')


if __name__ == '__main__':
    main()
