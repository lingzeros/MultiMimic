import yaml
import os
import sys
import argparse

# 设置项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(current_dir, "../..")))

from LinkerHand.linker_hand_api import LinkerHandApi
from LinkerHand.utils.load_write_yaml import LoadWriteYaml
from LinkerHand.utils.color_msg import ColorMsg

"""
LinkerHand 命令行控制脚本
通过位置列表控制灵巧手
"""

class HandController:
    def __init__(self, positions=None):
        self.yaml = LoadWriteYaml()
        self.setting = self.yaml.load_setting_yaml()
        self._init_hand()
        self.api = LinkerHandApi(hand_joint=self.hand_joint, hand_type=self.hand_type)
        self._set_default_speed()
        self.init_pos = self._get_default_positions(positions)

    def _init_hand(self):
        # 确定手类型（左右手互斥，优先左手）
        self.hand_type = "left" if self.setting['LINKER_HAND']['LEFT_HAND']['EXISTS'] else "right"
        print(f"使用{self.hand_type == 'left' and '左手' or '右手'}")
        self.hand_joint = self.setting['LINKER_HAND'][f"{self.hand_type.upper()}_HAND"]['JOINT']

    def _set_default_speed(self):
        speed_map = {
            "L7": [180, 250, 250, 250, 250, 250, 250],
            "L10": [180, 250, 250, 250, 250],
            "L20": [120, 180, 180, 180, 180],
            "L21": [60, 220, 220, 220, 220],
            "L25": [60, 250, 250, 250, 250]
        }
        speed = speed_map.get(self.hand_joint, [180, 250, 250, 250, 250])
        ColorMsg(msg=f"设置速度:{speed}", color="green")
        self.api.set_speed(speed)

    def _get_default_positions(self, positions):
        pos_map = {
            "L7": [250] * 7,
            "L10": [255] * 10,
            "L20": [255, 255, 255, 255, 255, 255, 10, 100, 180, 240, 245, 255, 255, 255, 255, 255, 255, 255, 255, 255],
            "L21": [96, 255, 255, 255, 255, 150, 114, 151, 189, 255, 180, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255],
            "L25": [96, 255, 255, 255, 255, 150, 114, 151, 189, 255, 180, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255]
        }
        return positions if positions else pos_map.get(self.hand_joint, [255] * 10)

    def control_hand(self, positions):
        expected_len = len(self.init_pos)
        if len(positions) != expected_len:
            ColorMsg(msg=f"错误: 控制信号长度 {len(positions)} 不匹配关节数量 {expected_len}", color="red")
            return
        ColorMsg(msg=f"执行控制信号: {positions}", color="green")
        self.api.finger_move(pose=positions)

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="通过控制信号控制 LinkerHand")
    # parser.add_argument("--positions", type=str, help="逗号分隔的位置列表，例如: 255,255,255,...")
    # args = parser.parse_args()
    # # "拇指根部", "食指根部", "中指根部", "无名指根部", "小指根部", "拇指侧摆", "食指侧摆",
    # # "中指侧摆", "无名指侧摆", "小指侧摆", "拇指横摆", "预留", "预留", "预留", "预留",
    # # "拇指中部", "预留", "预留", "预留", "预留", "拇指指尖", "食指指尖", "中指指尖",
    # # "无名指指尖", "小指指尖"
    positions = [50, 155, 125, 255, 255, 150, 114, 151, 189, 255]
    # if args.positions:
    #     try:
    #         positions = [int(p.strip()) for p in args.positions.split(',')]
    #     except ValueError:
    #         ColorMsg(msg="错误: 位置列表必须是逗号分隔的整数", color="red")
    #         sys.exit(1)

    controller = HandController(positions)
    controller.control_hand(positions or controller.init_pos)