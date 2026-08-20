#!/usr/bin/env python3
"""
单独测试 LinkerHand L10 get_state 回读。

用法:
  python3 test_l10_get_state.py
  python3 test_l10_get_state.py --hand_type left --can can0
"""

import argparse
import os
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.abspath(__file__))
_LINKER_HAND_DIR = os.path.join(_ROOT, 'linker_hand_python_sdk', 'LinkerHand')


def import_linker_hand_api():
    # 避免与仓库根目录 utils.py 冲突
    mod = sys.modules.get('utils')
    if mod is not None and not hasattr(mod, '__path__'):
        del sys.modules['utils']
        for name in list(sys.modules):
            if name.startswith('utils.'):
                del sys.modules[name]
    if _LINKER_HAND_DIR not in sys.path:
        sys.path.insert(0, _LINKER_HAND_DIR)
    from linker_hand_python_sdk.LinkerHand.linker_hand_api import LinkerHandApi
    return LinkerHandApi


def main():
    parser = argparse.ArgumentParser(description='Test LinkerHand L10 get_state')
    parser.add_argument('--hand_type', choices=['left', 'right'], default='right')
    parser.add_argument('--hand_joint', default='L10')
    parser.add_argument('--can', default='can0')
    parser.add_argument('--rounds', type=int, default=10, help='回读轮数')
    parser.add_argument('--wait', type=float, default=0.05, help='查询后等待秒数')
    args = parser.parse_args()

    pose_open = [255.0, 70.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0, 255.0]
    pose_mid = [180.0, 50.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 180.0, 200.0]

    print(f'=== L10 get_state 测试 ===')
    print(f'hand_type={args.hand_type} hand_joint={args.hand_joint} can={args.can}')

    LinkerHandApi = import_linker_hand_api()
    hand = LinkerHandApi(
        hand_type=args.hand_type,
        hand_joint=args.hand_joint,
        can=args.can,
    )

    # 底层对象（便于看 version / bus / x01 x04）
    low = hand.hand
    print(f'can_id=0x{getattr(low, "can_id", None):x}' if getattr(low, 'can_id', None) is not None else 'can_id=?')
    print(f'bus={getattr(low, "bus", None)}')
    print(f'version(raw)={getattr(low, "version", None)}')

    try:
        ver = hand.get_version() if hasattr(hand, 'get_version') else low.get_version()
    except Exception as e:
        ver = f'error: {e}'
    print(f'get_version() -> {ver}')

    print('\n[1] 未下发动作前直接 get_state')
    try:
        s0 = hand.get_state()
        print(f'  state={s0}')
    except Exception as e:
        print(f'  get_state failed: {e}')

    print('\n[2] finger_move(open) 后等待再读')
    hand.set_speed(speed=[180, 250, 250, 250, 250])
    hand.finger_move(pose=pose_open)
    time.sleep(0.5)
    for i in range(args.rounds):
        s = hand.get_state()
        x01 = getattr(low, 'x01', None)
        x04 = getattr(low, 'x04', None)
        ok = s is not None and len(s) >= 10 and all(0 <= float(v) <= 255 for v in s[:10])
        print(
            f'  round={i:02d} ok={ok} state={s} '
            f'x01={x01} x04={x04}'
        )
        time.sleep(args.wait)

    print('\n[3] finger_move(mid) 后等待再读（看是否跟随变化）')
    hand.finger_move(pose=pose_mid)
    time.sleep(0.5)
    for i in range(min(5, args.rounds)):
        s = hand.get_state()
        ok = s is not None and len(s) >= 10 and all(0 <= float(v) <= 255 for v in s[:10])
        diff = None
        if ok:
            diff = float(np.max(np.abs(np.array(s[:10], dtype=float) - np.array(pose_mid))))
        print(f'  round={i:02d} ok={ok} state={s} |state-cmd|_max={diff}')
        time.sleep(args.wait)

    print('\n判断:')
    print('  - 若一直 [-1]*10 且 version=None: CAN 回包未收到（接口/接线/左右手ID/固件）')
    print('  - 若稍后变为 0~255: 只是查询等待不足，部署里加 sleep 即可')
    print('  - 若能读但与 cmd 差很大: 回读可用，但仍建议部署用 cmd 作观测更稳')


if __name__ == '__main__':
    main()
