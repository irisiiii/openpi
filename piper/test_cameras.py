#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
相机测试工具
用于验证相机连接和查看相机画面
"""

import cv2
import numpy as np
import argparse


def test_single_camera(index: int, name: str = "Camera") -> bool:
    """
    测试单个相机
    
    Args:
        index: 相机索引
        name: 相机名称
        
    Returns:
        是否成功
    """
    print(f"\n测试 {name} (索引 {index})...")
    
    cap = cv2.VideoCapture(index)
    
    if not cap.isOpened():
        print(f"❌ {name} 打开失败")
        return False
    
    # 设置分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # 读取一帧
    ret, frame = cap.read()
    
    if not ret:
        print(f"❌ {name} 无法读取画面")
        cap.release()
        return False
    
    # 获取实际分辨率
    height, width = frame.shape[:2]
    print(f"✓ {name} 工作正常")
    print(f"  - 分辨率: {width}x{height}")
    print(f"  - 格式: {frame.dtype}")
    
    cap.release()
    return True


def view_cameras(indices: list):
    """
    实时查看多个相机画面
    
    Args:
        indices: 相机索引列表
    """
    print("\n开始实时查看相机画面...")
    print("按 'q' 或 ESC 退出\n")
    
    names = ['Top Camera', 'Left Wrist Camera', 'Right Wrist Camera']
    caps = []
    
    # 打开所有相机
    for i, idx in enumerate(indices):
        name = names[i] if i < len(names) else f"Camera {idx}"
        cap = cv2.VideoCapture(idx)
        
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            caps.append((cap, name))
            print(f"✓ {name} (索引 {idx}) 已打开")
        else:
            print(f"❌ {name} (索引 {idx}) 打开失败")
    
    if not caps:
        print("❌ 没有可用的相机")
        return
    
    print("\n显示相机画面中...")
    
    try:
        while True:
            frames = []
            
            # 读取所有相机的画面
            for cap, name in caps:
                ret, frame = cap.read()
                if ret:
                    # 添加文字标签
                    cv2.putText(
                        frame, name, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2
                    )
                    frames.append(frame)
                else:
                    # 创建黑色占位图
                    frames.append(np.zeros((480, 640, 3), dtype=np.uint8))
            
            # 如果有多个相机，拼接显示
            if len(frames) == 1:
                display = frames[0]
            elif len(frames) == 2:
                display = np.hstack(frames)
            elif len(frames) == 3:
                # 上面1个，下面2个
                top = frames[0]
                bottom = np.hstack(frames[1:])
                # 调整尺寸使其对齐
                if bottom.shape[1] > top.shape[1]:
                    top = cv2.resize(top, (bottom.shape[1], top.shape[0]))
                display = np.vstack([top, bottom])
            else:
                # 2x2 网格
                row1 = np.hstack(frames[:2])
                row2 = np.hstack(frames[2:4] if len(frames) >= 4 else 
                               [frames[2], np.zeros_like(frames[2])])
                display = np.vstack([row1, row2])
            
            # 显示
            cv2.imshow('Cameras Preview', display)
            
            # 检查按键
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' 或 ESC
                break
            
    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        # 释放所有相机
        for cap, name in caps:
            cap.release()
            print(f"✓ {name} 已关闭")
        
        cv2.destroyAllWindows()
        print("\n相机测试完成")


def list_cameras(max_index: int = 10):
    """
    列出所有可用的相机
    
    Args:
        max_index: 最大检测索引
    """
    print(f"检测相机 (索引 0-{max_index-1})...\n")
    
    available_cameras = []
    
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available_cameras.append(i)
                print(f"✓ 索引 {i}: 可用")
            else:
                print(f"⚠️ 索引 {i}: 打开成功但无法读取")
            cap.release()
    
    print(f"\n共找到 {len(available_cameras)} 个可用相机")
    if available_cameras:
        print(f"可用索引: {', '.join(map(str, available_cameras))}")
    
    return available_cameras


def main():
    parser = argparse.ArgumentParser(description="相机测试工具")
    parser.add_argument(
        '--mode', 
        type=str, 
        default='view',
        choices=['test', 'view', 'list'],
        help='测试模式: test=快速测试, view=实时查看, list=列出所有相机'
    )
    parser.add_argument(
        '--cameras',
        type=str,
        default='0,1,2',
        help='相机索引，逗号分隔 (例如: 0,1,2)'
    )
    parser.add_argument(
        '--max-index',
        type=int,
        default=10,
        help='list模式下的最大检测索引'
    )
    
    args = parser.parse_args()
    
    # 解析相机索引
    camera_indices = list(map(int, args.cameras.split(',')))
    
    print("=" * 60)
    print("相机测试工具")
    print("=" * 60)
    
    if args.mode == 'list':
        # 列出所有相机
        available = list_cameras(args.max_index)
        
        if available and len(available) >= 3:
            print(f"\n推荐配置: --cameras {','.join(map(str, available[:3]))}")
        
    elif args.mode == 'test':
        # 快速测试
        names = ['Top Camera', 'Left Wrist Camera', 'Right Wrist Camera']
        
        success_count = 0
        for i, idx in enumerate(camera_indices):
            name = names[i] if i < len(names) else f"Camera {i}"
            if test_single_camera(idx, name):
                success_count += 1
        
        print(f"\n测试结果: {success_count}/{len(camera_indices)} 个相机可用")
        
        if success_count == len(camera_indices):
            print("✅ 所有相机测试通过！")
        else:
            print("⚠️ 部分相机测试失败，请检查连接")
        
    elif args.mode == 'view':
        # 实时查看
        view_cameras(camera_indices)
    
    print("=" * 60)


if __name__ == '__main__':
    main()

