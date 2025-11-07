#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
双机械臂控制SDK
支持关节命令模式，数据格式与采集程序一致
"""

import time
import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from piper_sdk import *

class PiperDualArmController:
    """双机械臂控制器类"""
    
    def __init__(self, left_can_port: str = "can_left", right_can_port: str = "can_right"):
        """
        初始化双机械臂控制器
        
        Args:
            left_can_port: 左臂CAN端口名
            right_can_port: 右臂CAN端口名
        """
        self.left_can_port = left_can_port
        self.right_can_port = right_can_port
        
        # 机械臂接口
        self.piper_left = None
        self.piper_right = None
        
        # 连接状态
        self.is_connected = False
        self.is_enabled = False
        
        # 数据格式常数
        self.RAD_TO_PIPER = 57295.7795  # 弧度转piper内部单位 (1000*180/π)
        
        # --- [修正] ---
        # 原值为 1000000，导致夹爪指令值放大了1000倍。
        # SDK GripperCtrl 的 gripper_angle 参数单位是 0.001mm (微米)。
        # 因此，从毫米(mm)转换到该单位，需要乘以 1000。
        self.MM_TO_PIPER = 1000      # 毫米(mm)转piper内部单位(0.001mm)
        
        print(f"PiperDualArmController 初始化完成")
        print(f"左臂端口: {self.left_can_port}")
        print(f"右臂端口: {self.right_can_port}")

    def connect(self) -> bool:
        """
        连接双机械臂
        
        Returns:
            bool: 连接是否成功
        """
        try:
            print("正在连接双机械臂...")
            
            # 初始化机械臂接口
            self.piper_left = C_PiperInterface_V2(self.left_can_port)
            self.piper_right = C_PiperInterface_V2(self.right_can_port)
            
            # 连接端口
            self.piper_left.ConnectPort()
            self.piper_right.ConnectPort()
            
            # 等待连接稳定
            time.sleep(1.0)
            
            self.is_connected = True
            print("双机械臂连接成功!")
            return True
            
        except Exception as e:
            print(f"双机械臂连接失败: {e}")
            self.is_connected = False
            return False

    def enable(self) -> bool:
        """
        使能双机械臂
        
        Returns:
            bool: 使能是否成功
        """
        if not self.is_connected:
            print("机械臂未连接，请先调用connect()")
            return False
            
        try:
            print("正在使能双机械臂...")
            
            # 使能左臂
            while not self.piper_left.EnablePiper():
                time.sleep(0.01)
            print("左臂使能成功")
            
            # 使能右臂  
            while not self.piper_right.EnablePiper():
                time.sleep(0.01)
            print("右臂使能成功")
            
            # 等待使能稳定
            time.sleep(0.1)

            # --- [优化] ---
            # 在使能后立即设置运动模式，避免在发送命令时重复设置。
            # 0x01: CAN指令控制模式, 0x01: MOVE J (关节模式)
            self.piper_left.ModeCtrl(0x01, 0x01, 100, 0x00)
            self.piper_right.ModeCtrl(0x01, 0x01, 100, 0x00)
            
            self.is_enabled = True
            print("双机械臂使能完成，并已设置为关节控制模式!")
            return True
            
        except Exception as e:
            print(f"双机械臂使能失败: {e}")
            self.is_enabled = False
            return False

    def disable(self) -> bool:
        """
        失能双机械臂
        
        Returns:
            bool: 失能是否成功
        """
        if not self.is_connected:
            return True
            
        try:
            print("正在失能双机械臂...")
            
            # 失能左臂
            while self.piper_left.DisablePiper():
                time.sleep(0.01)
            print("左臂失能成功")
            
            # 失能右臂
            while self.piper_right.DisablePiper():
                time.sleep(0.01)
            print("右臂失能成功")
            
            self.is_enabled = False
            print("双机械臂失能完成!")
            return True
            
        except Exception as e:
            print(f"双机械臂失能失败: {e}")
            return False

    def send_joint_commands(self, left_joints: List[float], right_joints: List[float], 
                          left_gripper: float, right_gripper: float, 
                          speed_percent: int = 100) -> bool:
        """
        发送关节命令到双机械臂
        
        Args:
            left_joints: 左臂6个关节角度，单位弧度
            right_joints: 右臂6个关节角度，单位弧度
            left_gripper: 左臂夹爪位置，单位毫米
            right_gripper: 右臂夹爪位置，单位毫米
            speed_percent: 速度百分比 (1-100)
            
        Returns:
            bool: 命令发送是否成功
        """
        if not self.is_enabled:
            print("机械臂未使能，请先调用enable()")
            return False
            
        try:
            # 转换左臂关节角度为piper内部单位
            left_joint_piper = [round(joint * self.RAD_TO_PIPER) for joint in left_joints]
            
            # 转换右臂关节角度为piper内部单位
            right_joint_piper = [round(joint * self.RAD_TO_PIPER) for joint in right_joints]
            
            # 转换夹爪位置为piper内部单位 (毫米 -> 0.001毫米)
            left_gripper_piper = round(abs(left_gripper) * self.MM_TO_PIPER)
            right_gripper_piper = round(abs(right_gripper) * self.MM_TO_PIPER)
            
            # --- [优化] ---
            # 移除重复的模式设置调用，已在enable()中完成。
            # 如果需要动态改变速度，可以保留这两行，但通常不需要。
            # self.piper_left.MotionCtrl_2(0x01, 0x01, speed_percent, 0x00)
            # self.piper_right.MotionCtrl_2(0x01, 0x01, speed_percent, 0x00)
            
            # 发送左臂关节命令
            self.piper_left.JointCtrl(
                left_joint_piper[0], left_joint_piper[1], left_joint_piper[2],
                left_joint_piper[3], left_joint_piper[4], left_joint_piper[5]
            )
            
            # 发送右臂关节命令
            self.piper_right.JointCtrl(
                right_joint_piper[0], right_joint_piper[1], right_joint_piper[2],
                right_joint_piper[3], right_joint_piper[4], right_joint_piper[5]
            )
            
            # 发送夹爪命令
            self.piper_left.GripperCtrl(left_gripper_piper, 1000, 0x01, 0)
            self.piper_right.GripperCtrl(right_gripper_piper, 1000, 0x01, 0)
            
            return True
            
        except Exception as e:
            print(f"发送关节命令失败: {e}")
            return False

    def send_action_dict(self, action_dict: Dict[str, Any], speed_percent: int = 100) -> bool:
        """
        发送action字典格式的命令 (与GR00T模型输出兼容)
        
        Args:
            action_dict: 包含动作数据的字典，支持两种格式:
                        格式1 (传统): {
                            'action.left_arm_joints': [joint1-6弧度, gripper_mm],
                            'action.right_arm': [joint1-6弧度, gripper_mm] 
                        }
                        格式2 (分离): {
                            'action.left_arm_joints': [joint1-6弧度] 或 shape(N, 6),
                            'action.left_gripper': gripper_mm 或 shape(N,),
                            'action.right_arm_joints': [joint1-6弧度] 或 shape(N, 6),
                            'action.right_gripper': gripper_mm 或 shape(N,)
                        }
            speed_percent: 速度百分比
            
        Returns:
            bool: 命令发送是否成功
        """
        try:
            # 检测格式并解析左臂数据
            if 'action.left_arm_joints' in action_dict and 'action.left_gripper' in action_dict:
                # 格式2: 分离的关节和夹爪数据
                left_joints_data = action_dict['action.left_arm_joints']
                left_gripper_data = action_dict['action.left_gripper']
                
                # 转换为numpy数组便于处理
                if not isinstance(left_joints_data, np.ndarray):
                    left_joints_data = np.array(left_joints_data)
                if not isinstance(left_gripper_data, np.ndarray):
                    left_gripper_data = np.array(left_gripper_data)
                
                # 取第一个时间步的数据（如果是时间序列）
                if len(left_joints_data.shape) > 1:
                    left_joints = left_joints_data[0].tolist()  # shape (N, 6) -> 取第0个
                else:
                    left_joints = left_joints_data.tolist()  # shape (6,)
                
                if len(left_gripper_data.shape) > 0 and left_gripper_data.shape[0] > 1:
                    left_gripper = float(left_gripper_data[0])  # shape (N,) -> 取第0个
                else:
                    left_gripper = float(left_gripper_data.flat[0])  # 安全获取第一个元素
                    
            # elif 'action.left_arm_joints' in action_dict:
            #     # 格式1: 传统格式
            #     left_data = action_dict['action.left_arm_joints']
            #     if isinstance(left_data, np.ndarray):
            #         left_data = left_data.tolist()
            #     left_joints = left_data[:6]  # 前6个是关节角度
            #     left_gripper = left_data[6] if len(left_data) > 6 else 0.0  # 第7个是夹爪
            else:
                print("警告: 未找到左臂数据，使用零值")
                left_joints = [0.0] * 6
                left_gripper = 0.0
            
            # 检测格式并解析右臂数据
            if 'action.right_arm_joints' in action_dict and 'action.right_gripper' in action_dict:
                # 格式2: 分离的关节和夹爪数据
                right_joints_data = action_dict['action.right_arm_joints']
                right_gripper_data = action_dict['action.right_gripper']
                
                # 转换为numpy数组便于处理
                if not isinstance(right_joints_data, np.ndarray):
                    right_joints_data = np.array(right_joints_data)
                if not isinstance(right_gripper_data, np.ndarray):
                    right_gripper_data = np.array(right_gripper_data)
                
                # 取第一个时间步的数据（如果是时间序列）
                if len(right_joints_data.shape) > 1:
                    right_joints = right_joints_data[0].tolist()  # shape (N, 6) -> 取第0个
                else:
                    right_joints = right_joints_data.tolist()  # shape (6,)
                
                if len(right_gripper_data.shape) > 0 and right_gripper_data.shape[0] > 1:
                    right_gripper = float(right_gripper_data[0])  # shape (N,) -> 取第0个
                else:
                    right_gripper = float(right_gripper_data.flat[0])  # 安全获取第一个元素
                    
            # elif 'action.right_arm' in action_dict:
            #     # 格式1: 传统格式
            #     right_data = action_dict['action.right_arm']
            #     if isinstance(right_data, np.ndarray):
            #         right_data = right_data.tolist()
            #     right_joints = right_data[:6]  # 前6个是关节角度
            #     right_gripper = right_data[6] if len(right_data) > 6 else 0.0  # 第7个是夹爪
            else:
                print("警告: 未找到右臂数据，使用零值")
                right_joints = [0.0] * 6
                right_gripper = 0.0
            
            # 发送命令
            return self.send_joint_commands(left_joints, right_joints, 
                                          left_gripper, right_gripper, speed_percent)
            
        except Exception as e:
            print(f"解析action字典失败: {e}")
            return False

    def send_action_sequence(self, action_dict: Dict[str, Any], 
                           sequence_frequency: float = 30.0,
                           speed_percent: int = 100,
                           execute_last_only: bool = False) -> bool:
        """
        发送动作序列（时间序列）到双机械臂
        
        Args:
            action_dict: 包含时间序列动作数据的字典，格式:
                        {
                            'action.left_arm_joints': numpy array shape (N, 6),
                            'action.left_gripper': numpy array shape (N,),
                            'action.right_arm_joints': numpy array shape (N, 6),
                            'action.right_gripper': numpy array shape (N,)
                        }
            sequence_frequency: 序列执行频率 (Hz)
            speed_percent: 速度百分比
            execute_last_only: 如果为True，只执行序列的最后一个动作
            
        Returns:
            bool: 序列发送是否成功
        """
        try:
            # 检查是否包含必要的键
            required_keys = ['action.left_arm_joints', 'action.left_gripper', 
                           'action.right_arm_joints', 'action.right_gripper']
            if not all(key in action_dict for key in required_keys):
                print(f"❌ 动作字典缺少必要的键: {required_keys}")
                return False
            
            # 获取数据并转换为numpy数组
            left_joints_seq = np.array(action_dict['action.left_arm_joints'])
            left_gripper_seq = np.array(action_dict['action.left_gripper'])
            right_joints_seq = np.array(action_dict['action.right_arm_joints'])
            right_gripper_seq = np.array(action_dict['action.right_gripper'])
            
            # 检查数据形状
            sequence_length = left_joints_seq.shape[0]
            if (left_gripper_seq.shape[0] != sequence_length or
                right_joints_seq.shape[0] != sequence_length or
                right_gripper_seq.shape[0] != sequence_length):
                print(f"❌ 动作序列长度不一致")
                return False
            
            # 如果只执行最后一个动作
            if execute_last_only:
                print(f"🎯 只执行动作序列的最后一个动作 (第 {sequence_length} 步)")
                
                # 获取最后一步的动作
                left_joints = left_joints_seq[-1].tolist()
                left_gripper = float(left_gripper_seq[-1])
                right_joints = right_joints_seq[-1].tolist()
                right_gripper = float(right_gripper_seq[-1])
                
                # 发送最后一步的命令
                success = self.send_joint_commands(left_joints, right_joints, 
                                                 left_gripper, right_gripper, speed_percent)
                if not success:
                    print(f"❌ 最后一步动作发送失败")
                    return False
                
                print(f"✅ 最后一步动作执行完成")
                return True
            
            # 原有的执行整个序列的代码（通过注释可以选择性禁用）
            print(f"🎯 开始执行动作序列，长度: {sequence_length}, 频率: {sequence_frequency} Hz")
            
            # 计算每步的时间间隔
            step_interval = 1.0 / sequence_frequency
            
            # 逐步执行动作序列
            for i in range(sequence_length):
                if False:
                    continue
                step_start_time = time.time()
                
                # 获取当前步的动作
                left_joints = left_joints_seq[i].tolist()
                left_gripper = float(left_gripper_seq[i])
                right_joints = right_joints_seq[i].tolist()
                right_gripper = float(right_gripper_seq[i])
                
                # 发送当前步的命令
                success = self.send_joint_commands(left_joints, right_joints, 
                                                 left_gripper, right_gripper, speed_percent)
                if not success:
                    print(f"❌ 第 {i+1}/{sequence_length} 步动作发送失败")
                    return False
                
                # 调试信息
                if i % 5 == 0 or i == sequence_length - 1:  # 每5步或最后一步打印信息
                    print(f"✓ 第 {i+1}/{sequence_length} 步动作已发送")
                
                # 控制执行频率
                step_time = time.time() - step_start_time
                if step_time < step_interval:
                    time.sleep(step_interval - step_time)
            
            print(f"✅ 动作序列执行完成，共 {sequence_length} 步")
            return True
            
        except Exception as e:
            print(f"❌ 发送动作序列失败: {e}")
            return False

    def get_current_state(self) -> Dict[str, Any]:
        """
        获取当前机械臂状态
        
        Returns:
            Dict: 包含当前状态的字典，格式与采集程序一致
        """
        if not self.is_connected:
            return {}
            
        try:
            state_dict = {}
            
            # 获取左臂状态
            left_state = self._get_arm_state(self.piper_left, "left")
            if left_state:
                state_dict.update(left_state)
            
            # 获取右臂状态
            right_state = self._get_arm_state(self.piper_right, "right")
            if right_state:
                state_dict.update(right_state)
            
            return state_dict
            
        except Exception as e:
            print(f"获取当前状态失败: {e}")
            return {}

    def _get_arm_state(self, piper, arm_name: str) -> Dict[str, Any]:
        """
        获取单个机械臂的状态
        
        Args:
            piper: 机械臂接口对象
            arm_name: 机械臂名称 ("left" 或 "right")
            
        Returns:
            Dict: 机械臂状态数据
        """
        try:
            # 获取原始数据
            joint_data = piper.GetArmJointMsgs()
            gripper_data = piper.GetArmGripperMsgs()
            end_pose_data = piper.GetArmEndPoseMsgs()
            
            if not joint_data or not end_pose_data:
                return {}
            
            joint_state = joint_data.joint_state
            gripper_state = gripper_data.gripper_state if gripper_data else None
            end_pose = end_pose_data.end_pose
            
            # 转换为标准格式 (与采集程序一致)
            # [joint1-6(弧度), gripper_pos(mm), tcp_x(m), tcp_y(m), tcp_z(m), 
            #  tcp_roll(rad), tcp_pitch(rad), tcp_yaw(rad), gripper_effort]
            state_data = [
                # 关节角度(弧度)
                joint_state.joint_1 / self.RAD_TO_PIPER,
                joint_state.joint_2 / self.RAD_TO_PIPER,
                joint_state.joint_3 / self.RAD_TO_PIPER,
                joint_state.joint_4 / self.RAD_TO_PIPER,
                joint_state.joint_5 / self.RAD_TO_PIPER,
                joint_state.joint_6 / self.RAD_TO_PIPER,
                
                # 夹爪位置(毫米)
                gripper_state.grippers_angle / 1000.0 if gripper_state else 0.0,
                
                # TCP位置(米)
                end_pose.X_axis / 1000000.0,
                end_pose.Y_axis / 1000000.0,
                end_pose.Z_axis / 1000000.0,
                
                # TCP姿态(弧度)
                end_pose.RX_axis / 1000.0 * math.pi / 180.0,
                end_pose.RY_axis / 1000.0 * math.pi / 180.0,
                end_pose.RZ_axis / 1000.0 * math.pi / 180.0,
                
                # 夹爪扭矩
                gripper_state.grippers_effort / 1000.0 if gripper_state else 0.0
            ]
            
            return {f"state.{arm_name}_arm": np.array(state_data)}
            
        except Exception as e:
            print(f"获取{arm_name}臂状态失败: {e}")
            return {}

    def go_to_zero_position(self, speed_percent: int = 30) -> bool:
        """
        移动到零位（所有关节角度为0，夹爪闭合）
        
        预设位置说明:
        - 左臂: 所有关节角度为0，夹爪完全闭合
        - 右臂: 所有关节角度为0，夹爪完全闭合
        
        Args:
            speed_percent: 速度百分比
            
        Returns:
            bool: 是否成功
        """
        if not self.is_enabled:
            print("机械臂未使能，请先调用enable()")
            return False
            
        try:
            print("正在移动到零位...")
            
            # 设置运动控制模式
            self.piper_left.ModeCtrl(0x01, 0x01, speed_percent, 0x00)
            self.piper_right.ModeCtrl(0x01, 0x01, speed_percent, 0x00)
            
            # 发送零位命令（所有关节为0）
            self.piper_left.JointCtrl(0, 0, 0, 0, 0, 0)
            self.piper_right.JointCtrl(0, 0, 0, 0, 0, 0)
            
            # 夹爪完全闭合
            self.piper_left.GripperCtrl(0, 1000, 0x01, 0)
            self.piper_right.GripperCtrl(0, 1000, 0x01, 0)
            
            print("零位命令已发送")
            print(f"左臂关节位置: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] (弧度)")
            print(f"右臂关节位置: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] (弧度)")
            print(f"左爪位置: 0.0mm (闭合), 右爪位置: 0.0mm (闭合)")
            return True
            
        except Exception as e:
            print(f"移动到零位失败: {e}")
            return False

    def go_to_true_zero_position(self, speed_percent: int = 30) -> bool:
        """
        移动到真正的零位（所有关节角度为0）
        
        Args:
            speed_percent: 速度百分比
            
        Returns:
            bool: 是否成功
        """
        if not self.is_enabled:
            print("机械臂未使能，请先调用enable()")
            return False
            
        try:
            print("正在回到真正的零位...")
            
            # 设置运动控制模式
            self.piper_left.ModeCtrl(0x01, 0x01, speed_percent, 0x00)
            self.piper_right.ModeCtrl(0x01, 0x01, speed_percent, 0x00)
            
            # 发送真正的零位命令
            self.piper_left.JointCtrl(0, 0, 0, 0, 0, 0)
            self.piper_right.JointCtrl(0, 0, 0, 0, 0, 0)
            
            # 夹爪完全闭合
            self.piper_left.GripperCtrl(0, 1000, 0x01, 0)
            self.piper_right.GripperCtrl(0, 1000, 0x01, 0)
            
            print("真正的零位命令已发送")
            return True
            
        except Exception as e:
            print(f"回到真正零位失败: {e}")
            return False

    def cleanup(self):
        """清理资源"""
        try:
            if self.is_enabled:
                self.go_to_zero_position()
                time.sleep(1.0)
                # self.disable()
            self.is_connected = False
            print("资源清理完成")
        except:
            pass

    def __enter__(self):
        """上下文管理器入口"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.cleanup()

    def is_ready(self) -> bool:
        """检查机械臂是否就绪"""
        return self.is_connected and self.is_enabled

    def emergency_stop(self) -> bool:
        """紧急停止"""
        try:
            if self.piper_left:
                self.piper_left.MotionCtrl_1(0x01, 0, 0)
            if self.piper_right:
                self.piper_right.MotionCtrl_1(0x01, 0, 0)
            print("紧急停止已执行")
            return True
        except Exception as e:
            print(f"紧急停止失败: {e}")
            return False

    def print_status(self):
        """打印当前状态"""
        print(f"=== 双机械臂状态 ===")
        print(f"连接状态: {'已连接' if self.is_connected else '未连接'}")
        print(f"使能状态: {'已使能' if self.is_enabled else '未使能'}")
        print(f"就绪状态: {'就绪' if self.is_ready() else '未就绪'}")
        print(f"左臂端口: {self.left_can_port}")
        print(f"右臂端口: {self.right_can_port}")


# 测试代码
if __name__ == "__main__":
    # 使用示例
    controller = PiperDualArmController("can_left", "can_right")
    
    try:
        # 连接和使能
        if controller.connect() and controller.enable():
            controller.print_status()
            
            # 移动到预设初始位置（基于ROS2话题数据）
            controller.go_to_zero_position()
            time.sleep(3)
            
            # 测试关节命令
            test_left_joints = [0.1, 0.1, -0.1, 0.2, -0.1, 0.3]
            test_right_joints = [-0.1, -0.1, 0.1, -0.2, 0.1, -0.3]
            controller.send_joint_commands(test_left_joints, test_right_joints, 10.0, 10.0)
            
            time.sleep(2)
            
            # 获取当前状态
            current_state = controller.get_current_state()
            print("当前状态:", current_state)
            
    except KeyboardInterrupt:
        print("接收到键盘中断")
    finally:
        controller.cleanup()
