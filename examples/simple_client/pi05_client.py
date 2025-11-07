#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pi05 Jaka 机械臂 WebSocket 客户端
适配 OpenPi pi05_jaka 模型 - 使用 WebsocketClientPolicy
"""

import time
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge
import cv2
import logging
import pathlib

from openpi_client import websocket_client_policy as _websocket_client_policy

# 导入自定义消息类型
from jaka_robot_interfaces.msg import MultiMovJCommand, JointValue, MoveMode
from jiazhua_interfaces.msg import JiaZhuaDualCmd
from frame_sync_msgs.msg import StampedFloat64MultiArray

logger = logging.getLogger(__name__)


class Pi05JakaClient(Node):
    """Pi05 Jaka 模型的 WebSocket 客户端"""
    
    def __init__(self, server_host="192.168.1.88", server_port=8000, api_key=None):
        super().__init__('pi05_jaka_client')
        
        self.bridge = CvBridge()
        
        # 创建 WebSocket 策略客户端
        self.policy = _websocket_client_policy.WebsocketClientPolicy(
            host=server_host,
            port=server_port,
            api_key=api_key,
        )
        logger.info(f"Server metadata: {self.policy.get_server_metadata()}")
        
        # 数据缓存
        self.wrist_image = None
        self.top_image = None
        self.joint_positions = None
        self.gripper_position = None
        
        # 订阅话题
        self.create_subscription(
            Image, 
            '/camera1/left_camera/color/image_rect_raw',
            self.wrist_image_callback, 
            10
        )
        self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.top_image_callback,
            10
        )
        self.create_subscription(
            JointState,
            '/left_arm/joint_states',
            self.joint_state_callback,
            10
        )
        self.create_subscription(
            StampedFloat64MultiArray,
            '/left_arm/jiazhua_state',
            self.gripper_callback,
            10
        )
        
        # 发布话题
        self.arm_pub = self.create_publisher(
            MultiMovJCommand,
            '/multi_movj_cmd',
            10
        )
        self.gripper_pub = self.create_publisher(
            JiaZhuaDualCmd,
            '/jiazhua_cmd',
            10
        )
        
        print(f"✓ 客户端初始化完成，服务器: ws://{server_host}:{server_port}")
    
    def wrist_image_callback(self, msg):
        """左臂相机回调"""
        self.wrist_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
    
    def top_image_callback(self, msg):
        """头部相机回调"""
        self.top_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
    
    def joint_state_callback(self, msg):
        """关节状态回调"""
        if len(msg.position) >= 7:
            self.joint_positions = np.array(msg.position[:7], dtype=np.float32)
    
    def gripper_callback(self, msg):
        """夹爪状态回调"""
        if len(msg.data) > 0:
            self.gripper_position = np.array([msg.data[0]], dtype=np.float32)
    
    def get_observation(self, task_description="Pick up the object"):
        """获取当前观测数据，格式匹配 pi05 模型输入"""
        if (self.wrist_image is None or self.top_image is None or 
            self.joint_positions is None or self.gripper_position is None):
            return None
        
        # Resize wrist_image to 480x640 (匹配训练数据)
        wrist_image_resized = cv2.resize(self.wrist_image, (640, 480))
        
        # 合并关节和夹爪状态为 8 维 state
        state = np.concatenate([self.joint_positions, self.gripper_position])
        
        # 转换图像为 CHW 格式 (Channel, Height, Width)
        wrist_image_chw = np.transpose(wrist_image_resized, (2, 0, 1))  # (480, 640, 3) -> (3, 480, 640)
        top_image_chw = np.transpose(self.top_image, (2, 0, 1))  # (480, 640, 3) -> (3, 480, 640)
        
        # 构造observation字典
        obs = {
            "state": state.astype(np.float32),
            "wrist_image_left": wrist_image_chw.astype(np.uint8),  # CHW 格式
            "top": top_image_chw.astype(np.uint8),  # CHW 格式
            "prompt": task_description,
        }
        
        return obs
    
    def send_action(self, action):
        """发送动作到机械臂和夹爪
        
        Args:
            action: 8维数组 [7个关节角度, 1个夹爪位置]
        """
        # 发布机械臂指令
        arm_cmd = MultiMovJCommand()
        arm_cmd.robot_id = 0  # LEFT(0) - 只控制左臂
        
        # 设置运动模式为绝对位置模式
        arm_cmd.left_move_mode = MoveMode()
        arm_cmd.left_move_mode.mode = 0  # ABS = 0
        arm_cmd.right_move_mode = MoveMode()
        arm_cmd.right_move_mode.mode = 0  # ABS = 0
        
        arm_cmd.is_block = False  # 非阻塞模式
        
        # 设置关节位置
        arm_cmd.joint_pos_left = JointValue()
        arm_cmd.joint_pos_left.joint_values = action[:7].tolist()
        arm_cmd.joint_pos_right = JointValue()
        arm_cmd.joint_pos_right.joint_values = [0.0] * 7  # 右臂不动
        
        # 设置速度和加速度 (单位: rad/s 和 rad/s^2)
        arm_cmd.vel = [0.5, 0.5]  # 左臂和右臂的速度
        arm_cmd.acc = [2.0, 2.0]  # 左臂和右臂的加速度
        
        self.arm_pub.publish(arm_cmd)
        
        # 发布夹爪指令
        gripper_cmd = JiaZhuaDualCmd()
        gripper_cmd.val_left = float(action[7])
        gripper_cmd.speed_left = 0.5
        gripper_cmd.val_right = 1.0  # 右手不动，保持张开
        gripper_cmd.speed_right = 0.5
        self.gripper_pub.publish(gripper_cmd)
    
    def run_control_loop(self, task_description="Pick up the green bowl and put it into the brown woven basket",
                        max_steps=1000, control_hz=10):
        """运行控制循环"""
        print(f"=== 开始控制循环 ===")
        print(f"任务: {task_description}")
        print(f"控制频率: {control_hz} Hz")
        
        control_interval = 1.0 / control_hz
        step_count = 0
        
        # 等待数据就绪
        print("等待传感器数据...")
        while (self.wrist_image is None or self.top_image is None or
               self.joint_positions is None or self.gripper_position is None):
            rclpy.spin_once(self, timeout_sec=0.1)
            time.sleep(0.1)
        print("✓ 传感器数据就绪")
        
        # 发送几个观测以确保模型加载
        print("预热模型...")
        for _ in range(2):
            obs = self.get_observation(task_description)
            if obs is not None:
                self.policy.infer(obs)
        print("✓ 模型预热完成")
        
        try:
            while step_count < max_steps:
                loop_start = time.time()
                
                # 1. 获取观测
                obs = self.get_observation(task_description)
                if obs is None:
                    print("⚠️ 观测数据不完整")
                    time.sleep(0.1)
                    continue
                
                # 2. 使用 policy.infer() 请求动作
                print(f"\n[步骤 {step_count}] 请求动作...")
                inference_start = time.time()
                
                action_result = self.policy.infer(obs)
                
                inference_time = (time.time() - inference_start) * 1000
                print(f"✓ 推理耗时: {inference_time:.1f}ms")
                
                # 打印服务器时间统计
                if "server_timing" in action_result:
                    for key, value in action_result["server_timing"].items():
                        print(f"  server_{key}: {value:.1f}ms")
                
                # 获取动作数据
                if "actions" not in action_result:
                    print("❌ 响应中没有动作数据")
                    break
                
                actions = np.array(action_result["actions"])
                
                # 确保是2维数组
                if len(actions.shape) == 1:
                    actions = np.expand_dims(actions, axis=0)
                
                num_actions = actions.shape[0]
                print(f"✓ 收到 {num_actions} 步动作，开始执行...")
                
                # 3. 执行每一步动作
                for i in range(num_actions):
                    action_start = time.time()
                    
                    # 发送动作
                    self.send_action(actions[i])
                    
                    # 处理ROS回调
                    rclpy.spin_once(self, timeout_sec=0.0)
                    
                    # 控制频率
                    elapsed = time.time() - action_start
                    sleep_time = control_interval - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
                    
                    step_count += 1
                    
                    if step_count >= max_steps:
                        break
                
                # 打印循环时间
                loop_time = time.time() - loop_start
                print(f"本轮耗时: {loop_time:.2f}秒 (请求+执行{num_actions}步)")
                
        except KeyboardInterrupt:
            print("\n用户中断")
        except Exception as e:
            print(f"❌ 控制循环异常: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print(f"=== 控制循环结束，共执行 {step_count} 步 ===")


def main():
    """主函数"""
    rclpy.init()
    
    client = Pi05JakaClient(
        server_host="192.168.1.88",
        server_port=8000,
        api_key=None  # 如果需要可以添加 API key
    )
    
    try:
        # 运行控制循环
        client.run_control_loop(
            task_description="Pick up the green bowl and put it into the brown woven basket",
            max_steps=10000,
            control_hz=10  # Pi05 模型推理较慢，降低频率
        )
    except KeyboardInterrupt:
        print("\n程序被中断")
    finally:
        client.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()