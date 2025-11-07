#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GR00T + Piper单臂HTTP客户端 (简化版)
参考官方示例，不使用RTC，直接请求-执行模式
"""

import time
import numpy as np
import requests
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge
import json
import cv2

# --- 新增代码: 导入并激活 json_numpy 以支持 numpy 数组的 JSON 序列化 ---
import json_numpy
json_numpy.patch()
# --- 新增结束 ---

# 导入自定义消息类型
from jaka_robot_interfaces.msg import MultiMovJCommand, JointValue, MoveMode
from jiazhua_interfaces.msg import JiaZhuaDualCmd
from frame_sync_msgs.msg import StampedFloat64MultiArray


class SimpleGr00tClient(Node):
    """简单的GR00T HTTP客户端"""
    
    def __init__(self, server_host="192.168.1.88", server_port=8000):
        super().__init__('groot_client')
        
        self.server_url = f"http://{server_host}:{server_port}/act"
        self.bridge = CvBridge()
        
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
        
        print(f"✓ 客户端初始化完成，服务器: {self.server_url}")
    
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
    
    def get_observation(self):
        """获取当前观测数据"""
        if (self.wrist_image is None or self.top_image is None or 
            self.joint_positions is None or self.gripper_position is None):
            return None
        
        # Resize wrist_image to 480x640
        wrist_image_resized = cv2.resize(self.wrist_image, (640, 480))
        
        # 构造observation字典，键名需要匹配服务器期望的格式
        obs = {
            "video.wrist_left": np.expand_dims(
                wrist_image_resized, axis=0
            ).astype(np.uint8),
            "video.top": np.expand_dims(
                self.top_image, axis=0
            ).astype(np.uint8),
            "state.left_arm_joint": np.expand_dims(
                self.joint_positions,
                axis=0
            ).astype(np.float32),
            "state.left_gripper": np.expand_dims(
                self.gripper_position,
                axis=0
            ).astype(np.float32),
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
    
    def request_action(self, obs, task_description="Complete the task"):
        """向服务器请求动作"""
        try:
            # 添加任务描述
            obs["annotation.human.action.task_description"] = [task_description]
            
            # --- 修改: 直接发送包含numpy数组的字典 ---
            # json_numpy补丁会自动处理序列化
            response = requests.post(
                self.server_url,
                json={"observation": obs},
                timeout=10.0
            )
            
            if response.status_code == 200:
                action_data = response.json()
                
                # --- 修改: 将返回的list转为numpy数组 ---
                # requests.json() 会将JSON数组解析为Python list
                # 我们需要将其转换回numpy数组以供下游使用
                for key, value in action_data.items():
                    if isinstance(value, list):
                        action_data[key] = np.array(value)
                return action_data
            else:
                print(f"❌ 服务器返回错误: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ 请求失败: {e}")
            return None
    
    def run_controprol_loop(self, task_description="Pick up the object", 
                        max_steps=1000, control_hz=20):
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
        
        try:
            while step_count < max_steps:
                loop_start = time.time()
                
                # 1. 获取观测
                obs = self.get_observation()
                if obs is None:
                    print("⚠️ 观测数据不完整")
                    time.sleep(0.1)
                    continue
                
                # 2. 请求动作chunk
                print(f"\n[步骤 {step_count}] 请求动作...")
                action_chunk = self.request_action(obs, task_description)
                
                if action_chunk is None:
                    print("❌ 未获取到有效动作 (action_chunk为None)")
                    break
                
                # 3. 解析动作chunk
                # 服务器返回格式: {'action.left_arm_joint': (N,7), 'action.left_gripper': (N,1)}
                # 需要合并为 (N,8) 的数组
                if 'action.left_arm_joint' in action_chunk and 'action.left_gripper' in action_chunk:
                    arm_actions = action_chunk['action.left_arm_joint']
                    gripper_actions = action_chunk['action.left_gripper']
                    
                    # 确保是numpy数组
                    if not isinstance(arm_actions, np.ndarray):
                        arm_actions = np.array(arm_actions)
                    if not isinstance(gripper_actions, np.ndarray):
                        gripper_actions = np.array(gripper_actions)
                    
                    # 合并为 (N, 8) 数组
                    actions = np.concatenate([arm_actions, gripper_actions], axis=1)
                    
                elif 'action_pred' in action_chunk:
                    # 兼容旧格式
                    actions = action_chunk['action_pred']
                    if not isinstance(actions, np.ndarray):
                        actions = np.array(actions)
                elif 'action' in action_chunk:
                    # 兼容另一种格式
                    actions = action_chunk['action']
                    if not isinstance(actions, np.ndarray):
                        actions = np.array(actions)
                else:
                    print(f"❌ 未知的动作格式，键名: {list(action_chunk.keys())}")
                    break
                
                # 确保是2维数组
                if len(actions.shape) == 1:
                    actions = np.expand_dims(actions, axis=0)
                
                num_actions = actions.shape[0]
                print(f"✓ 收到 {num_actions} 步动作，开始执行...")
                
                # 执行每一步动作
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
    
    client = SimpleGr00tClient(
        server_host="192.168.1.88",
        server_port=8000
    )
    
    try:
        # 运行控制循环
        client.run_control_loop(
            task_description="Pick up the green bowl and put it into the brown woven basket",
            max_steps=10000,
            control_hz=20
        )
    except KeyboardInterrupt:
        print("\n程序被中断")
    finally:
        client.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
