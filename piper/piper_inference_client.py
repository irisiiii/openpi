#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
松灵机械臂（Piper）叠衣服推理客户端
连接到 pi05_fold_cloth 推理服务器进行实际控制
"""

import time
import numpy as np
import cv2
import logging
import pathlib
from typing import Optional, Dict, Any
from openpi_client import websocket_client_policy as _websocket_client_policy

# 导入松灵机械臂控制器
from piper_ctl import PiperDualArmController

logger = logging.getLogger(__name__)


class PiperInferenceClient:
    """松灵机械臂推理客户端 - 连接到 pi05_fold_cloth 服务器"""
    
    def __init__(
        self, 
        server_host: str = "localhost",
        server_port: int = 8000,
        api_key: Optional[str] = None,
        left_can_port: str = "can_left",
        right_can_port: str = "can_right",
        camera_indices: tuple = (0, 1, 2),  # (top_camera, left_wrist, right_wrist)
    ):
        """
        初始化松灵机械臂推理客户端
        
        Args:
            server_host: 推理服务器地址
            server_port: 推理服务器端口
            api_key: API密钥（可选）
            left_can_port: 左臂CAN端口名
            right_can_port: 右臂CAN端口名
            camera_indices: 相机索引 (顶部相机, 左腕相机, 右腕相机)
        """
        # 初始化松灵机械臂控制器
        logger.info("初始化松灵机械臂控制器...")
        self.robot = PiperDualArmController(left_can_port, right_can_port)
        
        # 连接和使能机械臂
        if not self.robot.connect():
            raise RuntimeError("机械臂连接失败")
        if not self.robot.enable():
            raise RuntimeError("机械臂使能失败")
        
        logger.info("✓ 松灵机械臂就绪")
        
        # 创建 WebSocket 推理客户端
        logger.info(f"连接到推理服务器 {server_host}:{server_port}...")
        self.policy = _websocket_client_policy.WebsocketClientPolicy(
            host=server_host,
            port=server_port,
            api_key=api_key,
        )
        logger.info(f"✓ 服务器元数据: {self.policy.get_server_metadata()}")
        
        # 初始化相机
        self.camera_indices = camera_indices
        self.cameras = {}
        self._init_cameras()
        
        # 数据缓存
        self.current_state = None
        
        logger.info("✓ 松灵机械臂推理客户端初始化完成")
    
    def _init_cameras(self):
        """初始化相机"""
        logger.info("初始化相机...")
        for name, idx in zip(['top', 'left_wrist', 'right_wrist'], self.camera_indices):
            try:
                cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    # 设置分辨率
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    self.cameras[name] = cap
                    logger.info(f"✓ {name} 相机 (索引 {idx}) 已就绪")
                else:
                    logger.warning(f"⚠️ {name} 相机 (索引 {idx}) 打开失败")
                    self.cameras[name] = None
            except Exception as e:
                logger.error(f"❌ {name} 相机初始化失败: {e}")
                self.cameras[name] = None
    
    def get_camera_image(self, camera_name: str) -> Optional[np.ndarray]:
        """
        获取指定相机的图像
        
        Args:
            camera_name: 相机名称 ('top', 'left_wrist', 'right_wrist')
            
        Returns:
            图像数组 (H, W, 3) BGR格式，如果失败返回 None
        """
        cap = self.cameras.get(camera_name)
        if cap is None:
            return None
        
        ret, frame = cap.read()
        if not ret:
            logger.warning(f"⚠️ {camera_name} 相机读取失败")
            return None
        
        return frame
    
    def get_robot_state(self) -> Optional[Dict[str, np.ndarray]]:
        """
        获取机械臂当前状态
        
        Returns:
            包含左右臂状态的字典，格式:
            {
                'state.left_arm': [joint1-6(rad), gripper(mm), tcp_xyz(m), tcp_rpy(rad), gripper_effort],
                'state.right_arm': [joint1-6(rad), gripper(mm), tcp_xyz(m), tcp_rpy(rad), gripper_effort]
            }
        """
        try:
            state_dict = self.robot.get_current_state()
            if not state_dict:
                return None
            return state_dict
        except Exception as e:
            logger.error(f"❌ 获取机械臂状态失败: {e}")
            return None
    
    def get_observation(self, task_description: str = "Fold the cloth") -> Optional[Dict[str, Any]]:
        """
        获取当前观测数据，格式匹配 pi05_fold_cloth 模型输入
        
        Args:
            task_description: 任务描述（prompt）
            
        Returns:
            观测字典，格式:
            {
                'observation/state': np.ndarray,  # shape (14,) [left_arm_joints(6) + left_gripper(1) + right_arm_joints(6) + right_gripper(1)]
                'observation/images/top': np.ndarray,  # shape (3, H, W) RGB
                'observation/images/wrist_left': np.ndarray,  # shape (3, H, W) RGB
                'observation/images/wrist_right': np.ndarray,  # shape (3, H, W) RGB
                'prompt': str
            }
        """
        # 获取相机图像
        top_image = self.get_camera_image('top')
        left_wrist_image = self.get_camera_image('left_wrist')
        right_wrist_image = self.get_camera_image('right_wrist')
        
        if top_image is None or left_wrist_image is None or right_wrist_image is None:
            logger.warning("⚠️ 部分相机图像缺失")
            return None
        
        # 获取机械臂状态
        robot_state = self.get_robot_state()
        if robot_state is None:
            logger.warning("⚠️ 机械臂状态缺失")
            return None
        
        # 提取关节位置和夹爪位置
        # state.left_arm: [joint1-6(rad), gripper(mm), ...]
        # state.right_arm: [joint1-6(rad), gripper(mm), ...]
        left_arm_state = robot_state.get('state.left_arm', np.zeros(14))
        right_arm_state = robot_state.get('state.right_arm', np.zeros(14))
        
        # 构造 14 维状态向量: [left_joints(6), left_gripper(1), right_joints(6), right_gripper(1)]
        state = np.concatenate([
            left_arm_state[:6],   # 左臂关节角度 (弧度)
            left_arm_state[6:7],  # 左臂夹爪位置 (毫米)
            right_arm_state[:6],  # 右臂关节角度 (弧度)
            right_arm_state[6:7], # 右臂夹爪位置 (毫米)
        ]).astype(np.float32)
        
        # 转换图像格式: BGR -> RGB -> CHW (Channel, Height, Width)
        def process_image(img):
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_chw = np.transpose(img_rgb, (2, 0, 1))  # HWC -> CHW
            return img_chw.astype(np.uint8)
        
        # 构造观测字典
        obs = {
            'observation/state': state,
            'observation/images/top': process_image(top_image),
            'observation/images/wrist_left': process_image(left_wrist_image),
            'observation/images/wrist_right': process_image(right_wrist_image),
            'prompt': task_description,
        }
        
        return obs
    
    def send_action(self, action: np.ndarray, speed_percent: int = 50) -> bool:
        """
        发送动作到松灵机械臂
        
        Args:
            action: 14维动作数组 [left_joints(6), left_gripper(1), right_joints(6), right_gripper(1)]
            speed_percent: 速度百分比
            
        Returns:
            是否成功
        """
        if len(action) != 14:
            logger.error(f"❌ 动作维度错误: 期望14维，实际{len(action)}维")
            return False
        
        try:
            # 解析动作
            left_joints = action[:6].tolist()
            left_gripper = float(action[6])
            right_joints = action[7:13].tolist()
            right_gripper = float(action[13])
            
            # 发送到机械臂
            success = self.robot.send_joint_commands(
                left_joints=left_joints,
                right_joints=right_joints,
                left_gripper=left_gripper,
                right_gripper=right_gripper,
                speed_percent=speed_percent
            )
            
            return success
            
        except Exception as e:
            logger.error(f"❌ 发送动作失败: {e}")
            return False
    
    def run_control_loop(
        self,
        task_description: str = "Fold the cloth",
        max_steps: int = 1000,
        control_hz: float = 10.0,
        speed_percent: int = 50,
        warmup_steps: int = 2,
    ):
        """
        运行控制循环
        
        Args:
            task_description: 任务描述
            max_steps: 最大步数
            control_hz: 控制频率 (Hz)
            speed_percent: 机械臂速度百分比
            warmup_steps: 预热步数
        """
        logger.info("=" * 70)
        logger.info("开始控制循环")
        logger.info(f"任务: {task_description}")
        logger.info(f"控制频率: {control_hz} Hz")
        logger.info(f"速度百分比: {speed_percent}%")
        logger.info("=" * 70)
        
        control_interval = 1.0 / control_hz
        step_count = 0
        
        # 预热模型
        logger.info(f"预热模型 ({warmup_steps} 步)...")
        for i in range(warmup_steps):
            obs = self.get_observation(task_description)
            if obs is not None:
                try:
                    self.policy.infer(obs)
                    logger.info(f"✓ 预热步骤 {i+1}/{warmup_steps} 完成")
                except Exception as e:
                    logger.warning(f"⚠️ 预热步骤 {i+1} 失败: {e}")
        logger.info("✓ 模型预热完成")
        
        try:
            while step_count < max_steps:
                loop_start = time.time()
                
                # 1. 获取观测
                obs = self.get_observation(task_description)
                if obs is None:
                    logger.warning("⚠️ 观测数据不完整，跳过此步")
                    time.sleep(0.1)
                    continue
                
                # 2. 使用 policy.infer() 请求动作
                logger.info(f"\n[步骤 {step_count}] 请求动作...")
                inference_start = time.time()
                
                try:
                    action_result = self.policy.infer(obs)
                except Exception as e:
                    logger.error(f"❌ 推理失败: {e}")
                    break
                
                inference_time = (time.time() - inference_start) * 1000
                logger.info(f"✓ 推理耗时: {inference_time:.1f}ms")
                
                # 打印服务器时间统计
                if "server_timing" in action_result:
                    for key, value in action_result["server_timing"].items():
                        logger.info(f"  server_{key}: {value:.1f}ms")
                
                # 获取动作数据
                if "actions" not in action_result:
                    logger.error("❌ 响应中没有动作数据")
                    break
                
                actions = np.array(action_result["actions"])
                
                # 确保是2维数组
                if len(actions.shape) == 1:
                    actions = np.expand_dims(actions, axis=0)
                
                num_actions = actions.shape[0]
                logger.info(f"✓ 收到 {num_actions} 步动作，开始执行...")
                
                # 3. 执行每一步动作
                for i in range(num_actions):
                    action_start = time.time()
                    
                    # 发送动作
                    success = self.send_action(actions[i], speed_percent=speed_percent)
                    if not success:
                        logger.warning(f"⚠️ 第 {i+1}/{num_actions} 步动作发送失败")
                    
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
                logger.info(f"本轮耗时: {loop_time:.2f}秒 (请求+执行{num_actions}步)")
                
        except KeyboardInterrupt:
            logger.info("\n用户中断")
        except Exception as e:
            logger.error(f"❌ 控制循环异常: {e}")
            import traceback
            traceback.print_exc()
        finally:
            logger.info(f"=" * 70)
            logger.info(f"控制循环结束，共执行 {step_count} 步")
            logger.info(f"=" * 70)
    
    def cleanup(self):
        """清理资源"""
        logger.info("清理资源...")
        
        # 关闭相机
        for name, cap in self.cameras.items():
            if cap is not None:
                cap.release()
                logger.info(f"✓ {name} 相机已关闭")
        
        # 清理机械臂（回到零位并失能）
        try:
            self.robot.cleanup()
        except Exception as e:
            logger.warning(f"⚠️ 机械臂清理失败: {e}")
        
        logger.info("✓ 资源清理完成")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.cleanup()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="松灵机械臂叠衣服推理客户端")
    parser.add_argument("--host", type=str, default="localhost", help="推理服务器地址")
    parser.add_argument("--port", type=int, default=8000, help="推理服务器端口")
    parser.add_argument("--task", type=str, default="Fold the cloth", help="任务描述")
    parser.add_argument("--max-steps", type=int, default=1000, help="最大执行步数")
    parser.add_argument("--control-hz", type=float, default=10.0, help="控制频率 (Hz)")
    parser.add_argument("--speed", type=int, default=50, help="机械臂速度百分比 (1-100)")
    parser.add_argument("--left-can", type=str, default="can_left", help="左臂CAN端口")
    parser.add_argument("--right-can", type=str, default="can_right", help="右臂CAN端口")
    parser.add_argument("--cameras", type=str, default="0,1,2", help="相机索引 (top,left_wrist,right_wrist)")
    
    args = parser.parse_args()
    
    # 解析相机索引
    camera_indices = tuple(map(int, args.cameras.split(',')))
    if len(camera_indices) != 3:
        logger.error("❌ 相机索引必须是3个数字，例如: 0,1,2")
        return
    
    try:
        # 创建客户端
        with PiperInferenceClient(
            server_host=args.host,
            server_port=args.port,
            left_can_port=args.left_can,
            right_can_port=args.right_can,
            camera_indices=camera_indices,
        ) as client:
            # 运行控制循环
            client.run_control_loop(
                task_description=args.task,
                max_steps=args.max_steps,
                control_hz=args.control_hz,
                speed_percent=args.speed,
            )
    except KeyboardInterrupt:
        logger.info("\n程序被中断")
    except Exception as e:
        logger.error(f"❌ 程序异常: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    main()

