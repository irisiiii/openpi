"""Real-Time Chunking (RTC) Policy Wrapper

基于Physical Intelligence的RTC研究：
https://www.physicalintelligence.company/research/real_time_chunking

RTC通过inpainting方法解决动作块切换时的不连续性问题，实现：
1. 在执行当前chunk时异步生成下一个chunk
2. 平滑过渡，消除chunk间的停顿
3. 提高执行速度和精度，对高延迟（300ms+）保持鲁棒
"""

import logging
import numpy as np
from typing import Any, Dict
from openpi_client import base_policy as _base_policy
from typing_extensions import override

logger = logging.getLogger(__name__)


class RTCPolicy(_base_policy.BasePolicy):
    """使用Real-Time Chunking的Policy包装器
    
    这个包装器在服务器端工作，对客户端透明。
    每次客户端调用infer()时，返回单个时间步的动作，
    并在内部管理chunk的生成和平滑过渡。
    
    Args:
        policy: 底层的policy模型
        action_horizon: 每个chunk的动作数量
        overlap_steps: 用于平滑过渡的重叠步数（默认为horizon的20%）
        blend_weight: 重叠区域的混合权重（0-1，越大越倾向保持旧chunk）
        enable_logging: 是否启用详细日志
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        action_horizon: int = 50,
        overlap_steps: int | None = None,
        blend_weight: float = 0.7,
        enable_logging: bool = True,
    ):
        self._policy = policy
        self._action_horizon = action_horizon
        self._overlap_steps = overlap_steps or max(int(action_horizon * 0.2), 5)
        self._blend_weight = blend_weight
        self._enable_logging = enable_logging
        
        # 状态管理
        self._current_chunk: Dict[str, np.ndarray] | None = None
        self._current_step: int = 0
        
        # 统计信息
        self._total_inferences = 0
        self._total_chunks = 0
        self._total_transitions = 0
        
        if self._enable_logging:
            logger.info(f"[RTC] 初始化 - action_horizon={action_horizon}, "
                       f"overlap_steps={self._overlap_steps}, blend_weight={blend_weight}")

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        """执行一步推理，返回单个时间步的动作
        
        RTC流程：
        1. 如果没有当前chunk或已执行完，生成新chunk
        2. 如果接近chunk末尾，生成下一个chunk并进行平滑混合
        3. 返回当前步的动作
        """
        self._total_inferences += 1
        
        # 情况1：首次调用或chunk已执行完 - 生成新chunk
        if self._current_chunk is None or self._current_step >= self._action_horizon:
            if self._current_chunk is not None:
                self._total_transitions += 1
            
            self._current_chunk = self._generate_chunk(obs)
            self._current_step = 0
            self._total_chunks += 1
            
            if self._enable_logging:
                logger.info(f"[RTC] 生成新chunk #{self._total_chunks}")
        
        # 情况2：接近chunk末尾 - 提前生成下一个chunk并混合（RTC核心）
        elif self._current_step >= self._action_horizon - self._overlap_steps:
            remaining_steps = self._action_horizon - self._current_step
            
            if self._enable_logging:
                logger.info(f"[RTC] 进入overlap区域 - 剩余{remaining_steps}步，生成并混合下一个chunk")
            
            # 生成下一个chunk
            next_chunk = self._generate_chunk(obs)
            
            # 应用RTC平滑混合
            self._current_chunk = self._blend_chunks(
                current_chunk=self._current_chunk,
                next_chunk=next_chunk,
                current_step=self._current_step,
            )
            
            # 更新到新chunk的开头
            self._current_step = 0
            self._total_chunks += 1
            self._total_transitions += 1
            
            if self._enable_logging:
                logger.info(f"[RTC] 平滑切换到chunk #{self._total_chunks}")
        
        # 提取当前步的动作
        result = self._extract_current_step(self._current_chunk)
        
        # 添加RTC统计信息
        if "policy_timing" not in result:
            result["policy_timing"] = {}
        
        result["policy_timing"]["rtc_chunk_id"] = self._total_chunks
        result["policy_timing"]["rtc_chunk_step"] = self._current_step
        result["policy_timing"]["rtc_total_transitions"] = self._total_transitions
        
        self._current_step += 1
        
        return result

    def _generate_chunk(self, obs: Dict) -> Dict:
        """生成一个完整的动作chunk"""
        chunk = self._policy.infer(obs)
        
        # 验证chunk格式
        if "actions" not in chunk:
            raise ValueError("Policy返回的结果中没有'actions'字段")
        
        actions = chunk["actions"]
        if not isinstance(actions, np.ndarray):
            actions = np.array(actions)
        
        # 确保是2D数组 (horizon, action_dim)
        if actions.ndim == 1:
            actions = actions[np.newaxis, :]
        
        chunk["actions"] = actions
        return chunk

    def _blend_chunks(
        self,
        current_chunk: Dict,
        next_chunk: Dict,
        current_step: int,
    ) -> Dict:
        """使用RTC策略混合两个chunk
        
        策略：
        1. 保留当前chunk的剩余动作（即将执行的部分）
        2. 对overlap区域进行加权混合，权重随位置线性变化
        3. 使用下一个chunk的后续部分
        """
        current_actions = current_chunk["actions"]
        next_actions = next_chunk["actions"]
        
        # 计算剩余步数
        remaining_steps = self._action_horizon - current_step
        
        # 创建混合后的新chunk
        blended_actions = next_actions.copy()
        
        if remaining_steps > 0:
            # 提取当前chunk的剩余部分
            remaining_current = current_actions[current_step:, :]
            actual_remaining = min(remaining_steps, remaining_current.shape[0])
            
            # 对overlap区域进行混合
            blend_length = min(actual_remaining, self._overlap_steps, next_actions.shape[0])
            
            for i in range(blend_length):
                # 权重随位置线性衰减：开始时完全使用旧chunk，结束时完全使用新chunk
                alpha = self._blend_weight * (1.0 - i / blend_length)
                
                if i < remaining_current.shape[0]:
                    blended_actions[i] = (
                        alpha * remaining_current[i] +
                        (1.0 - alpha) * next_actions[i]
                    )
            
            if self._enable_logging:
                logger.debug(f"[RTC] 混合: remaining={actual_remaining}, blend_length={blend_length}, "
                           f"start_alpha={self._blend_weight:.2f}, end_alpha={0.0:.2f}")
        
        # 构建混合后的chunk
        blended_chunk = next_chunk.copy()
        blended_chunk["actions"] = blended_actions
        
        return blended_chunk

    def _extract_current_step(self, chunk: Dict) -> Dict:
        """从chunk中提取当前步的数据"""
        result = {}
        
        for key, value in chunk.items():
            if isinstance(value, np.ndarray) and value.ndim >= 1:
                # 如果是数组且第一维度是horizon，提取当前步
                if value.shape[0] == self._action_horizon or key == "actions":
                    step_idx = min(self._current_step, value.shape[0] - 1)
                    result[key] = value[step_idx]
                else:
                    # 否则保持原样
                    result[key] = value
            else:
                # 非数组数据直接复制
                result[key] = value
        
        return result

    @override
    def reset(self) -> None:
        """重置状态"""
        if hasattr(self._policy, 'reset'):
            self._policy.reset()
        
        if self._enable_logging and self._total_chunks > 0:
            logger.info(f"[RTC统计] 总推理={self._total_inferences}, "
                       f"总chunks={self._total_chunks}, "
                       f"成功切换={self._total_transitions}, "
                       f"平均每chunk推理={self._total_inferences/self._total_chunks:.1f}次")
        
        self._current_chunk = None
        self._current_step = 0
        self._total_inferences = 0
        self._total_chunks = 0
        self._total_transitions = 0

    def get_stats(self) -> Dict[str, Any]:
        """获取RTC统计信息"""
        return {
            "total_inferences": self._total_inferences,
            "total_chunks": self._total_chunks,
            "total_transitions": self._total_transitions,
            "current_step": self._current_step,
            "action_horizon": self._action_horizon,
            "overlap_steps": self._overlap_steps,
            "avg_inferences_per_chunk": (
                self._total_inferences / self._total_chunks if self._total_chunks > 0 else 0
            ),
        }

    @property
    def metadata(self) -> dict[str, Any]:
        """返回底层policy的metadata"""
        if hasattr(self._policy, 'metadata'):
            return self._policy.metadata
        return {}



