# Real-Time Chunking (RTC) 使用指南

## 什么是 RTC？

**Real-Time Chunking (RTC)** 是Physical Intelligence开发的实时动作分块技术，用于解决机器人控制中的两个核心问题：

1. **推理延迟**：模型推理需要时间，期间机器人处于"思考"状态
2. **动作不连续**：切换动作块时会产生停顿和不连续性

### RTC的优势

- ✅ **消除停顿**：chunk间无缝切换，无需等待
- ✅ **平滑过渡**：使用inpainting技术确保动作连续性
- ✅ **提高精度**：消除训练数据中不存在的停顿，提升任务精度
- ✅ **高延迟鲁棒性**：即使推理延迟达到300ms+仍能保持稳定
- ✅ **无需重新训练**：纯推理时算法，直接应用于已训练模型

### 参考资料

- 论文: [Real-Time Action Chunking with Large Models](https://www.physicalintelligence.company/research/real_time_chunking)
- Physical Intelligence官方研究

## 架构说明

```
┌─────────────────┐     WebSocket      ┌──────────────────────┐
│  客户端工控机    │ ←───────────────→  │   服务器工控机        │
│                 │                    │                      │
│  ROS2机械臂控制  │                    │  ┌────────────────┐  │
│  相机数据采集    │                    │  │  RTC Wrapper   │  │
│  WebsocketClient │                    │  │      ↓         │  │
│                 │                    │  │  Base Policy   │  │
│                 │                    │  └────────────────┘  │
└─────────────────┘                    └──────────────────────┘
```

- **服务器端**: RTC作为policy的包装器，自动处理chunk管理和平滑过渡
- **客户端**: 代码无需修改，按照正常频率请求动作即可

## 使用方法

### 1. 启动带RTC的服务器

```bash
# 基本用法：启用RTC
python scripts/serve_policy.py \
  --policy.config pi05_jaka \
  --policy.dir /path/to/checkpoint \
  --enable-rtc \
  --port 8000

# 自定义RTC参数
python scripts/serve_policy.py \
  --policy.config pi05_jaka \
  --policy.dir /path/to/checkpoint \
  --enable-rtc \
  --rtc-action-horizon 50 \
  --rtc-overlap-steps 10 \
  --rtc-blend-weight 0.7 \
  --rtc-verbose \
  --port 8000
```

### 2. 客户端保持不变

客户端代码**无需任何修改**！继续使用原有的控制循环即可：

```python
# 你的客户端代码保持不变
client = Pi05JakaClient(server_host="192.168.1.88", server_port=8000)
client.run_control_loop(
    task_description="Pick up the bowl",
    max_steps=10000,
    control_hz=10
)
```

## RTC 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--enable-rtc` | False | 是否启用RTC |
| `--rtc-action-horizon` | 50 | 每个chunk的动作数量 |
| `--rtc-overlap-steps` | auto (20%) | 用于平滑过渡的重叠步数 |
| `--rtc-blend-weight` | 0.7 | 重叠区域的混合权重（0-1） |
| `--rtc-verbose` | True | 是否显示详细日志 |

### 参数调优建议

#### action_horizon
- **推荐**: 50 (对应50Hz下1秒，或10Hz下5秒)
- **较大值** (100+): 更长期的规划，但切换不够频繁
- **较小值** (25-): 更频繁的反应，但计算开销增加

#### overlap_steps
- **推荐**: horizon的20-30% (如50→10-15步)
- **较大值**: 更平滑的过渡，但反应稍慢
- **较小值**: 更快的反应，但可能有轻微抖动

#### blend_weight
- **推荐**: 0.6-0.8
- **较大值** (0.8+): 更倾向保持旧轨迹，过渡更平滑
- **较小值** (0.5-): 更快适应新策略，但可能不够平滑

## 对比测试

### 不启用RTC（默认）

```bash
# 启动服务器 - 无RTC
python scripts/serve_policy.py \
  --policy.config pi05_jaka \
  --policy.dir /path/to/checkpoint \
  --port 8000
```

**现象**:
- ❌ chunk切换时有明显停顿
- ❌ 容易出现"重播"现象
- ❌ 对高延迟敏感

### 启用RTC

```bash
# 启动服务器 - 启用RTC
python scripts/serve_policy.py \
  --policy.config pi05_jaka \
  --policy.dir /path/to/checkpoint \
  --enable-rtc \
  --port 8000
```

**现象**:
- ✅ 无停顿，平滑执行
- ✅ 消除重播问题
- ✅ 对延迟鲁棒

## 实际示例

### 示例1: Jaka机械臂（你的场景）

```bash
# 服务器端（GPU工控机）
cd /home/beautycube/jwq/openpi2/openpi

# 启动带RTC的服务
python scripts/serve_policy.py \
  --policy.config pi05_jaka \
  --policy.dir /path/to/your/checkpoint \
  --enable-rtc \
  --rtc-action-horizon 50 \
  --rtc-verbose \
  --port 8000
```

客户端代码保持不变，运行即可看到效果！

### 示例2: ALOHA

```bash
python scripts/serve_policy.py \
  --env aloha \
  --enable-rtc \
  --rtc-action-horizon 25 \
  --port 8000
```

## 监控和调试

### 查看RTC日志

启用 `--rtc-verbose` 后，会看到详细的RTC工作日志：

```
[RTC] 初始化 - action_horizon=50, overlap_steps=10, blend_weight=0.7
[RTC] 生成新chunk #1
[RTC] 进入overlap区域 - 剩余10步，生成并混合下一个chunk
[RTC] 平滑切换到chunk #2
[RTC统计] 总推理=245, 总chunks=5, 成功切换=4, 平均每chunk推理=49.0次
```

### 性能指标

客户端接收的响应中包含RTC统计：

```python
result = policy.infer(obs)
print(result["policy_timing"])
# 输出:
# {
#   "infer_ms": 145.2,
#   "rtc_chunk_id": 3,
#   "rtc_chunk_step": 25,
#   "rtc_total_transitions": 2
# }
```

## 常见问题

### Q: RTC需要重新训练模型吗？
**A**: 不需要！RTC是纯推理时算法，直接应用于已训练的模型。

### Q: 客户端需要修改代码吗？
**A**: 不需要！客户端完全透明，只需在服务器端启用RTC。

### Q: RTC会增加延迟吗？
**A**: 不会。RTC消除了chunk间的停顿，实际上会**降低**总体延迟。

### Q: 如何判断RTC是否生效？
**A**: 
1. 查看服务器日志中的RTC消息
2. 观察机器人动作是否更平滑
3. 检查是否消除了"重播"问题

### Q: 遇到抖动怎么办？
**A**: 尝试调整参数：
- 增加 `rtc-blend-weight` (如0.8)
- 增加 `rtc-overlap-steps` (如horizon的30%)

### Q: 反应太慢怎么办？
**A**: 尝试：
- 减小 `rtc-blend-weight` (如0.6)
- 减小 `rtc-action-horizon` (如25)

## 技术细节

### RTC工作原理

1. **Chunk管理**: 维护当前正在执行的动作chunk
2. **提前生成**: 在接近chunk末尾时提前生成下一个chunk
3. **Inpainting混合**: 
   - 保留旧chunk的剩余部分（即将执行）
   - 对overlap区域进行加权混合
   - 使用新chunk的后续部分
4. **平滑过渡**: 混合权重随位置线性变化，确保平滑

### 混合公式

```python
for i in range(overlap_length):
    alpha = blend_weight * (1.0 - i / overlap_length)
    blended_action[i] = alpha * old_action[i] + (1 - alpha) * new_action[i]
```

## 总结

RTC是一个强大的工具，可以显著提升机器人控制的质量：

- 🚀 **简单**: 一个参数启用，客户端无需修改
- 🎯 **有效**: 消除停顿和不连续性
- 💪 **鲁棒**: 对高延迟保持稳定
- 🔧 **灵活**: 可调参数满足不同需求

立即尝试RTC，让你的机器人控制更加流畅！

