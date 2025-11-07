# 松灵机械臂叠衣服推理客户端使用指南

本文档介绍如何使用训练好的 pi05_fold_cloth 模型在松灵（Piper）双臂机械臂上进行叠衣服任务的实际推理控制。

## 系统架构

```
┌─────────────────────┐         WebSocket         ┌──────────────────────┐
│ 松灵机械臂客户端     │ ◄──────────────────────► │  推理服务器           │
│ piper_inference_    │                            │  (pi05_fold_cloth)   │
│ client.py           │                            │                      │
├─────────────────────┤                            ├──────────────────────┤
│ • 获取相机图像       │                            │ • 加载训练好的模型    │
│ • 获取机械臂状态     │                            │ • 处理观测数据       │
│ • 发送动作指令       │                            │ • 生成动作序列       │
└─────────────────────┘                            └──────────────────────┘
         │
         │ CAN总线 + USB摄像头
         ▼
┌─────────────────────┐
│  松灵双臂机械臂      │
│  + 3个摄像头         │
│  (top, left, right) │
└─────────────────────┘
```

## 前提条件

### 1. 硬件要求

- 松灵（Piper）双臂机械臂
- 3个USB摄像头（顶部视角、左腕、右腕）
- CAN总线接口（连接机械臂）
- 推理服务器（可以是同一台机器或远程机器）

### 2. 软件依赖

```bash
# 安装 openpi-client
pip install -e packages/openpi-client

# 安装松灵机械臂SDK
pip install piper_sdk

# 安装其他依赖
pip install numpy opencv-python
```

### 3. 训练好的模型

确保你已经完成了 pi05_fold_cloth 模型的训练，并且有一个可用的 checkpoint：

```
checkpoints/pi05_fold_cloth/fold_cloth_experiment/19999/
```

## 使用流程

### 步骤 1: 启动推理服务器

在第一个终端中，启动 pi05_fold_cloth 推理服务器：

```bash
# 基础命令
python scripts/serve_policy.py \
  --policy.config pi05_fold_cloth \
  --policy.dir checkpoints/pi05_fold_cloth/fold_cloth_experiment/19999 \
  --port 8000
```

**启用 RTC（推荐）** - 实时动作块平滑，消除停顿：

```bash
python scripts/serve_policy.py \
  --policy.config pi05_fold_cloth \
  --policy.dir checkpoints/pi05_fold_cloth/fold_cloth_experiment/19999 \
  --port 8000 \
  --enable-rtc \
  --rtc-action-horizon 50 \
  --rtc-blend-weight 0.7 \
  --rtc-verbose
```

看到以下输出表示服务器启动成功：

```
Creating server (host: xxx, ip: xxx.xxx.xxx.xxx)
Serving on ws://0.0.0.0:8000
```

### 步骤 2: 连接并启动机械臂控制客户端

在第二个终端中，启动松灵机械臂推理客户端：

**方法 1: 使用启动脚本（推荐）**

```bash
bash piper/run_piper_inference.sh
```

**方法 2: 使用 Python 直接运行**

```bash
python piper/piper_inference_client.py \
  --host localhost \
  --port 8000 \
  --task "Fold the cloth" \
  --max-steps 1000 \
  --control-hz 10.0 \
  --speed 50 \
  --left-can can_left \
  --right-can can_right \
  --cameras 0,1,2
```

### 步骤 3: 监控执行

客户端启动后会：

1. ✅ 连接松灵机械臂并使能
2. ✅ 连接到推理服务器
3. ✅ 初始化3个摄像头
4. ✅ 预热模型（执行2步推理）
5. ✅ 进入控制循环

你会看到类似以下的输出：

```
======================================================================
开始控制循环
任务: Fold the cloth
控制频率: 10.0 Hz
速度百分比: 50%
======================================================================

[步骤 0] 请求动作...
✓ 推理耗时: 235.6ms
  server_total_ms: 232.1ms
  server_preprocess_ms: 12.3ms
  server_inference_ms: 215.8ms
✓ 收到 20 步动作，开始执行...
本轮耗时: 2.34秒 (请求+执行20步)

[步骤 20] 请求动作...
...
```

## 命令行参数说明

### 推理客户端参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host` | localhost | 推理服务器地址 |
| `--port` | 8000 | 推理服务器端口 |
| `--task` | "Fold the cloth" | 任务描述（prompt） |
| `--max-steps` | 1000 | 最大执行步数 |
| `--control-hz` | 10.0 | 控制频率（Hz） |
| `--speed` | 50 | 机械臂速度百分比（1-100） |
| `--left-can` | can_left | 左臂CAN端口名 |
| `--right-can` | can_right | 右臂CAN端口名 |
| `--cameras` | 0,1,2 | 相机索引（top, left_wrist, right_wrist） |

### 推理服务器参数（RTC相关）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--enable-rtc` | False | 启用实时动作块平滑 |
| `--rtc-action-horizon` | 50 | 每个动作块的步数 |
| `--rtc-overlap-steps` | None (自动) | 动作块重叠步数 |
| `--rtc-blend-weight` | 0.7 | 动作混合权重（0-1） |
| `--rtc-verbose` | True | 显示详细日志 |

## 数据格式说明

### 观测数据（客户端 → 服务器）

```python
observation = {
    'observation/state': np.ndarray,  # shape (14,) 
                                      # [left_joints(6), left_gripper(1), 
                                      #  right_joints(6), right_gripper(1)]
    'observation/images/top': np.ndarray,  # shape (3, H, W) RGB
    'observation/images/wrist_left': np.ndarray,  # shape (3, H, W) RGB
    'observation/images/wrist_right': np.ndarray,  # shape (3, H, W) RGB
    'prompt': str  # 任务描述
}
```

### 动作数据（服务器 → 客户端）

```python
action_result = {
    'actions': np.ndarray,  # shape (N, 14)
                           # N = action_horizon (例如 20)
                           # 14 = [left_joints(6), left_gripper(1),
                           #       right_joints(6), right_gripper(1)]
    'server_timing': {
        'total_ms': float,
        'preprocess_ms': float,
        'inference_ms': float,
        ...
    }
}
```

## 常见问题

### 1. 相机无法打开

**问题**: `⚠️ xxx 相机 (索引 x) 打开失败`

**解决方案**:
```bash
# 检查可用的相机设备
ls /dev/video*

# 测试相机
python -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"

# 修改相机索引
python piper/piper_inference_client.py --cameras 2,4,6
```

### 2. 机械臂连接失败

**问题**: `机械臂连接失败`

**解决方案**:
```bash
# 检查CAN设备
ifconfig | grep can

# 检查CAN端口名称
ls /dev/ | grep can

# 启动CAN接口（如果未启动）
sudo ip link set can0 up type can bitrate 1000000
sudo ip link set can1 up type can bitrate 1000000
```

### 3. 推理服务器连接失败

**问题**: 无法连接到推理服务器

**解决方案**:
```bash
# 检查服务器是否运行
netstat -tuln | grep 8000

# 如果是远程服务器，检查防火墙
# 使用正确的IP地址
python piper/piper_inference_client.py --host 192.168.1.100
```

### 4. 动作执行不流畅

**问题**: 机械臂动作有停顿

**解决方案**:
- 启用 RTC（Real-Time Chunking）
- 降低控制频率 `--control-hz 5.0`
- 增加机械臂速度 `--speed 80`

### 5. 模型推理太慢

**问题**: 推理耗时 > 1000ms

**解决方案**:
- 使用GPU加速（确保CUDA可用）
- 降低模型action_horizon
- 考虑使用更快的硬件

## 安全注意事项

⚠️ **重要安全提示**:

1. **首次运行时使用低速**: `--speed 30`
2. **保持急停按钮可触及**
3. **确保工作空间内无人员**
4. **监控机械臂运动，发现异常立即停止**
5. **测试环境中先验证，再实际部署**

## 调试技巧

### 查看详细日志

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 记录推理时间

```bash
# 服务器端启用记录
python scripts/serve_policy.py --record ...

# 生成的记录保存在 policy_records/ 目录
```

### 可视化相机画面

在 `piper_inference_client.py` 中添加：

```python
cv2.imshow('Top Camera', top_image)
cv2.imshow('Left Wrist', left_wrist_image)
cv2.imshow('Right Wrist', right_wrist_image)
cv2.waitKey(1)
```

## 性能优化

### 推荐配置（实时控制）

```bash
# 服务器
python scripts/serve_policy.py \
  --policy.config pi05_fold_cloth \
  --policy.dir checkpoints/pi05_fold_cloth/fold_cloth_experiment/19999 \
  --enable-rtc \
  --rtc-action-horizon 50 \
  --rtc-blend-weight 0.7

# 客户端
python piper/piper_inference_client.py \
  --control-hz 10.0 \
  --speed 60
```

### 高精度配置（慢速精细控制）

```bash
# 服务器
python scripts/serve_policy.py \
  --policy.config pi05_fold_cloth \
  --policy.dir checkpoints/pi05_fold_cloth/fold_cloth_experiment/19999

# 客户端
python piper/piper_inference_client.py \
  --control-hz 5.0 \
  --speed 30
```

## 技术支持

如有问题，请检查：

1. 日志输出中的错误信息
2. 机械臂和相机的硬件连接
3. 推理服务器的运行状态
4. 网络连接（如果是远程服务器）

## 相关文档

- [训练指南](../JAKA_TRAINING_GUIDE.md)
- [RTC实时平滑技术](../RTC_GUIDE.md)
- [远程推理部署](../docs/remote_inference.md)
- [松灵机械臂控制SDK](piper_ctl.py)

