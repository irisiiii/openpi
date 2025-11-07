# 松灵机械臂叠衣服推理 - 快速开始

这是一个简化的快速开始指南，帮助你在5分钟内启动松灵机械臂的叠衣服推理系统。

## 📋 前置检查清单

- [ ] 松灵双臂机械臂已连接并通电
- [ ] 3个USB摄像头已连接（顶部、左腕、右腕）
- [ ] 已完成 pi05_fold_cloth 模型训练（checkpoint 在 `checkpoints/pi05_fold_cloth/fold_cloth_experiment/19999`）
- [ ] 已安装必要的Python依赖

## 🚀 三步启动

### 第 1 步: 测试相机连接

```bash
# 列出所有可用相机
python piper/test_cameras.py --mode list

# 实时查看相机画面（确认相机位置正确）
python piper/test_cameras.py --mode view --cameras 0,1,2
```

**预期结果**: 看到3个相机的实时画面，分别显示顶部视角、左腕视角、右腕视角。

### 第 2 步: 启动推理服务器

打开**终端1**，启动推理服务器：

```bash
# 推荐配置（启用RTC平滑）
python scripts/serve_policy.py \
  --policy.config pi05_fold_cloth \
  --policy.dir checkpoints/pi05_fold_cloth/fold_cloth_experiment/19999 \
  --port 8000 \
  --enable-rtc \
  --rtc-action-horizon 50 \
  --rtc-blend-weight 0.7
```

**预期输出**:
```
✓ Policy loaded successfully
Creating server (host: xxx, ip: xxx.xxx.xxx.xxx)
Serving on ws://0.0.0.0:8000
```

### 第 3 步: 启动机械臂控制客户端

打开**终端2**，启动松灵机械臂客户端：

```bash
# 使用启动脚本（最简单）
bash piper/run_piper_inference.sh

# 或者直接使用Python（可自定义参数）
python piper/piper_inference_client.py \
  --host localhost \
  --port 8000 \
  --task "Fold the cloth" \
  --control-hz 10.0 \
  --speed 50 \
  --cameras 0,1,2
```

**预期输出**:
```
✓ 松灵机械臂就绪
✓ 服务器元数据: {...}
✓ top 相机已就绪
✓ left_wrist 相机已就绪
✓ right_wrist 相机已就绪
======================================================================
开始控制循环
任务: Fold the cloth
控制频率: 10.0 Hz
======================================================================
```

## 🎮 控制说明

- **启动后**: 机械臂会自动开始执行叠衣服任务
- **停止**: 按 `Ctrl+C` 停止客户端
- **急停**: 使用机械臂硬件急停按钮

## 📊 参数调优

### 如果动作太快/不稳定

```bash
python piper/piper_inference_client.py \
  --speed 30 \              # 降低速度
  --control-hz 5.0          # 降低控制频率
```

### 如果动作太慢

```bash
python piper/piper_inference_client.py \
  --speed 80 \              # 提高速度
  --control-hz 15.0         # 提高控制频率
```

### 如果推理服务器在远程机器

```bash
python piper/piper_inference_client.py \
  --host 192.168.1.100 \    # 远程服务器IP
  --port 8000
```

## ⚠️ 安全提示

1. **首次运行使用低速**: `--speed 30`
2. **保持急停按钮可触及**
3. **确保工作区域安全**
4. **监控机械臂运动**

## 🔧 常见问题快速解决

### 相机无法打开

```bash
# 列出所有相机
python piper/test_cameras.py --mode list

# 使用正确的相机索引
python piper/piper_inference_client.py --cameras 2,4,6
```

### 机械臂连接失败

```bash
# 检查CAN设备
ifconfig | grep can

# 如果看不到 can0, can1，需要先启动CAN接口
sudo ip link set can0 up type can bitrate 1000000
sudo ip link set can1 up type can bitrate 1000000
```

### 推理速度太慢

- 确保使用GPU（检查 `nvidia-smi`）
- 考虑使用更小的 action_horizon
- 确保服务器有足够的计算资源

### 无法连接到服务器

```bash
# 检查服务器是否运行
netstat -tuln | grep 8000

# 检查防火墙
sudo ufw allow 8000/tcp
```

## 📚 下一步

- 详细使用说明: [README_inference.md](README_inference.md)
- RTC平滑技术: [../RTC_GUIDE.md](../RTC_GUIDE.md)
- 训练自己的模型: [../JAKA_TRAINING_GUIDE.md](../JAKA_TRAINING_GUIDE.md)

## 🆘 获取帮助

查看详细日志:
```bash
python piper/piper_inference_client.py --help
```

## 📝 完整命令参考

### 推理服务器（终端1）

```bash
# 最小配置
python scripts/serve_policy.py \
  --policy.config pi05_fold_cloth \
  --policy.dir checkpoints/pi05_fold_cloth/fold_cloth_experiment/19999

# 推荐配置（RTC平滑）
python scripts/serve_policy.py \
  --policy.config pi05_fold_cloth \
  --policy.dir checkpoints/pi05_fold_cloth/fold_cloth_experiment/19999 \
  --enable-rtc --rtc-action-horizon 50 --rtc-blend-weight 0.7

# 记录推理数据（用于调试）
python scripts/serve_policy.py \
  --policy.config pi05_fold_cloth \
  --policy.dir checkpoints/pi05_fold_cloth/fold_cloth_experiment/19999 \
  --record
```

### 机械臂客户端（终端2）

```bash
# 使用脚本启动
bash piper/run_piper_inference.sh

# Python直接启动（基础）
python piper/piper_inference_client.py

# Python启动（自定义参数）
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

# 远程服务器
python piper/piper_inference_client.py \
  --host 192.168.1.100 \
  --port 8000

# 低速安全模式（首次运行推荐）
python piper/piper_inference_client.py \
  --speed 30 \
  --control-hz 5.0
```

## ✅ 成功标志

当看到以下输出时，表示系统正常运行：

```
[步骤 X] 请求动作...
✓ 推理耗时: 200-300ms
✓ 收到 20 步动作，开始执行...
本轮耗时: 2-3秒
```

机械臂应该：
- 平滑地执行动作
- 没有突然的停顿或抖动
- 响应推理服务器的指令

如果一切正常，恭喜你成功运行了松灵机械臂的叠衣服推理系统！🎉

