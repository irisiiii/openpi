# Jaka机器人训练指南

本指南将帮助你使用OpenPI训练Jaka机器人的策略模型。

## 数据集信息

你的数据集已经转换为LeRobot格式，位于 `data/jaka_bowel_lerobot/`，包含：

- **机器人类型**: piper_dual_arm (双臂机器人)
- **观测维度**:
  - `observation/wrist_image_left`: 左腕相机图像 (480×640×3)
  - `observation/top`: 顶部相机图像 (480×640×3)
  - `observation/state`: 机器人状态 (8维: 7个关节 + 1个夹爪位置)
- **动作维度**: `action` (8维: 7个关节 + 1个夹爪位置)
- **数据集统计**:
  - 总episodes: 48
  - 总帧数: 24,000
  - FPS: 20

## 训练步骤

### 第1步：计算归一化统计

在训练之前，需要计算数据集的归一化统计信息。这对于模型训练的稳定性很重要。

根据你要使用的模型，运行以下命令之一：

#### Pi0 模型：
```bash
uv run scripts/compute_norm_stats.py --config-name pi0_jaka
```

#### Pi0.5 模型（推荐，性能更好）：
```bash
uv run scripts/compute_norm_stats.py --config-name pi05_jaka
```

#### Pi0-FAST 模型（自回归模型）：
```bash
uv run scripts/compute_norm_stats.py --config-name pi0_fast_jaka
```

这个命令会：
- 读取你的LeRobot数据集
- 计算state和action的统计信息（均值、标准差、分位数等）
- 将结果保存到 `assets/jaka/norm_stats.json`

**注意**: 如果你的数据集包含某些很少使用的维度，可能会导致极小的 `q01`, `q99`, 或 `std` 值。如果训练时出现loss发散，请检查 `norm_stats.json` 并手动调整这些值。

### 第2步：开始训练

计算完归一化统计后，就可以开始训练了。

#### 基础训练命令（Pi0模型）：
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_jaka --exp-name=my_jaka_experiment --overwrite
```

#### Pi0.5模型训练（推荐）：
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_jaka --exp-name=my_jaka_experiment --overwrite
```

#### Pi0-FAST模型训练：
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_fast_jaka --exp-name=my_jaka_experiment --overwrite
```

**参数说明**:
- `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9`: 允许JAX使用最多90%的GPU内存（默认75%）
- `--exp-name`: 实验名称，用于区分不同的训练运行
- `--overwrite`: 如果同名实验已存在，覆盖之前的检查点

训练过程中：
- 训练日志会输出到控制台
- 检查点保存到 `checkpoints/pi0_jaka/my_jaka_experiment/` (或对应的配置名称)
- 每1000步保存一次检查点
- 可以在Weights & Biases查看训练进度（如果配置了的话）

### 第3步：监控训练

训练过程中可以观察：
- **Loss**: 应该逐渐下降
- **GPU内存使用**: 应该在合理范围内
- 如果GPU内存不足，可以：
  - 减小batch_size
  - 使用FSDP（全分片数据并行）：添加 `--fsdp-devices <GPU数量>`

### 第4步：运行推理

训练完成后，可以启动策略服务器进行推理：

```bash
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_jaka \
    --policy.dir=checkpoints/pi05_jaka/my_jaka_experiment/20000
```

这会启动一个监听8000端口的服务器，等待接收观测数据。

**推理示例代码**：
```python
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download

# 加载配置和检查点
config = _config.get_config("pi05_jaka")
checkpoint_dir = "checkpoints/pi05_jaka/my_jaka_experiment/20000"

# 创建训练好的策略
policy = policy_config.create_trained_policy(config, checkpoint_dir)

# 运行推理
example = {
    "observation/wrist_image_left": ...,  # 左腕相机图像 (480, 640, 3) uint8
    "observation/top": ...,                # 顶部相机图像 (480, 640, 3) uint8
    "observation/state": ...,              # 机器人状态 (8,) float32
    "prompt": "pick up the object"         # 任务指令
}
action_chunk = policy.infer(example)["actions"]
```

## 配置详解

已经为你创建了三个训练配置，位于 `src/openpi/training/config.py`:

### 1. `pi0_jaka` - Pi0模型
- **模型**: Pi0 (flow-based VLA)
- **Action horizon**: 10
- **Batch size**: 32
- **训练步数**: 20,000
- **基础模型**: pi0_base

### 2. `pi05_jaka` - Pi0.5模型（推荐）
- **模型**: Pi0.5 (改进的Pi0，泛化能力更好)
- **Action horizon**: 10
- **Batch size**: 64
- **训练步数**: 20,000
- **学习率调度**: Cosine decay
- **基础模型**: pi05_base

### 3. `pi0_fast_jaka` - Pi0-FAST模型
- **模型**: Pi0-FAST (自回归VLA)
- **Action dimension**: 8
- **Action horizon**: 10
- **Max token length**: 180
- **Batch size**: 32
- **训练步数**: 20,000
- **基础模型**: pi0_fast_base

## 自定义配置

如果需要修改配置，编辑 `src/openpi/training/config.py` 中的相应配置：

### 修改数据集路径
```python
repo_id="data/jaka_bowel_lerobot",  # 改为你的数据集路径
```

### 调整是否使用delta actions
```python
use_delta_joint_actions=True,  # True: 数据集包含绝对位置
                                # False: 数据集已经是delta值
```

### 添加默认提示词
```python
default_prompt="pick up the object",  # 如果数据集没有prompt，使用默认值
```

### 调整训练超参数
```python
batch_size=32,           # 根据GPU内存调整
num_train_steps=20_000,  # 训练总步数
save_interval=1000,      # 保存检查点的间隔
```

## 数据格式说明

你的数据已经符合要求，但为了理解数据映射，这里说明一下：

### 训练时的数据流：
1. **从数据集读取** → 原始LeRobot格式
2. **Repack transform** → 重命名键
3. **Data transform (JakaInputs)** → 转换为模型输入格式
4. **Delta action transform** (如果启用) → 转换为delta actions
5. **Model transform** → 分词、归一化等
6. **输入模型进行训练**

### 推理时的数据流：
1. **从机器人获取观测** → 你的格式
2. **Data transform (JakaInputs)** → 转换为模型输入格式
3. **输入模型** → 获得动作输出
4. **Data transform (JakaOutputs)** → 转换回你的格式
5. **发送到机器人执行**

## 故障排除

### 1. GPU内存不足
```bash
# 使用更多GPU内存
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run scripts/train.py pi05_jaka ...

# 或使用FSDP（如果有多GPU）
uv run scripts/train.py pi05_jaka ... --fsdp-devices 2
```

### 2. 训练loss发散
- 检查 `assets/jaka/norm_stats.json`
- 查看 `q01`, `q99`, `std` 值是否有异常小的值
- 手动调整这些统计值

### 3. 数据加载慢
- 增加 `num_workers` (在config中)
- 确保数据在SSD上

### 4. 找不到norm_stats
- 确保运行了 `compute_norm_stats.py`
- 检查 `assets/jaka/` 目录是否存在

## 下一步

1. **验证模型**: 在验证集上评估模型性能
2. **调优超参数**: 根据结果调整学习率、batch size等
3. **部署到机器人**: 将训练好的模型部署到实际机器人上
4. **远程推理**: 参考 `docs/remote_inference.md` 设置远程推理服务

## 相关文件

- **Policy定义**: `src/openpi/policies/jaka_policy.py`
- **训练配置**: `src/openpi/training/config.py` (搜索 "Jaka")
- **计算norm stats脚本**: `scripts/compute_norm_stats.py`
- **训练脚本**: `scripts/train.py`
- **推理服务脚本**: `scripts/serve_policy.py`

## 参考资料

- [README.md](README.md) - 总体介绍
- [LIBERO示例](examples/libero/README.md) - 类似的训练示例
- [远程推理文档](docs/remote_inference.md) - 部署指南
- [归一化统计文档](docs/norm_stats.md) - norm stats详解

祝训练顺利！🚀

