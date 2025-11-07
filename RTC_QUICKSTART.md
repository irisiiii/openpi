# RTC 快速开始 ⚡

## 问题
推理时出现**动作重播**，chunk切换不流畅 ❌

## 解决方案
启用**Real-Time Chunking (RTC)** ✅

---

## 🚀 3步启用RTC

### 步骤1: 编辑启动脚本
```bash
cd /home/beautycube/jwq/openpi2/openpi
vim scripts/serve_pi05_jaka_rtc.sh
```

修改checkpoint路径：
```bash
CHECKPOINT_DIR="/path/to/your/checkpoint"  # 改成你的路径
```

### 步骤2: 启动服务器（带RTC）
```bash
bash scripts/serve_pi05_jaka_rtc.sh
```

### 步骤3: 运行客户端（无需修改！）
```bash
# 在客户端工控机上，正常运行你的代码
python your_client.py
```

---

## 📋 命令对比

### 不启用RTC（旧方式）
```bash
python scripts/serve_policy.py \
  --policy.config pi05_jaka \
  --policy.dir /path/to/checkpoint \
  --port 8000
```

### 启用RTC（新方式）⭐
```bash
python scripts/serve_policy.py \
  --policy.config pi05_jaka \
  --policy.dir /path/to/checkpoint \
  --enable-rtc \
  --port 8000
```

---

## ✅ 验证RTC工作

启动服务器后，应该看到：
```
======================================================================
启用 Real-Time Chunking (RTC)
  - action_horizon: 50
  - overlap_steps: auto
  - blend_weight: 0.7
  - verbose: True
======================================================================
INFO - [RTC] 初始化 - action_horizon=50, overlap_steps=10, blend_weight=0.7
```

运行时应该看到：
```
INFO - [RTC] 生成新chunk #1
INFO - [RTC] 进入overlap区域 - 剩余10步，生成并混合下一个chunk
INFO - [RTC] 平滑切换到chunk #2
```

---

## 🎯 预期效果

| 问题 | 不启用RTC | 启用RTC |
|------|----------|---------|
| 动作重播 | ❌ 经常出现 | ✅ 消除 |
| 停顿 | ❌ chunk间有停顿 | ✅ 无停顿 |
| 流畅度 | ❌ 不流畅 | ✅ 平滑 |
| 精度 | 一般 | ✅ 提高 |

---

## 🔧 遇到问题？

### 动作还是抖动
```bash
# 增加平滑度
bash scripts/serve_pi05_jaka_rtc.sh --rtc-blend-weight 0.8
```

### 反应太慢
```bash
# 减小平滑度
bash scripts/serve_pi05_jaka_rtc.sh --rtc-blend-weight 0.6
```

### 查看详细文档
```bash
cat RTC_GUIDE.md              # 完整使用指南
cat RTC_COMPARISON_TEST.md    # 对比测试
cat RTC_IMPLEMENTATION_SUMMARY.md  # 技术细节
```

---

## 💡 核心要点

1. ✅ **服务器端启用** - 只需修改服务器启动命令
2. ✅ **客户端零改动** - 代码完全不需要修改
3. ✅ **立即见效** - 无需重新训练
4. ✅ **简单易用** - 一个参数开关

---

## 📞 需要帮助？

- 📖 **详细文档**: `RTC_GUIDE.md`
- 🧪 **测试指南**: `RTC_COMPARISON_TEST.md`
- 🔍 **技术细节**: `RTC_IMPLEMENTATION_SUMMARY.md`
- 💻 **代码**: `src/openpi/policies/rtc_policy.py`

---

**现在就试试RTC，让机器人控制更流畅！** 🎉

