#!/bin/bash
# 启动Pi05 Jaka策略服务器 - 启用RTC
# 
# 用法:
#   bash scripts/serve_pi05_jaka_rtc.sh
#
# 或者自定义参数:
#   bash scripts/serve_pi05_jaka_rtc.sh --rtc-action-horizon 25 --rtc-blend-weight 0.8

set -e

# 默认配置
CONFIG_NAME="pi05_jaka"
CHECKPOINT_DIR=""  # 请设置你的checkpoint路径
PORT=8000

# RTC参数
ENABLE_RTC=true
RTC_HORIZON=50
RTC_OVERLAP=""  # 留空表示自动
RTC_BLEND=0.7
RTC_VERBOSE=true

# 打印配置
echo "=========================================="
echo "启动 Pi05 Jaka 策略服务器（带RTC）"
echo "=========================================="
echo "配置: $CONFIG_NAME"
echo "端口: $PORT"
echo ""
echo "RTC配置:"
echo "  - 启用: $ENABLE_RTC"
echo "  - Action Horizon: $RTC_HORIZON"
echo "  - Overlap Steps: ${RTC_OVERLAP:-auto}"
echo "  - Blend Weight: $RTC_BLEND"
echo "  - Verbose: $RTC_VERBOSE"
echo "=========================================="
echo ""

# 检查checkpoint路径
if [ -z "$CHECKPOINT_DIR" ]; then
    echo "警告: CHECKPOINT_DIR未设置！"
    echo "请编辑此脚本并设置CHECKPOINT_DIR变量"
    echo ""
    echo "示例:"
    echo "  CHECKPOINT_DIR=\"/path/to/your/checkpoint\""
    echo ""
    read -p "输入checkpoint路径（或按Ctrl+C退出）: " CHECKPOINT_DIR
    if [ -z "$CHECKPOINT_DIR" ]; then
        echo "错误: 必须提供checkpoint路径"
        exit 1
    fi
fi

# 构建命令
CMD="python scripts/serve_policy.py"
CMD="$CMD --policy.config $CONFIG_NAME"
CMD="$CMD --policy.dir $CHECKPOINT_DIR"
CMD="$CMD --port $PORT"

# 添加RTC参数
if [ "$ENABLE_RTC" = true ]; then
    CMD="$CMD --enable-rtc"
    CMD="$CMD --rtc-action-horizon $RTC_HORIZON"
    
    if [ -n "$RTC_OVERLAP" ]; then
        CMD="$CMD --rtc-overlap-steps $RTC_OVERLAP"
    fi
    
    CMD="$CMD --rtc-blend-weight $RTC_BLEND"
    
    if [ "$RTC_VERBOSE" = true ]; then
        CMD="$CMD --rtc-verbose"
    fi
fi

# 添加用户提供的额外参数
if [ $# -gt 0 ]; then
    CMD="$CMD $@"
    echo "额外参数: $@"
    echo ""
fi

# 执行
echo "执行命令:"
echo "$CMD"
echo ""
echo "按Ctrl+C停止服务器"
echo "=========================================="
echo ""

exec $CMD

