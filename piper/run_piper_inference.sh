#!/bin/bash
# 启动松灵机械臂叠衣服推理客户端
# 
# 用法:
#   bash piper/run_piper_inference.sh
#
# 或者自定义参数:
#   bash piper/run_piper_inference.sh --host 192.168.1.100 --port 8000 --speed 60

set -e

# 默认配置
SERVER_HOST="localhost"
SERVER_PORT=8000
TASK_DESCRIPTION="Fold the cloth"
MAX_STEPS=1000
CONTROL_HZ=10.0
SPEED_PERCENT=50
LEFT_CAN="can_left"
RIGHT_CAN="can_right"
CAMERAS="0,1,2"  # top, left_wrist, right_wrist

# 打印配置
echo "=========================================="
echo "启动松灵机械臂叠衣服推理客户端"
echo "=========================================="
echo "推理服务器: $SERVER_HOST:$SERVER_PORT"
echo "任务描述: $TASK_DESCRIPTION"
echo "控制频率: ${CONTROL_HZ} Hz"
echo "速度百分比: ${SPEED_PERCENT}%"
echo "左臂CAN端口: $LEFT_CAN"
echo "右臂CAN端口: $RIGHT_CAN"
echo "相机索引: $CAMERAS"
echo "=========================================="
echo ""

# 检查 Python 环境
if ! command -v python &> /dev/null; then
    echo "错误: 未找到 Python"
    exit 1
fi

# 构建命令
CMD="python piper/piper_inference_client.py"
CMD="$CMD --host $SERVER_HOST"
CMD="$CMD --port $SERVER_PORT"
CMD="$CMD --task \"$TASK_DESCRIPTION\""
CMD="$CMD --max-steps $MAX_STEPS"
CMD="$CMD --control-hz $CONTROL_HZ"
CMD="$CMD --speed $SPEED_PERCENT"
CMD="$CMD --left-can $LEFT_CAN"
CMD="$CMD --right-can $RIGHT_CAN"
CMD="$CMD --cameras $CAMERAS"

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
echo "按Ctrl+C停止客户端"
echo "=========================================="
echo ""

eval $CMD

