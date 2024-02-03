#!/usr/bin/env bash
# export CUDA_VISIBLE_DEVICES= 0,1,2,3 #指定GPU
CONFIG=$1 #将脚本的第一个命令行参数（通常是一个配置文件的路径）分配给名为 CONFIG 的变量。
CHECKPOINT=$2 #将脚本的第二个命令行参数（通常是一个模型checkpoint文件的路径）分配给名为 CHECKPOINT 的变量。
GPUS=$3 #?GPU数量
PORT=${PORT:-20004}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4} --deterministic --eval bbox
    #代码运行 test.py 脚本，传递了配置文件 ($CONFIG) 和模型检查点 ($CHECKPOINT) 作为参数
    #${@:4}：将脚本的命令行参数从第四个参数开始传递给 test.py 脚本
