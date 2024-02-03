#!/usr/bin/env bash
# 多机多节点
# CONFIG=$1 
# GPUS=$2
# NNODES=${NNODES:-1} #分布式计算节点总数；如果在环境中定义NNODES则脚本将使用用户定义的值否则将使用默认值1
# NODE_RANK=${NODE_RANK:-0} #当前节点在分布式计算中的排名/编号；使用已经定义的值or默认值0
# PORT=${PORT:-29501} #使用已经定义的值or默认值29501
# MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"} #分布式计算中主节点地址


# export CUDA_VISIBLE_DEVICES=6,7
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# python -m torch.distributed.launch \     #分布式系统启动多个py进程进行并行计算
#    --nnodes=$NNODES \
#    --node_rank=$NODE_RANK \
#    --master_addr=$MASTER_ADDR \
#    --nproc_per_node=$GPUS \            #每个节点GPU数量
#    --master_port=$PORT \
#    $(dirname "$0")/train.py \          #获取当前脚本所在的目录，然后加上 /train.py 构成完整的脚本路径
#    $CONFIG \
#    --seed 0 \
#    --launcher pytorch ${@:3}


###########################################################################
# 无分布式
#不论原来的PYTHONPATH是否有值，都在前添加一个新路径并用：隔开

# CONFIG=$1 
# GPUS=$2

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \    
# python $(dirname "$0")/train.py $CONFIG  --seed 0                                   #随机数种子确保可重复性

# export PYTHONPATH="$(dirname "$0")/..":$PYTHONPATH
# python $(dirname "$0")/train.py $CONFIG --seed 0
###########################################################################



# #!/usr/bin/env bash
# 单机多节点
CONFIG=$1
GPUS=$2
PORT=${PORT:-28500}

export CUDA_VISIBLE_DEVICES=6,7

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} --deterministic
