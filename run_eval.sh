cd $(readlink -f `dirname $0`) #将工作目录改为脚本所在目录
#conda activate OpenOccupancy

echo $1 #打印第一个命令行参数
if [ -f $1 ]; then
  config=$1 #路径赋值给变量config
else
  echo "need a config file"
  exit
fi

export PYTHONPATH="." #将当前目录添加到Python的搜索路径中，以便Python可以找到脚本中引用的模块

ckpt=$2
gpu=$3
bash tools/dist_test.sh $config $ckpt $gpu ${@:4}