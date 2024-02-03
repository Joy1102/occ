#bash run.sh ./projects/baselines/LiDAR_128x128x10.py 8
cd $(readlink -f `dirname $0`) #获取脚本文件本身的路径-转换为绝对路径
export PYTHONPATH="." #将当前脚本所在的目录添加到Python的模块搜索路径中

echo $1 #打印第一个参数 ./projects/baselines/LiDAR_128x128x10.py
if [ -f $1 ]; then #如$1指定文件存在，则将其设为config文件
  config=$1
else
  echo "need a config file"
  exit
fi

bash tools/dist_train.sh $config $2 ${@:3} # ${@:3}所有的额外参数

