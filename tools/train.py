# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
 
from __future__ import division

import argparse
import copy
import mmcv
import os
import time
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from os import path as osp

from mmdet import __version__ as mmdet_version
from mmdet3d import __version__ as mmdet3d_version
from mmseg import __version__ as mmseg_version
#from mmdet3d.apis import train_model

from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import collect_env, get_root_logger
from mmdet.apis import set_random_seed

from mmcv.utils import TORCH_VERSION, digit_version
from projects.occ_plugin.occupancy.apis.train import custom_train_model


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector') #创建名为parser的ArgumentParser对象
    parser.add_argument('config', help='train config file path') #位置参数
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training') #带有特定动作的可选参数

    group_gpus = parser.add_mutually_exclusive_group() #创建互斥参数组，只可使用其中一个参数指定gpu（给定数量/id）
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')

    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',  
        nargs='+',   #可接受一个/多个值
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],   #pytorch：DDP
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(       #自动根据gpu数量调整学习率
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin') and cfg.plugin:                          #配置文件是否用插件
        assert cfg.plugin_dir is not None                              #断言，插件目录存在

        import importlib                                               #将插件目录的路径转换为Python模块路径（/到.）
        plugin_dir = cfg.plugin_dir 
        _module_dir = os.path.dirname(plugin_dir) 
        _module_dir = _module_dir.split('/') 
        _module_path = _module_dir[0] 

        for m in _module_dir[1:]:
            _module_path = _module_path + '.' + m
        print(_module_path)
        plg_lib = importlib.import_module(_module_path)                 #动态导入，根据配置动态加载指定Python模块
    
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):                               #为True则执行后续（存在返回值，不存在返回False）
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: 命令行参数CLI > 配置文件设置segment in file > 默认文件名filename
    if args.work_dir is not None:                                      #args.work_dir是否指定工作目录
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:                            #配置文件cfg存在work_dir
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0]) #确定work_dir值，名字同配置文件名
        
    # if args.resume_from is not None:                                 #从之前的中断继续训练
    if args.resume_from is not None and osp.isfile(args.resume_from):
        cfg.resume_from = args.resume_from
        
    if args.gpu_ids is not None:                                      #指定gpu_id
        cfg.gpu_ids = args.gpu_ids
    else:                                                             #指定gpu数量
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    if args.autoscale_lr:                                             #缩放学习率--autoscale-lr
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False                                          #不使用分布式
    else:
        distributed = True                                           #使用分布式，pytorch slurm mpi
        init_dist(args.launcher, **cfg.dist_params)
        # re-set gpu_ids with distributed training mode
        _, world_size = get_dist_info()                              #参与计算的总进程数
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))                   #创建work_dir
    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))      #保存config到work_dir
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')            #日志文件路径
    logger = get_root_logger(                                        #初始化日志记录器
        log_file=log_file, log_level=cfg.log_level, name='mmdet')


    # init the meta dict to record some important information such as environment info and seed, which will be logged
    meta = dict()                                                   #初始化元数据字典meta
    # log env info                                                  #收集并记录环境信息
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text                                #记录配置信息

    # log some basic info
    logger.info(f'Distributed training: {distributed}')             #是否正在进行分布式训练

    # set random seeds
    if args.seed is not None:                                       #设置并记录随机数种子
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)                   #将配置文件名保存为实验名称


    model = build_model(                                           # 构建模型
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad) #模型参数数量（只计算可训练（参与梯度下降）的参数）
    logger.info(f'Number of params: {n_parameters}')
    model.init_weights()                                          #初始化模型权重
    
    datasets = [build_dataset(cfg.data.train)]                    #创建数据集/projects/configs/_base_/datasets/custom_nus-3d.py  


    # add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES                           #类别？
    
    custom_train_model(                                           #启动训练过程 projects.occ_plugin.occupancy.apis.train.py    
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)


if __name__ == '__main__':
    main()
