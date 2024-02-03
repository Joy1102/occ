checkpoint_config = dict(interval=1) #一个训练周期保存一次
# yapf:disable push
# By default we use textlogger hook and tensorboard
# For more loggers see
# https://mmcv.readthedocs.io/en/latest/api.html#mmcv.runner.LoggerHook
log_config = dict(
    interval=50,                  #50个批次保存一次日志
    hooks=[
        dict(type='TextLoggerHook'),  #记录文本日志
        dict(type='TensorboardLoggerHook')  #使用Tensorboard进行可视化日志记录
    ])
# yapf:enable
dist_params = dict(backend='nccl') #指定分布式训练后端
log_level = 'INFO'
work_dir = None
load_from = None
resume_from = None
workflow = [('train', 1)]
