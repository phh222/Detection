_base_ = [
    '../configs/_base_/models/yolov3_d53_mstrain-608_273e_coco.py',
    '../configs/_base_/datasets/voc0712.py',
    '../configs/_base_/schedules/schedule_1x.py',
    '../configs/_base_/default_runtime.py'
]

model = dict(
    bbox_head=dict(
        num_classes=20  # VOC数据集中的类别数
    )
)

# 训练相关的配置
total_epochs = 50  # 总的训练轮数
checkpoint_config = dict(interval=10)  # 每隔10个epoch保存一次模型
log_config = dict(interval=10, hooks=[dict(type='TextLoggerHook')])  # 每隔10个epoch记录一次日志