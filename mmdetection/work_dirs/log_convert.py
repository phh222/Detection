
import os
import json
import os
from tensorboardX import SummaryWriter
log_file = 'faster-rcnn_r50_fpn_1x_voc0712/20240530_155255/vis_data/20240530_155255.json'
log_dir = 'faster-rcnn_r50_fpn_1x_voc0712/20240530_155255/tensorboard_logs'

# 将mmdetection 的log 转换成tensorboard支持的log

# 创建 TensorBoard SummaryWriter
writer = SummaryWriter(log_dir=log_dir)

# 读取日志文件
with open(log_file, 'r') as f:
    for line in f:
        log = json.loads(line.strip())
        
        # 记录训练过程中的损失函数和学习率
        if 'loss' in log:
            writer.add_scalar('train/loss', log['loss'], log['iter'])
            writer.add_scalar('train/loss_rpn_cls', log['loss_rpn_cls'], log['iter'])
            writer.add_scalar('train/loss_rpn_bbox', log['loss_rpn_bbox'], log['iter'])
            writer.add_scalar('train/loss_cls', log['loss_cls'], log['iter'])
            writer.add_scalar('train/loss_bbox', log['loss_bbox'], log['iter'])
            writer.add_scalar('train/lr', log['lr'], log['iter'])
            writer.add_scalar('acc', log['acc'], log['iter'])
        
        # 记录验证集上的 mAP
        if 'pascal_voc/mAP' in log:
            writer.add_scalar('val/mAP', log['pascal_voc/mAP'], log['step'])
            writer.add_scalar('val/AP50', log['pascal_voc/AP50'], log['step'])

writer.close()
# log_file = 'yolov3_d53_mstrain-608_100e_voc0712/20240527_211551/vis_data/20240527_211551.json'
# log_dir = 'yolov3_d53_mstrain-608_100e_voc0712/20240527_211551/tensorboard_logs'
# # 创建 TensorBoard SummaryWriter
# writer = SummaryWriter(log_dir=log_dir)

# # 读取日志文件
# with open(log_file, 'r') as f:
#     for line in f:
#         log = json.loads(line.strip())
        
#         # 记录训练过程中的损失函数和学习率
#         if 'loss' in log:
#             writer.add_scalar('train/loss', log['loss'], log['iter'])
#             writer.add_scalar('train/loss_conf', log['loss_conf'], log['iter'])
#             writer.add_scalar('train/loss_xy', log['loss_xy'], log['iter'])
#             writer.add_scalar('train/loss_cls', log['loss_cls'], log['iter'])
#             writer.add_scalar('train/loss_wh', log['loss_wh'], log['iter'])
#             writer.add_scalar('train/lr', log['lr'], log['iter'])
#             writer.add_scalar('grad_norm', log['grad_norm'], log['iter'])
        
#         # 记录验证集上的 mAP
#         if 'pascal_voc/mAP' in log:
#             writer.add_scalar('val/mAP', log['pascal_voc/mAP'], log['step'])
#             writer.add_scalar('val/AP50', log['pascal_voc/AP50'], log['step'])

# writer.close()