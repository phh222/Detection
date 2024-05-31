# import os
# import torch
# from mmengine.config import Config
# from mmengine.visualization import Visualizer
# from mmdet.apis import init_detector, inference_detector
# from mmdet.utils import register_all_modules
# import mmcv
# import numpy as np

# def load_checkpoint_with_fix(path, map_location='cpu'):
#     checkpoint = torch.load(path, map_location=map_location)
#     if 'meta' in checkpoint and isinstance(checkpoint['meta'].get('dataset_meta'), list):
#         # Convert list to dict
#         checkpoint['meta']['dataset_meta'] = {str(i): v for i, v in enumerate(checkpoint['meta']['dataset_meta'])}
#     return checkpoint

# def init_yolov3_with_fixed_checkpoint(config_file, checkpoint_file, device='cuda:0'):
#     # 加载并修复 checkpoint_meta
#     checkpoint = load_checkpoint_with_fix(checkpoint_file, map_location=device)
    
#     # 初始化模型（不加载检查点）
#     model = init_detector(config_file, None, device=device)
    
#     # 加载检查点到模型
#     model.load_state_dict(checkpoint['state_dict'])
    
#     return model

# def run_inference(config_file, checkpoint_file, image_file, device='cuda:0', score_thr=0.3, out_file=None):
#     # 初始化并加载模型
#     model = init_yolov3_with_fixed_checkpoint(config_file, checkpoint_file, device)
    
#     # 进行推理
#     result = inference_detector(model, image_file)
    
#     # 可视化结果
#     visualize_result(image_file, model, result, out_file, score_thr)

# def visualize_result(image_path, model, result, out_file=None, score_thr=0.3):
#     # 读取图像
#     image = mmcv.imread(image_path)
    
#     # 创建 Visualizer 实例
#     visualizer = Visualizer(image=image,vis_backends=[dict(type='LocalVisBackend')],save_dir=out_file)
    
#     # 设置数据集元信息
#     visualizer.dataset_meta = model.dataset_meta

#     # 过滤置信度低的检测结果
#     scores = result.pred_instances.scores
#     high_score_idxs = scores > score_thr
#     bboxes = result.pred_instances.bboxes[high_score_idxs]
#     scores = scores[high_score_idxs]
#     labels = result.pred_instances.labels[high_score_idxs]

#     # 绘制检测结果
#     visualizer.draw_bboxes(bboxes, edge_colors='red')
#     for bbox, score, label in zip(bboxes, scores, labels):
#         bbox_np = bbox.cpu().numpy() if isinstance(bbox, torch.Tensor) else bbox
#         position = np.array([bbox_np[0], bbox_np[1]])
#         visualizer.draw_texts(f'{model.dataset_meta["classes"][label]} {score:.2f}', position, colors='red')


#     # 显示或保存结果
    
#     visualizer.add_image("68", visualizer.get_image())

# if __name__ == "__main__":
#     # 配置文件路径
#     config_file = 'configs/pascal_voc/yolov3_d53_mstrain-608_100e_voc0712.py'
#     # 检查点文件路径
#     checkpoint_file = 'work_dirs/yolov3_d53_mstrain-608_100e_voc0712/epoch_100.pth'
#     # 测试图片路径
#     image_file = 'data/VOCdevkit/VOC2007/JPEGImages/000333.jpg'
#     # 设备
#     device = 'cuda:0'

#     run_inference(config_file, checkpoint_file, image_file, device,out_file="outputs/vis")
import torch
from mmengine.config import Config
from mmdet.apis import init_detector
from mmdet.utils import register_all_modules

def load_checkpoint_with_fix(path, map_location='cpu'):
    checkpoint = torch.load(path, map_location=map_location)
    if 'meta' in checkpoint and isinstance(checkpoint['meta'].get('dataset_meta'), list):
        # Convert list to dict
        checkpoint['meta']['dataset_meta'] = {str(i): v for i, v in enumerate(checkpoint['meta']['dataset_meta'])}
    return checkpoint

def init_yolov3_with_fixed_checkpoint(config_file, checkpoint_file, device='cuda:0'):
    # 加载并修复 checkpoint_meta
    checkpoint = load_checkpoint_with_fix(checkpoint_file, map_location=device)
    
    # 初始化模型（不加载检查点）
    model = init_detector(config_file, None, device=device)
    
    # 加载检查点到模型
    model.load_state_dict(checkpoint['state_dict'])
    
    return model, checkpoint

def save_fixed_checkpoint(model, checkpoint, new_checkpoint_file):
    # 创建新的检查点
    new_checkpoint = {
        'meta': checkpoint['meta'],
        'state_dict': model.state_dict(),
        'optimizer': checkpoint.get('optimizer', None),
        'param_schedulers': checkpoint.get('param_schedulers', None),
        'message_hub': checkpoint.get('message_hub', None)
    }
    # 保存新的检查点文件
    torch.save(new_checkpoint, new_checkpoint_file)

if __name__ == "__main__":
    # 配置文件路径
    config_file = 'configs/pascal_voc/yolov3_d53_mstrain-608_100e_voc0712.py'
    # 原始检查点文件路径
    checkpoint_file = 'work_dirs/yolov3_d53_mstrain-608_100e_voc0712/epoch_100.pth'
    # 修复后的检查点文件路径
    new_checkpoint_file = 'work_dirs/yolov3_d53_mstrain-608_100e_voc0712/epoch_100_fixed.pth'
    # 设备
    device = 'cuda:0'

    # 初始化并修复模型和检查点
    model, checkpoint = init_yolov3_with_fixed_checkpoint(config_file, checkpoint_file, device)
    
    # 保存修复后的检查点
    save_fixed_checkpoint(model, checkpoint, new_checkpoint_file)
    print(f"Fixed checkpoint saved to {new_checkpoint_file}")
