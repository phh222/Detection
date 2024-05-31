## 使用目标检测框架mmdetection，在VOC数据集上训练并测试目标检测模型Faster R-CNN和YOLO V3

Faster R-CNN 和 yolo v3模型的配置文件和训练好的模型权重链接如下.
|   Model         | config name  | Download |
|:---------------:|:-----------:|:---------:|
| Faster R-CNN  | [Faster R-RNN](https://github.com/phh222/Detection/blob/master/mmdetection/configs/pascal_voc/faster-rcnn_r50_fpn_1x_voc0712.py) | [checkpoint](https://pan.baidu.com/s/1aipdqtlVXYALo8P94ovu5g?pwd=xga5)  |
|YOLOv3 | [YOLOv3](https://github.com/phh222/Detection/blob/master/mmdetection/configs/pascal_voc/yolov3_d53_mstrain-608_100e_voc0712.py) | [checkpoint](https://pan.baidu.com/s/1aipdqtlVXYALo8P94ovu5g?pwd=xga5)  |

### Data
需提前自行下载好 voc 2007 和 2012数据，放在data目录下

### Training
在mmdetection 文件夹下执行 
```
tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} 
```

### Test
在测试集上评估模型效果，mAP
```
tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM}
```

测试单张图片的检测结果
```
python demo/image_demo.py ${image_path} ${CONFIG_FILE} --weights ${CHECKPOINT_FILE} --device cpu
```
