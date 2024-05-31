## Detection 

The config files for faster R-CNN, FCOS, and YOLOv3 are shown in the following table.
|   Model         | config name  | Download |
|:---------------:|:-----------:|:---------:|
| Faster R-CNN  | [Faster R-RNN](https://github.com/phh222/Detection/blob/master/mmdetection/configs/pascal_voc/faster-rcnn_r50_fpn_1x_voc0712.py) | [checkpoint](https://pan.baidu.com/s/1CyFIBYO1TQSDm6anTxy-sA)  |
|YOLOv3 | [YOLOv3](https://github.com/phh222/Detection/blob/master/mmdetection/configs/pascal_voc/yolov3_d53_mstrain-608_100e_voc0712.py) | [checkpoint](https://pan.baidu.com/s/1xJV3-rZ7-dTuvbTsCHt-uw)  |

### Training
please first turn to the mmdetection and then run 
```
tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} 
```

### Test
To test our trained model, please run
```
tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM}
```
