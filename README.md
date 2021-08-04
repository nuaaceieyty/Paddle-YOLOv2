# Paddle-YOLOv2

PaddlePaddle 复现YOLO9000(YOLOv2)

## 代码结构

本项目基于PaddleDetection开源项目，实现了YOLOv2(YOLO9000)的复现任务。

其中将YOLOv2模型分为了backbone部分、neck部分、head部分。backbone部分为Darknet19，代码在ppdet/modeling/backbones/darknet19.py中；neck部分代码在ppdet/modeling/necks中；head部分代码在ppdet/modeling/heads中。

YOLOv2训练配置文件在configs/yolov2中。
