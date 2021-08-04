# Paddle-YOLOv2

PaddlePaddle 复现YOLO9000(YOLOv2)

## 代码结构

1. 本项目基于PaddleDetection开源项目，实现了YOLOv2(YOLO9000)的复现任务。
2. 其中将YOLOv2模型分为了backbone部分、neck部分、head部分。backbone部分为Darknet19，代码在ppdet/modeling/backbones/darknet19.py中；neck部分代码在ppdet/modeling/necks中；head部分代码在ppdet/modeling/heads中。
3. YOLOv2训练配置文件在configs/yolov2中。

## 开始训练

1. cd进本项目目录
2. pip install -r requirements.txt
3. 在顶层目录下创建output文件夹，并在此处下载主干网络Darknet19的预训练权重（我已经将darknet官方提供的转为了pdparams格式），地址为：
4. 本项目使用单卡Tesla V100-32G即可训练，注意：voc数据集应该提前下好，并且解压到顶层目录下（数据集地址为：https://aistudio.baidu.com/aistudio/datasetdetail/63105 ）。
5. python tools/train.py -c configs/yolov2/yolov2_voc.yml --eval
6. 至此训练开始

## 评估

python tools/eval.py -c configs/yolov2/yolov2_voc.yml
