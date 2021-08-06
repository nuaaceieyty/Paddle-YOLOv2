# Paddle-YOLOv2

PaddlePaddle_v2.1 复现YOLO9000(YOLOv2)

## 代码结构

1. 本项目基于PaddleDetection开源项目，实现了YOLOv2(YOLO9000)的复现任务。
2. 其中将YOLOv2模型分为了backbone部分、neck部分、head部分。backbone部分为Darknet19，代码在ppdet/modeling/backbones/darknet19.py中；neck部分代码在ppdet/modeling/necks中；head部分代码在ppdet/modeling/heads中。
3. YOLOv2训练配置文件在configs/yolov2中。

## 开始训练

1. cd进本项目目录。
2. pip install -r requirements.txt
3. 在顶层目录下创建output文件夹，并在此处下载主干网络Darknet19的预训练权重（我已经将darknet官方提供的转为了pdparams格式），地址为：https://aistudio.baidu.com/aistudio/datasetdetail/103069 。
4. 本项目使用单卡Tesla V100-32G即可训练，注意：voc数据集应该提前下好，并且解压到顶层目录下（数据集地址为：https://aistudio.baidu.com/aistudio/datasetdetail/63105 ）。如果出现数据集地址问题，请在configs/datasets/voc.yml文件中将相应地址改为绝对路径。
5. python tools/train.py -c configs/yolov2/yolov2_voc.yml --eval
6. 至此训练开始。

## 评估

1. 如果您不想自己训练，可以在顶层文件下创建output/yolov2_voc文件夹，并下载我训练好的权重数据到此文件夹中（地址为：https://aistudio.baidu.com/aistudio/datasetdetail/103354 ），然后运行第2步中命令。
2. python tools/eval.py -c configs/yolov2/yolov2_voc.yml （使用上述权重即可得到精度76.82%）

## 预测


(附注：训练时在接近后期阶段时，mAP可以达到很接近76.8的程度，这时需要临时微调一些超参数以达到复现精度)
