# YOLO Video Review

用于批量查看 `video/` 目录下所有视频上的 YOLO 检测效果。

## 启动

```powershell
py -3.13 -m src.video_review --config config/video_review.yaml
```

可选参数：

```powershell
py -3.13 -m src.video_review --config config/video_review.yaml --class-name ambulance
py -3.13 -m src.video_review --config config/video_review.yaml --video-name ambulance_010.mp4
py -3.13 -m src.video_review --config config/video_review.yaml --conf 0.40 --iou 0.50
```

## 功能

- 自动播放当前视频
- 播放结束后自动从头循环
- `n`：下一个视频
- `c`：下一类视频
- `space`：暂停 / 继续
- `r`：当前视频从头播放
- `q`：退出

## 说明

- 模型默认读取 `model/weights/best.pt`
- 视频默认读取 `video/`
- 画面上会显示当前类别、当前视频、源视频 FPS、单帧推理耗时和快捷键提示
