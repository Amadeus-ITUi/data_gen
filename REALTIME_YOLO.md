# Realtime YOLO Preview

使用当前项目训练得到的 `model/weights/best.pt` 做本地实时检测预览。

## 配置

主配置文件：

- `config/realtime_yolo.yaml`

其中包含两部分：

- `model`: 权重路径、类别名、输入尺寸、置信度阈值等
- `camera`: 与 `video_gen` 保持一致的相机参数镜像

当前默认相机参数：

- backend: `DSHOW`
- fourcc: `MJPG`
- width: `320`
- height: `240`
- fps: `120`
- default_index: `1`

## 启动

```powershell
py -3.13 -m src.realtime_yolo --config config/realtime_yolo.yaml
```

可选参数：

```powershell
py -3.13 -m src.realtime_yolo --config config/realtime_yolo.yaml --camera-index 0
py -3.13 -m src.realtime_yolo --config config/realtime_yolo.yaml --conf 0.40 --iou 0.50
py -3.13 -m src.realtime_yolo --config config/realtime_yolo.yaml --weights model/weights/best.pt
```

## 交互

- 按 `q` 退出预览

## 说明

- 当前只支持 `.pt` 权重
- 画面默认显示：检测框、类别名、置信度
- 如果摄像头实际返回的分辨率与配置不同，脚本会继续运行并打印实际值
