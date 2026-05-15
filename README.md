# LoongCar Data Generator

用于把 `video/<class>/*.mp4` 批量处理成分类模型训练数据集。

## 流程

1. `detect`: 用 YOLO 对视频逐帧粗定位，输出 `work/detections/*.jsonl`
2. `crop`: 根据检测框裁剪并扩框到 2 倍（中心不变，不缩放），输出 `work/roi_pool/`
3. `select`: 用 ROI 像素差异做贪心去重，给每个实例挑选候选图，输出 `work/review/` 和 `work/review_manifest.csv`
4. 人工复核：直接删除 `work/review/` 中误检或不想保留的图片
5. `finalize`: 按每类 500 张做实例均衡下采样，输出 `dataset/final/`
6. `rough_crop`: 直接对视频逐帧检测并扩框裁剪（中心不变，长宽放大），输出 `work/rough_crop/`

## 安装

```powershell
py -3.13 -m pip install -r requirements.txt
```

如果你直接运行 `python`，这台机器当前优先命中的是 MSYS2 Python，默认不带可直接使用的 `pip`。建议统一用 `py -3.13` 或 `D:\Python\Python313\python.exe`。

## 配置

主配置文件是 [config/dataset_build.yaml](config/dataset_build.yaml)。

需要重点修改：

- `detection.yolo_weights`: 你的 YOLO 权重路径
- `crop.roi_size`: ROI 比较用的目标尺寸（用于去重）
- `crop.scale_factor`: 扩框倍数（中心不变）
- `selection.candidate_per_instance`: 每个实例候选张数
- `output.target_per_class`: 每类最终目标张数
- `rough_crop.scale_factor`: 扩框倍数（中心不变）
- `rough_crop.frame_stride`: 逐帧处理步长（1 表示全帧）

## 命令

```powershell
python -m src.dataset_builder detect --config config/dataset_build.yaml
python -m src.dataset_builder crop --config config/dataset_build.yaml
python -m src.dataset_builder select --config config/dataset_build.yaml
python -m src.dataset_builder finalize --config config/dataset_build.yaml
python -m src.dataset_builder rough_crop --config config/dataset_build.yaml
```

更稳的调用方式：

```powershell
py -3.13 -m src.dataset_builder detect --config config/dataset_build.yaml
```

## 输出目录

- `work/detections/<class>/<instance>.jsonl`
- `work/roi_pool/<class>/<instance>/*.png`
- `work/review/<class>/<instance>/*.png`
- `work/review_manifest.csv`
- `work/rough_crop/<class>/<instance>/*.png`
- `work/rough_crop/manifest.csv`
- `dataset/final/<class>/*.png`
- `dataset/final/manifest.csv`
- `dataset/final/summary.csv`

## 说明

- 当前 `detect` 阶段依赖已经训练好的 YOLO 权重。
- 每帧默认只保留置信度最高的一个目标框，适合“一段视频只拍一个实例”的采集方式。
- `rough_crop` 直接用 YOLO 提供 ROI，不做类别区分，类别以视频文件夹名为准。
- 如果某类人工复核后不足 500 张，`finalize` 会保留全部并在 `summary.csv` 里报告缺口。
- 当 `output.review_enabled: false` 时，`select` 不再复制到 `work/review/`，而是直接把候选图写入 manifest，供 `finalize` 直接使用。
