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
- 每帧默认只保留置信度最高的一个目标框，适合”一段视频只拍一个实例”的采集方式。
- `rough_crop` 直接用 YOLO 提供 ROI，不做类别区分，类别以视频文件夹名为准。
- 如果某类人工复核后不足 500 张，`finalize` 会保留全部并在 `summary.csv` 里报告缺口。
- 当 `output.review_enabled: false` 时，`select` 不再复制到 `work/review/`，而是直接把候选图写入 manifest，供 `finalize` 直接使用。

---

## 后处理工具

### 1. 打平实例文件夹 (`flatten_instances`)

把大类下的实例子文件夹合并，所有图片直接放在大类文件夹内。

```powershell
# 打平 roi_crop，输出到 work/roi_crop_flatten/
python -m src.flatten_instances --source work/roi_crop_sampled_0515_30006_size_0515_64

# 指定输出目录
python -m src.flatten_instances --source work/roi_crop --output work/my_flat

# 只统计不复制
python -m src.flatten_instances --source work/roi_crop --dry-run

# 用符号链接代替复制（省空间）
python -m src.flatten_instances --source work/roi_crop --symlink
```

### 2. 调试平台网页 (5000) — Fine Crop Debug

HSV 调参 + 红矩形检测 + 交互式 ROI 绘制 + 批量导出。

```powershell
python -m src.fine_crop_debug --port 5000
# 浏览器访问 http://127.0.0.1:5000
```

- 三个导航层级（大类 / 实例 / 图片），支持键盘 `←` `→` 切换
- 四个面板：原图 + ROI 裁剪 + Raw HSV Mask + Morph Mask
- 可调滑块：H_span、S_min、V_min、Close、Open、Mode、R_W、R_H、Lift
- 支持手动拖拽绘制 ROI，或自动检测红色方块
- **Batch Export** 按钮：用当前参数批量导出全部图片的 ROI 裁剪结果到 `work/roi_crop/`

### 3. 数据集采样网页 (5001) — Dataset Sampler

从 work 下的数据集中按时间间隔均匀采样，控制每类总量。

```powershell
python -m src.dataset_sampler --port 5001
# 浏览器访问 http://127.0.0.1:5001
```

- 自动扫描 `work/` 下含 6 大类子文件夹的数据源
- 设置每类抽取数量（默认 5000），额度按实例均分
- 按文件名排序后间隔选取，保证时间维度均匀覆盖
- 输出到 `work/<source>_sampled_<MMDD>_<N>/`

### 4. 缩放填白网页 (5002) — Image Resizer

等比缩放到 n×n 并灰色填充，保持原图不变形。

```powershell
python -m src.image_resizer --port 5002
# 浏览器访问 http://127.0.0.1:5002
```

- 自动扫描 `work/` 下的数据源
- 设置目标尺寸（默认 64×64）
- 长边等比缩放，短边居中，空白处填充灰色 (128, 128, 128)
- 输出到 `work/<source>_size_<MMDD>_<N>/`
