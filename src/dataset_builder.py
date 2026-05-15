from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


def _lazy_import_numpy():
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing dependency 'numpy'. Install requirements first: "
            "python -m pip install -r requirements.txt"
        ) from exc
    return np


def _lazy_import_cv2():
    try:
        import cv2
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing dependency 'opencv-python'. Install requirements first: "
            "python -m pip install -r requirements.txt"
        ) from exc
    return cv2


def _lazy_import_yaml():
    try:
        import yaml
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing dependency 'PyYAML'. Install requirements first: "
            "python -m pip install -r requirements.txt"
        ) from exc
    return yaml


@dataclass
class Config:
    project_root: Path
    video_root: Path
    work_root: Path
    dataset_root: Path
    yolo_weights: Path | None
    detector_conf_threshold: float
    detector_device: str | None
    detector_classes: list[int] | None
    frame_sample_stride: int
    roi_size: int
    crop_margin_ratio: float
    min_bbox_size: int
    crop_scale_factor: float
    candidate_per_instance: int
    roi_pool_limit_per_instance: int
    pixel_diff_metric: str
    pixel_diff_threshold: float
    review_enabled: bool
    target_per_class: int
    image_extension: str
    rough_crop_root: Path
    rough_crop_frame_stride: int
    rough_crop_scale_factor: float
    rough_crop_image_extension: str
    rough_crop_png_compression: int

    @property
    def detections_root(self) -> Path:
        return self.work_root / "detections"

    @property
    def roi_pool_root(self) -> Path:
        return self.work_root / "roi_pool"

    @property
    def review_root(self) -> Path:
        return self.work_root / "review"

    @property
    def final_root(self) -> Path:
        return self.dataset_root / "final"

    @property
    def review_manifest_path(self) -> Path:
        return self.work_root / "review_manifest.csv"

    @property
    def final_manifest_path(self) -> Path:
        return self.final_root / "manifest.csv"

    @property
    def roi_manifest_path(self) -> Path:
        return self.work_root / "roi_manifest.csv"

    @property
    def rough_crop_manifest_path(self) -> Path:
        return self.rough_crop_root / "manifest.csv"

    @classmethod
    def from_yaml(cls, config_path: Path) -> "Config":
        yaml = _lazy_import_yaml()
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Config file must contain a mapping: {config_path}")

        project_root = config_path.resolve().parent.parent

        paths = raw.get("paths", {})
        detection = raw.get("detection", {})
        crop = raw.get("crop", {})
        selection = raw.get("selection", {})
        output = raw.get("output", {})
        rough_crop = raw.get("rough_crop", {})

        def as_path(value: str | None, default: str) -> Path:
            target = value if value else default
            return (project_root / target).resolve()

        yolo_weights_raw = detection.get("yolo_weights")
        yolo_weights = None
        if yolo_weights_raw:
            yolo_weights = as_path(yolo_weights_raw, yolo_weights_raw)

        detector_classes = detection.get("classes")
        if detector_classes is not None and not isinstance(detector_classes, list):
            raise ValueError("detection.classes must be a list of class ids or null")

        return cls(
            project_root=project_root,
            video_root=as_path(paths.get("video_root"), "video"),
            work_root=as_path(paths.get("work_root"), "work"),
            dataset_root=as_path(paths.get("dataset_root"), "dataset"),
            yolo_weights=yolo_weights,
            detector_conf_threshold=float(detection.get("conf_threshold", 0.25)),
            detector_device=detection.get("device"),
            detector_classes=detector_classes,
            frame_sample_stride=int(detection.get("frame_sample_stride", 3)),
            roi_size=int(crop.get("roi_size", 64)),
            crop_margin_ratio=float(crop.get("margin_ratio", 0.15)),
            min_bbox_size=int(crop.get("min_bbox_size", 12)),
            crop_scale_factor=float(crop.get("scale_factor", 2.0)),
            candidate_per_instance=int(selection.get("candidate_per_instance", 20)),
            roi_pool_limit_per_instance=int(selection.get("roi_pool_limit_per_instance", 120)),
            pixel_diff_metric=str(selection.get("pixel_diff_metric", "mae_gray")),
            pixel_diff_threshold=float(selection.get("pixel_diff_threshold", 0.08)),
            review_enabled=bool(output.get("review_enabled", True)),
            target_per_class=int(output.get("target_per_class", 500)),
            image_extension=str(output.get("image_extension", "png")),
            rough_crop_root=as_path(rough_crop.get("output_root"), "work/rough_crop"),
            rough_crop_frame_stride=int(rough_crop.get("frame_stride", 1)),
            rough_crop_scale_factor=float(rough_crop.get("scale_factor", 2.0)),
            rough_crop_image_extension=str(rough_crop.get("image_extension", "png")),
            rough_crop_png_compression=int(rough_crop.get("png_compression", 0)),
        )


MANIFEST_FIELDNAMES = [
    "class_name",
    "instance_id",
    "video_path",
    "frame_index",
    "timestamp_ms",
    "bbox_xyxy",
    "detector_score",
    "roi_path",
    "diff_score",
    "review_status",
    "final_split",
]


class YoloDetector:
    def __init__(
        self,
        weights_path: Path,
        conf_threshold: float,
        device: str | None,
        class_ids: list[int] | None,
    ) -> None:
        if not weights_path.exists():
            raise FileNotFoundError(
                f"YOLO weights not found: {weights_path}. Update detection.yolo_weights in config."
            )
        try:
            from ultralytics import YOLO
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Missing dependency 'ultralytics'. Install requirements first: "
                "python -m pip install -r requirements.txt"
            ) from exc

        self.model = YOLO(str(weights_path))
        self.conf_threshold = conf_threshold
        self.device = device
        self.class_ids = class_ids

    def detect(self, frame: Any) -> dict[str, Any] | None:
        results = self.model.predict(
            source=frame,
            verbose=False,
            conf=self.conf_threshold,
            device=self.device,
            classes=self.class_ids,
        )
        if not results:
            return None

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return None

        best_index = None
        best_score = -1.0
        for idx, conf in enumerate(boxes.conf.tolist()):
            if conf > best_score:
                best_index = idx
                best_score = float(conf)

        if best_index is None:
            return None

        xyxy = [int(round(v)) for v in boxes.xyxy[best_index].tolist()]
        class_id = int(boxes.cls[best_index].item()) if boxes.cls is not None else None
        return {
            "bbox_xyxy": xyxy,
            "detector_score": best_score,
            "detector_class_id": class_id,
        }


def discover_videos(video_root: Path) -> list[tuple[str, str, Path]]:
    videos: list[tuple[str, str, Path]] = []
    for class_dir in sorted(p for p in video_root.iterdir() if p.is_dir()):
        for video_path in sorted(class_dir.glob("*.mp4")):
            videos.append((class_dir.name, video_path.stem, video_path))
    return videos


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def format_image_name(class_name: str, instance_id: str, frame_index: int, score: float, extension: str) -> str:
    score_token = f"{score:.4f}".replace(".", "_")
    return f"{class_name}__{instance_id}__f{frame_index:06d}__s{score_token}.{extension}"


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(value, high))


def square_crop_box(x1: int, y1: int, x2: int, y2: int, width: int, height: int, margin_ratio: float) -> tuple[int, int, int, int]:
    bbox_w = max(1, x2 - x1)
    bbox_h = max(1, y2 - y1)
    side = int(math.ceil(max(bbox_w, bbox_h) * (1.0 + margin_ratio * 2.0)))
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    left = int(round(cx - side / 2.0))
    top = int(round(cy - side / 2.0))
    right = left + side
    bottom = top + side

    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > width:
        shift = right - width
        left -= shift
        right = width
    if bottom > height:
        shift = bottom - height
        top -= shift
        bottom = height

    left = clamp(left, 0, width)
    top = clamp(top, 0, height)
    right = clamp(right, 0, width)
    bottom = clamp(bottom, 0, height)
    return left, top, right, bottom


def expand_bbox_xyxy(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    width: int,
    height: int,
    scale_factor: float,
) -> tuple[int, int, int, int]:
    bbox_w = max(1, x2 - x1)
    bbox_h = max(1, y2 - y1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    target_w = max(1, int(round(bbox_w * scale_factor)))
    target_h = max(1, int(round(bbox_h * scale_factor)))

    left = int(round(cx - target_w / 2.0))
    top = int(round(cy - target_h / 2.0))
    right = left + target_w
    bottom = top + target_h

    if left < 0:
        right -= left
        left = 0
    if top < 0:
        bottom -= top
        top = 0
    if right > width:
        shift = right - width
        left -= shift
        right = width
    if bottom > height:
        shift = bottom - height
        top -= shift
        bottom = height

    left = clamp(left, 0, width)
    top = clamp(top, 0, height)
    right = clamp(right, 0, width)
    bottom = clamp(bottom, 0, height)
    return left, top, right, bottom


def refine_roi(frame: Any, bbox_xyxy: list[int], roi_size: int, margin_ratio: float, min_bbox_size: int) -> Any | None:
    cv2 = _lazy_import_cv2()
    x1, y1, x2, y2 = bbox_xyxy
    bbox_w = x2 - x1
    bbox_h = y2 - y1
    if bbox_w < min_bbox_size or bbox_h < min_bbox_size:
        return None

    height, width = frame.shape[:2]
    left, top, right, bottom = square_crop_box(x1, y1, x2, y2, width, height, margin_ratio)
    if right <= left or bottom <= top:
        return None

    crop = frame[top:bottom, left:right]
    if crop.size == 0:
        return None

    resized = cv2.resize(crop, (roi_size, roi_size), interpolation=cv2.INTER_AREA)
    return resized


def roi_distance(candidate: Any, selected: Any, metric: str, target_size: tuple[int, int] | None) -> float:
    np = _lazy_import_numpy()
    cv2 = _lazy_import_cv2()

    if metric != "mae_gray":
        raise ValueError(f"Unsupported pixel_diff_metric: {metric}")

    if target_size is not None:
        candidate = cv2.resize(candidate, target_size, interpolation=cv2.INTER_AREA)
        selected = cv2.resize(selected, target_size, interpolation=cv2.INTER_AREA)

    c_gray = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    s_gray = cv2.cvtColor(selected, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    return float(np.mean(np.abs(c_gray - s_gray)))


def load_roi_image(path: Path) -> Any:
    cv2 = _lazy_import_cv2()
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Failed to read ROI image: {path}")
    return image


def compute_selection(
    candidates: list[dict[str, Any]],
    limit: int,
    metric: str,
    threshold: float,
    target_size: tuple[int, int] | None,
) -> list[dict[str, Any]]:
    if len(candidates) <= limit:
        for item in candidates:
            item["diff_score"] = 1.0
        return candidates

    loaded_images = {item["roi_path"]: load_roi_image(Path(item["roi_path"])) for item in candidates}

    selected: list[dict[str, Any]] = [candidates[0].copy()]
    selected[0]["diff_score"] = 1.0

    remaining = [item.copy() for item in candidates[1:]]
    while remaining and len(selected) < limit:
        best_index = None
        best_score = -1.0
        for idx, item in enumerate(remaining):
            item_image = loaded_images[item["roi_path"]]
            min_distance = min(
                roi_distance(item_image, loaded_images[chosen["roi_path"]], metric, target_size)
                for chosen in selected
            )
            if min_distance > best_score:
                best_score = min_distance
                best_index = idx

        if best_index is None:
            break

        chosen = remaining.pop(best_index)
        chosen["diff_score"] = best_score
        if best_score < threshold and len(selected) >= 1:
            break
        selected.append(chosen)

    if len(selected) < limit:
        remaining.sort(
            key=lambda item: min(
                roi_distance(loaded_images[item["roi_path"]], loaded_images[chosen["roi_path"]], metric, target_size)
                for chosen in selected
            ),
            reverse=True,
        )
        for item in remaining:
            if len(selected) >= limit:
                break
            item["diff_score"] = min(
                roi_distance(loaded_images[item["roi_path"]], loaded_images[chosen["roi_path"]], metric, target_size)
                for chosen in selected
            )
            selected.append(item)

    return selected[:limit]


def command_detect(config: Config) -> None:
    detector = YoloDetector(
        weights_path=config.yolo_weights if config.yolo_weights else Path(""),
        conf_threshold=config.detector_conf_threshold,
        device=config.detector_device,
        class_ids=config.detector_classes,
    )
    cv2 = _lazy_import_cv2()

    videos = discover_videos(config.video_root)
    if not videos:
        raise RuntimeError(f"No videos found under: {config.video_root}")

    for class_name, instance_id, video_path in videos:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        detections: list[dict[str, Any]] = []
        frame_index = 0
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % config.frame_sample_stride != 0:
                frame_index += 1
                continue

            detection = detector.detect(frame)
            if detection is not None:
                detection_row = {
                    "class_name": class_name,
                    "instance_id": instance_id,
                    "video_path": str(video_path.resolve()),
                    "frame_index": frame_index,
                    "timestamp_ms": int(round(frame_index / fps * 1000.0)) if fps > 0 else "",
                    "bbox_xyxy": detection["bbox_xyxy"],
                    "detector_score": float(detection["detector_score"]),
                    "detector_class_id": detection["detector_class_id"],
                }
                detections.append(detection_row)
            frame_index += 1

        cap.release()

        detection_path = config.detections_root / class_name / f"{instance_id}.jsonl"
        write_jsonl(detection_path, detections)
        print(f"[detect] {class_name}/{instance_id}: {len(detections)} detections -> {detection_path}")


def command_crop(config: Config) -> None:
    cv2 = _lazy_import_cv2()
    detection_files = sorted(config.detections_root.glob("*/*.jsonl"))
    if not detection_files:
        raise RuntimeError(
            f"No detection manifests found under {config.detections_root}. Run detect stage first."
        )

    roi_rows: list[dict[str, Any]] = []
    for detection_file in detection_files:
        rows = read_jsonl(detection_file)
        if not rows:
            print(f"[crop] skip empty detection manifest: {detection_file}")
            continue

        class_name = rows[0]["class_name"]
        instance_id = rows[0]["instance_id"]
        video_path = Path(rows[0]["video_path"])
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        frames_to_extract = {int(row["frame_index"]): row for row in rows}
        roi_dir = config.roi_pool_root / class_name / instance_id
        roi_dir.mkdir(parents=True, exist_ok=True)

        saved = 0
        frame_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            row = frames_to_extract.get(frame_index)
            if row is None:
                frame_index += 1
                continue

            bbox = row["bbox_xyxy"]
            if (bbox[2] - bbox[0]) < config.min_bbox_size or (bbox[3] - bbox[1]) < config.min_bbox_size:
                frame_index += 1
                continue

            height, width = frame.shape[:2]
            left, top, right, bottom = expand_bbox_xyxy(
                bbox[0],
                bbox[1],
                bbox[2],
                bbox[3],
                width,
                height,
                config.crop_scale_factor,
            )
            if right <= left or bottom <= top:
                frame_index += 1
                continue

            roi = frame[top:bottom, left:right]
            if roi.size > 0:
                file_name = format_image_name(
                    class_name,
                    instance_id,
                    int(row["frame_index"]),
                    float(row["detector_score"]),
                    config.image_extension,
                )
                roi_path = roi_dir / file_name
                cv2.imwrite(str(roi_path), roi)
                roi_rows.append(
                    {
                        "class_name": class_name,
                        "instance_id": instance_id,
                        "video_path": row["video_path"],
                        "frame_index": row["frame_index"],
                        "timestamp_ms": row["timestamp_ms"],
                        "bbox_xyxy": json.dumps(row["bbox_xyxy"], ensure_ascii=True),
                        "detector_score": row["detector_score"],
                        "roi_path": str(roi_path.resolve()),
                        "diff_score": "",
                        "review_status": "pending",
                        "final_split": "",
                    }
                )
                saved += 1
            frame_index += 1

        cap.release()
        print(f"[crop] {class_name}/{instance_id}: {saved} crops -> {roi_dir}")

    if roi_rows:
        write_csv(config.roi_manifest_path, roi_rows)
        print(f"[crop] roi manifest -> {config.roi_manifest_path}")


def command_rough_crop(config: Config) -> None:
    if not config.yolo_weights:
        raise RuntimeError("Config detection.yolo_weights is required for rough_crop stage.")

    detector = YoloDetector(
        weights_path=config.yolo_weights,
        conf_threshold=config.detector_conf_threshold,
        device=config.detector_device,
        class_ids=config.detector_classes,
    )
    cv2 = _lazy_import_cv2()

    videos = discover_videos(config.video_root)
    if not videos:
        raise RuntimeError(f"No videos found under: {config.video_root}")

    manifest_rows: list[dict[str, Any]] = []
    for class_name, instance_id, video_path in videos:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        output_dir = config.rough_crop_root / class_name / instance_id
        output_dir.mkdir(parents=True, exist_ok=True)

        saved = 0
        frame_index = 0
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % config.rough_crop_frame_stride != 0:
                frame_index += 1
                continue

            detection = detector.detect(frame)
            if detection is None:
                frame_index += 1
                continue

            bbox = detection["bbox_xyxy"]
            if (bbox[2] - bbox[0]) < config.min_bbox_size or (bbox[3] - bbox[1]) < config.min_bbox_size:
                frame_index += 1
                continue

            height, width = frame.shape[:2]
            left, top, right, bottom = expand_bbox_xyxy(
                bbox[0],
                bbox[1],
                bbox[2],
                bbox[3],
                width,
                height,
                config.rough_crop_scale_factor,
            )
            if right <= left or bottom <= top:
                frame_index += 1
                continue

            crop = frame[top:bottom, left:right]
            if crop.size == 0:
                frame_index += 1
                continue

            file_name = format_image_name(
                class_name,
                instance_id,
                frame_index,
                float(detection["detector_score"]),
                config.rough_crop_image_extension,
            )
            output_path = output_dir / file_name
            write_params: list[int] = []
            if config.rough_crop_image_extension.lower() == "png":
                write_params = [cv2.IMWRITE_PNG_COMPRESSION, config.rough_crop_png_compression]
            cv2.imwrite(str(output_path), crop, write_params)

            manifest_rows.append(
                {
                    "class_name": class_name,
                    "instance_id": instance_id,
                    "video_path": str(video_path.resolve()),
                    "frame_index": frame_index,
                    "timestamp_ms": int(round(frame_index / fps * 1000.0)) if fps > 0 else "",
                    "bbox_xyxy": json.dumps(bbox, ensure_ascii=True),
                    "detector_score": float(detection["detector_score"]),
                    "roi_path": str(output_path.resolve()),
                }
            )
            saved += 1
            frame_index += 1

        cap.release()
        print(f"[rough_crop] {class_name}/{instance_id}: {saved} crops -> {output_dir}")

    if manifest_rows:
        write_generic_csv(config.rough_crop_manifest_path, manifest_rows)
        print(f"[rough_crop] manifest -> {config.rough_crop_manifest_path}")


def command_select(config: Config) -> None:
    if not config.roi_manifest_path.exists():
        raise RuntimeError(
            f"ROI manifest not found: {config.roi_manifest_path}. Run crop stage first."
        )

    roi_manifest_rows = read_csv(config.roi_manifest_path)
    rows_by_instance: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in roi_manifest_rows:
        rows_by_instance[(row["class_name"], row["instance_id"])].append(row)

    candidate_rows: list[dict[str, Any]] = []

    for (class_name, instance_id), rows in sorted(rows_by_instance.items()):
        rows.sort(key=lambda row: int(row["frame_index"]))
        candidates = rows[: config.roi_pool_limit_per_instance]
        if not candidates:
            continue

        chosen = compute_selection(
            candidates=candidates,
            limit=config.candidate_per_instance,
            metric=config.pixel_diff_metric,
            threshold=config.pixel_diff_threshold,
            target_size=(config.roi_size, config.roi_size),
        )

        if config.review_enabled:
            review_dir = config.review_root / class_name / instance_id
            if review_dir.exists():
                for stale_file in review_dir.glob(f"*.{config.image_extension}"):
                    stale_file.unlink()
            review_dir.mkdir(parents=True, exist_ok=True)

            for item in chosen:
                source_path = Path(item["roi_path"])
                target_path = review_dir / source_path.name
                shutil.copy2(source_path, target_path)
                item["roi_path"] = str(target_path.resolve())
                candidate_rows.append(item)

            print(
                f"[select] {class_name}/{instance_id}: "
                f"{len(chosen)} review candidates -> {review_dir}"
            )
        else:
            for item in chosen:
                item["review_status"] = "auto_kept"
                candidate_rows.append(item)
            print(
                f"[select] {class_name}/{instance_id}: "
                f"{len(chosen)} candidates retained without review"
            )

    if not candidate_rows:
        raise RuntimeError(
            f"No ROI crops found under {config.roi_pool_root}. Run crop stage first."
        )

    write_csv(config.review_manifest_path, candidate_rows)
    print(f"[select] review manifest -> {config.review_manifest_path}")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            normalized = {field: row.get(field, "") for field in MANIFEST_FIELDNAMES}
            writer.writerow(normalized)


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def allocate_quota(items_by_instance: dict[str, list[dict[str, Any]]], target_total: int) -> dict[str, int]:
    instance_ids = sorted(items_by_instance.keys())
    quotas = {instance_id: 0 for instance_id in instance_ids}
    total_available = sum(len(items) for items in items_by_instance.values())
    if total_available <= target_total:
        return {instance_id: len(items) for instance_id, items in items_by_instance.items()}

    base = target_total // len(instance_ids)
    remainder = target_total % len(instance_ids)

    capacities = {
        instance_id: len(items)
        for instance_id, items in items_by_instance.items()
    }
    for instance_id in instance_ids:
        quotas[instance_id] = min(base, capacities[instance_id])

    remaining = target_total - sum(quotas.values())
    ranked = sorted(
        instance_ids,
        key=lambda instance_id: (capacities[instance_id] - quotas[instance_id], instance_id),
        reverse=True,
    )
    rank_index = 0
    while remaining > 0 and ranked:
        instance_id = ranked[rank_index % len(ranked)]
        if quotas[instance_id] < capacities[instance_id]:
            quotas[instance_id] += 1
            remaining -= 1
        rank_index += 1

    if remainder and remaining == 0:
        return quotas
    return quotas


def command_finalize(config: Config) -> None:
    if not config.review_manifest_path.exists():
        raise RuntimeError(
            f"Review manifest not found: {config.review_manifest_path}. Run select stage first."
        )

    manifest_rows = read_csv(config.review_manifest_path)
    kept_rows: list[dict[str, Any]] = []
    for row in manifest_rows:
        roi_path = Path(row["roi_path"])
        if roi_path.exists():
            row["review_status"] = "kept"
            kept_rows.append(row)
        else:
            row["review_status"] = "deleted"

    by_class: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in kept_rows:
        by_class[row["class_name"]][row["instance_id"]].append(row)

    config.final_root.mkdir(parents=True, exist_ok=True)
    final_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for class_name, items_by_instance in sorted(by_class.items()):
        class_dir = config.final_root / class_name
        if class_dir.exists():
            for stale_file in class_dir.glob(f"*.{config.image_extension}"):
                stale_file.unlink()
        class_dir.mkdir(parents=True, exist_ok=True)

        quotas = allocate_quota(items_by_instance, config.target_per_class)
        class_total = 0
        for instance_id, rows in sorted(items_by_instance.items()):
            rows.sort(
                key=lambda row: (
                    float(row["diff_score"] or 0.0),
                    float(row["detector_score"] or 0.0),
                    int(row["frame_index"] or -1),
                ),
                reverse=True,
            )
            chosen = rows[: quotas[instance_id]]
            for row in chosen:
                source_path = Path(row["roi_path"])
                final_name = source_path.name
                target_path = class_dir / final_name
                shutil.copy2(source_path, target_path)
                row["roi_path"] = str(target_path.resolve())
                row["final_split"] = "train"
                final_rows.append(row)
                class_total += 1

        summary_rows.append(
            {
                "class_name": class_name,
                "kept_after_review": sum(len(rows) for rows in items_by_instance.values()),
                "final_count": class_total,
                "target_per_class": config.target_per_class,
                "gap": max(0, config.target_per_class - class_total),
            }
        )
        print(f"[finalize] {class_name}: {class_total} images -> {class_dir}")

    write_csv(config.final_manifest_path, final_rows)
    write_generic_csv(config.final_root / "summary.csv", summary_rows)
    print(f"[finalize] final manifest -> {config.final_manifest_path}")


def write_generic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    ensure_parent(path)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build classification dataset from instance videos.")
    parser.add_argument(
        "stage",
        choices=["detect", "crop", "select", "finalize", "rough_crop"],
        help="Pipeline stage to run.",
    )
    parser.add_argument(
        "--config",
        default="config/dataset_build.yaml",
        help="Path to YAML config file. Default: config/dataset_build.yaml",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = Config.from_yaml(Path(args.config).resolve())

    try:
        if args.stage == "detect":
            if not config.yolo_weights:
                raise RuntimeError(
                    "Config detection.yolo_weights is required for detect stage."
                )
            command_detect(config)
        elif args.stage == "crop":
            command_crop(config)
        elif args.stage == "select":
            command_select(config)
        elif args.stage == "finalize":
            command_finalize(config)
        elif args.stage == "rough_crop":
            command_rough_crop(config)
        else:
            parser.error(f"Unsupported stage: {args.stage}")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
