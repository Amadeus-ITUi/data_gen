from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import yaml


BACKEND_MAP = {
    "ANY": cv2.CAP_ANY,
    "DSHOW": getattr(cv2, "CAP_DSHOW", cv2.CAP_ANY),
    "MSMF": getattr(cv2, "CAP_MSMF", cv2.CAP_ANY),
    "V4L2": getattr(cv2, "CAP_V4L2", cv2.CAP_ANY),
    "FFMPEG": getattr(cv2, "CAP_FFMPEG", cv2.CAP_ANY),
}


@dataclass(slots=True)
class ModelConfig:
    weights_path: Path
    class_names: list[str]
    imgsz: int
    conf_threshold: float
    iou_threshold: float
    device: str | None
    max_det: int
    show_labels: bool
    show_conf: bool


@dataclass(slots=True)
class CameraConfig:
    default_index: int
    scan_max_index: int
    backend: str
    fourcc: str
    width: int
    height: int
    fps: int


@dataclass(slots=True)
class RuntimeConfig:
    model: ModelConfig
    camera: CameraConfig


@dataclass(slots=True)
class CaptureProfile:
    backend_name: str
    fourcc: str
    width: int
    height: int
    fps: float


def load_runtime_config(path: Path) -> RuntimeConfig:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a mapping: {path}")

    model_raw = raw.get("model", {})
    camera_raw = raw.get("camera", {})

    if not isinstance(model_raw, dict) or not isinstance(camera_raw, dict):
        raise ValueError("Config sections `model` and `camera` must be mappings.")

    class_names = model_raw.get("class_names", [])
    if not isinstance(class_names, list) or not class_names:
        raise ValueError("model.class_names must be a non-empty list.")

    weights_path = Path(str(model_raw.get("weights_path", ""))).resolve()
    if weights_path.suffix.lower() != ".pt":
        raise ValueError("Only `.pt` weights are supported in v1.")
    if not weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    return RuntimeConfig(
        model=ModelConfig(
            weights_path=weights_path,
            class_names=[str(item) for item in class_names],
            imgsz=int(model_raw.get("imgsz", 640)),
            conf_threshold=float(model_raw.get("conf_threshold", 0.35)),
            iou_threshold=float(model_raw.get("iou_threshold", 0.45)),
            device=(str(model_raw["device"]) if model_raw.get("device") not in (None, "") else None),
            max_det=int(model_raw.get("max_det", 20)),
            show_labels=bool(model_raw.get("show_labels", True)),
            show_conf=bool(model_raw.get("show_conf", True)),
        ),
        camera=CameraConfig(
            default_index=int(camera_raw.get("default_index", 1)),
            scan_max_index=int(camera_raw.get("scan_max_index", 5)),
            backend=str(camera_raw.get("backend", "DSHOW")),
            fourcc=str(camera_raw.get("fourcc", "MJPG")),
            width=int(camera_raw.get("width", 320)),
            height=int(camera_raw.get("height", 240)),
            fps=int(camera_raw.get("fps", 120)),
        ),
    )


def open_camera(config: CameraConfig, camera_index: int) -> tuple[cv2.VideoCapture, CaptureProfile]:
    backend_name = config.backend.upper()
    backend_value = BACKEND_MAP.get(backend_name, cv2.CAP_ANY)
    cap = cv2.VideoCapture(camera_index, backend_value)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Camera {camera_index} did not open with backend {backend_name}.")

    if config.fourcc.upper() != "ANY":
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*config.fourcc[:4]))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.height)
    cap.set(cv2.CAP_PROP_FPS, config.fps)

    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        raise RuntimeError(
            f"Camera {camera_index} opened with backend {backend_name}, "
            f"but no frame was returned for {config.width}x{config.height}@{config.fps}."
        )

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or frame.shape[1])
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or frame.shape[0])
    actual_fps = float(cap.get(cv2.CAP_PROP_FPS) or config.fps)
    profile = CaptureProfile(
        backend_name=backend_name,
        fourcc=config.fourcc.upper(),
        width=actual_width,
        height=actual_height,
        fps=actual_fps if actual_fps > 0 else float(config.fps),
    )
    return cap, profile


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Realtime YOLO preview using the local camera.")
    parser.add_argument(
        "--config",
        default="config/realtime_yolo.yaml",
        help="Path to runtime config. Default: config/realtime_yolo.yaml",
    )
    parser.add_argument("--weights", help="Override model weights path.")
    parser.add_argument("--camera-index", type=int, help="Override camera index.")
    parser.add_argument("--conf", type=float, help="Override confidence threshold.")
    parser.add_argument("--iou", type=float, help="Override IOU threshold.")
    return parser


def load_model(weights_path: Path):
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing dependency `ultralytics`. Install requirements first."
        ) from exc
    return YOLO(str(weights_path))


def class_color(class_id: int) -> tuple[int, int, int]:
    palette = [
        (255, 143, 61),
        (76, 201, 240),
        (67, 170, 139),
        (249, 199, 79),
        (249, 65, 68),
        (87, 117, 144),
        (144, 190, 109),
        (39, 125, 161),
    ]
    return palette[class_id % len(palette)]


def format_label(class_name: str, confidence: float, show_conf: bool) -> str:
    return f"{class_name} {confidence:.2f}" if show_conf else class_name


def draw_detections(frame: Any, result: Any, model_config: ModelConfig) -> Any:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return frame

    xyxy_list = boxes.xyxy.tolist()
    conf_list = boxes.conf.tolist() if boxes.conf is not None else [0.0] * len(xyxy_list)
    cls_list = boxes.cls.tolist() if boxes.cls is not None else [0] * len(xyxy_list)

    for xyxy, conf, cls_id in zip(xyxy_list, conf_list, cls_list):
        x1, y1, x2, y2 = [int(round(v)) for v in xyxy]
        class_index = int(cls_id)
        class_name = (
            model_config.class_names[class_index]
            if 0 <= class_index < len(model_config.class_names)
            else f"class_{class_index}"
        )
        color = class_color(class_index)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        if model_config.show_labels:
            label = format_label(class_name, float(conf), model_config.show_conf)
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_top = max(0, y1 - text_h - baseline - 6)
            text_bottom = text_top + text_h + baseline + 6
            cv2.rectangle(frame, (x1, text_top), (x1 + text_w + 8, text_bottom), color, thickness=-1)
            cv2.putText(
                frame,
                label,
                (x1 + 4, text_bottom - baseline - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )
    return frame


def run_preview(runtime_config: RuntimeConfig, camera_index: int | None, conf: float | None, iou: float | None, weights: str | None) -> int:
    model_config = runtime_config.model
    if weights:
        model_config = ModelConfig(
            weights_path=Path(weights).resolve(),
            class_names=model_config.class_names,
            imgsz=model_config.imgsz,
            conf_threshold=model_config.conf_threshold,
            iou_threshold=model_config.iou_threshold,
            device=model_config.device,
            max_det=model_config.max_det,
            show_labels=model_config.show_labels,
            show_conf=model_config.show_conf,
        )
        if not model_config.weights_path.exists():
            raise FileNotFoundError(f"Override weights not found: {model_config.weights_path}")
        if model_config.weights_path.suffix.lower() != ".pt":
            raise ValueError("Only `.pt` weights are supported in v1.")

    if conf is not None:
        model_config.conf_threshold = conf
    if iou is not None:
        model_config.iou_threshold = iou

    active_camera_index = runtime_config.camera.default_index if camera_index is None else camera_index
    cap, profile = open_camera(runtime_config.camera, active_camera_index)
    model = load_model(model_config.weights_path)

    print(
        f"Camera {active_camera_index} opened: "
        f"{profile.backend_name} {profile.width}x{profile.height} @{profile.fps:.1f} FPS, "
        f"FOURCC={profile.fourcc}"
    )
    if profile.width != runtime_config.camera.width or profile.height != runtime_config.camera.height:
        print(
            f"Requested {runtime_config.camera.width}x{runtime_config.camera.height}, "
            f"actual {profile.width}x{profile.height}."
        )

    window_name = "Realtime YOLO Preview"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, max(960, profile.width * 2), max(540, profile.height * 2))

    last_infer_time = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError("Failed to read frame from camera.")

            infer_start = time.perf_counter()
            results = model.predict(
                source=frame,
                imgsz=model_config.imgsz,
                conf=model_config.conf_threshold,
                iou=model_config.iou_threshold,
                device=model_config.device,
                max_det=model_config.max_det,
                verbose=False,
                agnostic_nms=False,
            )
            last_infer_time = time.perf_counter() - infer_start

            annotated = frame.copy()
            if results:
                annotated = draw_detections(annotated, results[0], model_config)

            cv2.putText(
                annotated,
                f"infer {last_infer_time * 1000.0:.1f} ms",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                annotated,
                f"infer {last_infer_time * 1000.0:.1f} ms",
                (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (40, 40, 40),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        runtime_config = load_runtime_config(Path(args.config).resolve())
        return run_preview(
            runtime_config=runtime_config,
            camera_index=args.camera_index,
            conf=args.conf,
            iou=args.iou,
            weights=args.weights,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
