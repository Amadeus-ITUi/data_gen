from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import yaml

from src.realtime_yolo import ModelConfig, draw_detections, load_model, load_runtime_config


@dataclass(slots=True)
class ReviewConfig:
    model: ModelConfig
    video_root: Path
    playback_fps: float | None
    window_name: str
    start_class: str | None
    start_video: str | None


@dataclass(slots=True)
class VideoLibrary:
    class_names: list[str]
    videos_by_class: dict[str, list[Path]]


def load_review_config(path: Path) -> ReviewConfig:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a mapping: {path}")

    runtime_config = load_runtime_config(path)
    review_raw = raw.get("review", {})
    if not isinstance(review_raw, dict):
        raise ValueError("Config section `review` must be a mapping.")

    video_root = Path(str(review_raw.get("video_root", "video"))).resolve()
    if not video_root.exists():
        raise FileNotFoundError(f"Video root not found: {video_root}")

    playback_fps = review_raw.get("playback_fps")
    return ReviewConfig(
        model=runtime_config.model,
        video_root=video_root,
        playback_fps=float(playback_fps) if playback_fps not in (None, "") else None,
        window_name=str(review_raw.get("window_name", "YOLO Video Review")),
        start_class=str(review_raw["start_class"]) if review_raw.get("start_class") else None,
        start_video=str(review_raw["start_video"]) if review_raw.get("start_video") else None,
    )


def discover_library(video_root: Path) -> VideoLibrary:
    class_names: list[str] = []
    videos_by_class: dict[str, list[Path]] = {}
    for class_dir in sorted(path for path in video_root.iterdir() if path.is_dir()):
        videos = sorted(class_dir.glob("*.mp4"))
        if videos:
            class_names.append(class_dir.name)
            videos_by_class[class_dir.name] = videos

    if not class_names:
        raise RuntimeError(f"No videos found under: {video_root}")

    return VideoLibrary(class_names=class_names, videos_by_class=videos_by_class)


def find_start_indices(library: VideoLibrary, start_class: str | None, start_video: str | None) -> tuple[int, int]:
    class_index = 0
    if start_class and start_class in library.class_names:
        class_index = library.class_names.index(start_class)

    video_index = 0
    if start_video:
        current_class = library.class_names[class_index]
        video_names = [path.name for path in library.videos_by_class[current_class]]
        if start_video in video_names:
            video_index = video_names.index(start_video)

    return class_index, video_index


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Review YOLO detections over local videos.")
    parser.add_argument(
        "--config",
        default="config/video_review.yaml",
        help="Path to review config. Default: config/video_review.yaml",
    )
    parser.add_argument("--weights", help="Override model weights path.")
    parser.add_argument("--conf", type=float, help="Override confidence threshold.")
    parser.add_argument("--iou", type=float, help="Override IOU threshold.")
    parser.add_argument("--class-name", help="Start from a specific class name.")
    parser.add_argument("--video-name", help="Start from a specific video filename.")
    return parser


def override_model_config(model_config: ModelConfig, weights: str | None, conf: float | None, iou: float | None) -> ModelConfig:
    updated = ModelConfig(
        weights_path=model_config.weights_path,
        class_names=model_config.class_names,
        imgsz=model_config.imgsz,
        conf_threshold=model_config.conf_threshold,
        iou_threshold=model_config.iou_threshold,
        device=model_config.device,
        max_det=model_config.max_det,
        show_labels=model_config.show_labels,
        show_conf=model_config.show_conf,
    )

    if weights:
        updated.weights_path = Path(weights).resolve()
        if updated.weights_path.suffix.lower() != ".pt":
            raise ValueError("Only `.pt` weights are supported in v1.")
        if not updated.weights_path.exists():
            raise FileNotFoundError(f"Override weights not found: {updated.weights_path}")

    if conf is not None:
        updated.conf_threshold = conf
    if iou is not None:
        updated.iou_threshold = iou
    return updated


def open_video(video_path: Path) -> tuple[cv2.VideoCapture, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    return cap, fps


def next_video_index(library: VideoLibrary, class_index: int, video_index: int) -> int:
    current_class = library.class_names[class_index]
    return (video_index + 1) % len(library.videos_by_class[current_class])


def next_class_indices(library: VideoLibrary, class_index: int) -> tuple[int, int]:
    next_class = (class_index + 1) % len(library.class_names)
    return next_class, 0


def draw_overlay(
    frame,
    class_name: str,
    video_path: Path,
    class_index: int,
    class_count: int,
    video_index: int,
    video_count: int,
    source_fps: float,
    infer_ms: float,
    paused: bool,
) -> None:
    lines = [
        f"class {class_index + 1}/{class_count}: {class_name}",
        f"video {video_index + 1}/{video_count}: {video_path.name}",
        f"source_fps: {source_fps:.1f}  infer: {infer_ms:.1f} ms",
        "keys: n next video  c next class  space pause  r restart  q quit",
    ]
    if paused:
        lines.append("status: paused")

    top = 22
    for line in lines:
        cv2.putText(frame, line, (10, top), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, line, (10, top), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (35, 35, 35), 1, cv2.LINE_AA)
        top += 22


def run_review(config: ReviewConfig, weights: str | None, conf: float | None, iou: float | None, start_class: str | None, start_video: str | None) -> int:
    model_config = override_model_config(config.model, weights, conf, iou)
    library = discover_library(config.video_root)
    class_index, video_index = find_start_indices(
        library=library,
        start_class=start_class or config.start_class,
        start_video=start_video or config.start_video,
    )
    model = load_model(model_config.weights_path)

    cv2.namedWindow(config.window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(config.window_name, 1280, 720)

    paused = False
    cap: cv2.VideoCapture | None = None
    source_fps = 0.0
    last_frame = None
    last_infer_ms = 0.0

    try:
        while True:
            class_name = library.class_names[class_index]
            videos = library.videos_by_class[class_name]
            current_video_path = videos[video_index]

            if cap is None:
                cap, source_fps = open_video(current_video_path)
                print(f"Reviewing {class_name}/{current_video_path.name}")
                last_frame = None
                last_infer_ms = 0.0

            if not paused or last_frame is None:
                ok, frame = cap.read()
                if not ok or frame is None:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = cap.read()
                    if not ok or frame is None:
                        raise RuntimeError(f"Failed to loop video: {current_video_path}")

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
                last_infer_ms = (time.perf_counter() - infer_start) * 1000.0

                annotated = frame.copy()
                if results:
                    annotated = draw_detections(annotated, results[0], model_config)
                last_frame = annotated

            annotated = last_frame.copy()
            draw_overlay(
                annotated,
                class_name=class_name,
                video_path=current_video_path,
                class_index=class_index,
                class_count=len(library.class_names),
                video_index=video_index,
                video_count=len(videos),
                source_fps=source_fps,
                infer_ms=last_infer_ms,
                paused=paused,
            )
            cv2.imshow(config.window_name, annotated)

            wait_ms = 1
            target_fps = config.playback_fps or source_fps
            if target_fps > 0:
                wait_ms = max(1, int(round(1000.0 / target_fps)))

            key = cv2.waitKey(wait_ms) & 0xFF
            if key == ord("q"):
                break
            if key == ord("n"):
                video_index = next_video_index(library, class_index, video_index)
                cap.release()
                cap = None
                paused = False
                continue
            if key == ord("c"):
                class_index, video_index = next_class_indices(library, class_index)
                cap.release()
                cap = None
                paused = False
                continue
            if key == ord("r"):
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                paused = False
                continue
            if key == ord(" "):
                paused = not paused
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = load_review_config(Path(args.config).resolve())
        return run_review(
            config=config,
            weights=args.weights,
            conf=args.conf,
            iou=args.iou,
            start_class=args.class_name,
            start_video=args.video_name,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
