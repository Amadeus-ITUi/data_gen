"""Flatten instance subfolders into class folders.

Before:
  source/ambulance/ambulance_001/*.png
                    /ambulance_002/*.png
After:
  output/ambulance/*.png  (all images from all instances)

Usage:
  python -m src.flatten_instances --source work/roi_crop
  python -m src.flatten_instances --source work/roi_crop --output work/roi_crop_flat
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

EXPECTED_CLASSES = {"ambulance", "armored_car", "bomb", "gun", "medicine", "telescope"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Flatten instance subfolders into class folders.")
    parser.add_argument("--source", required=True, help="Path to source folder containing 6 class subdirs.")
    parser.add_argument("--output", default=None, help="Output folder. Default: <source>_flatten")
    parser.add_argument("--symlink", action="store_true", help="Create symlinks instead of copying.")
    parser.add_argument("--dry-run", action="store_true", help="Count only, do not copy.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    src_root = Path(args.source).resolve()
    if not src_root.is_dir():
        print(f"ERROR: source not found: {src_root}", file=sys.stderr)
        return 1

    out_root = Path(args.output).resolve() if args.output else src_root.parent / f"{src_root.name}_flatten"

    # Find class dirs
    class_dirs = {}
    for d in sorted(src_root.iterdir()):
        if d.is_dir() and d.name in EXPECTED_CLASSES:
            class_dirs[d.name] = d

    if not class_dirs:
        print(f"ERROR: no class folders found in {src_root}", file=sys.stderr)
        print(f"  Expected: {sorted(EXPECTED_CLASSES)}", file=sys.stderr)
        return 1

    print(f"Source: {src_root}")
    print(f"Output: {out_root}")
    print(f"Classes found: {len(class_dirs)}")
    print()

    total = 0
    for cls in sorted(class_dirs):
        cls_src = class_dirs[cls]
        cls_out = out_root / cls

        # Count instance dirs
        inst_dirs = sorted(d for d in cls_src.iterdir() if d.is_dir())
        if not inst_dirs:
            print(f"  {cls}: 0 instances — skipped")
            continue

        cls_total = 0
        for inst_dir in inst_dirs:
            images = sorted(
                p for p in inst_dir.glob("*")
                if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg")
            )
            if not images:
                continue

            if not args.dry_run:
                cls_out.mkdir(parents=True, exist_ok=True)

            for img_path in images:
                dest = cls_out / img_path.name
                if args.dry_run:
                    cls_total += 1
                    continue
                if dest.exists():
                    # Instance name is in filename, collisions are very unlikely
                    print(f"    WARNING: overwriting duplicate {img_path.name}")
                try:
                    if args.symlink:
                        dest.symlink_to(img_path.resolve())
                    else:
                        shutil.copy2(str(img_path), str(dest))
                    cls_total += 1
                except OSError as e:
                    print(f"    ERROR: {img_path.name}: {e}", file=sys.stderr)

        total += cls_total
        inst_count = len(inst_dirs)
        print(f"  {cls}: {cls_total} images from {inst_count} instances")

    print(f"\nTotal: {total} images")
    if args.dry_run:
        print("(dry run — no files copied)")
    else:
        print(f"Output: {out_root}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
