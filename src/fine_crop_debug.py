from __future__ import annotations

import argparse
import io
import sys
import threading
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_file

app = Flask(__name__)

WORK_ROOT: Path | None = None
CATEGORIES: list[str] = []
INSTANCES: dict[str, list[str]] = {}
IMAGES: dict[str, dict[str, list[str]]] = {}


# ---------------------------------------------------------------------------
#  CV: find the largest red region in an image
# ---------------------------------------------------------------------------

def detect_red_roi(
    image: np.ndarray,
    h_span: int = 12,
    s_min: int = 50,
    v_min: int = 50,
    close_iter: int = 1,
    open_iter: int = 1,
) -> dict | None:
    """Find the largest red-ish rectangular region. Returns {x,y,w,h,area} or None."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, s_min, v_min], dtype=np.uint8)
    upper_red1 = np.array([h_span, 255, 255], dtype=np.uint8)
    lower_red2 = np.array([180 - h_span, s_min, v_min], dtype=np.uint8)
    upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((3, 3), np.uint8)
    if close_iter > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=close_iter)
    if open_iter > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=open_iter)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area <= 50:
        return None

    x, y, w, h = cv2.boundingRect(largest)
    return {"x": int(x), "y": int(y), "w": int(w), "h": int(h), "area": float(area)}


def detect_red_roi_v2(
    image: np.ndarray,
    h_span: int = 12,
    s_min: int = 50,
    v_min: int = 50,
    close_iter: int = 1,
    open_iter: int = 1,
    ratio_w: float = 1.0,
    ratio_h: float = 0.42,
    offset_ratio: float = 0.417,
) -> dict | None:
    """Bottom-anchored ROI with lift offset.

    Physical model: red rectangle (120x50mm) sits below a 120x120mm square.
    The contour bounding rect is found within the square; its bottom edge
    is lifted by N = bw * offset_ratio pixels, then the ROI is drawn
    upward from that anchor point.

      anchor_y = (by + bh) - bw * offset_ratio    (lift from contour bottom)
      new_w = bbox.w * ratio_w
      new_h = bbox.w * ratio_h
      new_x = bbox.x + (bbox.w - new_w) / 2       (centred horizontally)
      new_y = anchor_y - new_h                     (ROI extends upward from anchor)
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, s_min, v_min], dtype=np.uint8)
    upper_red1 = np.array([h_span, 255, 255], dtype=np.uint8)
    lower_red2 = np.array([180 - h_span, s_min, v_min], dtype=np.uint8)
    upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((3, 3), np.uint8)
    if close_iter > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=close_iter)
    if open_iter > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=open_iter)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area <= 50:
        return None

    bx, by, bw, bh = cv2.boundingRect(largest)
    img_h, img_w = image.shape[:2]

    # Derive ROI: lift anchor from contour bottom, then draw upward
    lift = int(round(bw * offset_ratio))
    anchor_y = (by + bh) - lift
    new_w = max(3, int(round(bw * ratio_w)))
    new_h = max(3, int(round(bw * ratio_h)))
    new_x = int(round(bx + (bw - new_w) / 2.0))
    new_y = anchor_y - new_h

    # Clamp to image bounds
    if new_x < 0:
        new_x = 0
    if new_y < 0:
        new_y = 0
    if new_x + new_w > img_w:
        new_x = img_w - new_w
    if new_y + new_h > img_h:
        new_y = img_h - new_h

    return {
        "x": new_x, "y": new_y, "w": new_w, "h": new_h,
        "area": float(area),
        "bbox": {"x": int(bx), "y": int(by), "w": int(bw), "h": int(bh)},
        "lift": lift,
        "anchor_y": anchor_y,
    }


def make_morph_mask(
    image: np.ndarray,
    h_span: int = 12,
    s_min: int = 50,
    v_min: int = 50,
    close_iter: int = 1,
    open_iter: int = 1,
) -> np.ndarray:
    """Return the binary mask AFTER morphology (close+open), before contour extraction."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, s_min, v_min], dtype=np.uint8)
    upper_red1 = np.array([h_span, 255, 255], dtype=np.uint8)
    lower_red2 = np.array([180 - h_span, s_min, v_min], dtype=np.uint8)
    upper_red2 = np.array([180, 255, 255], dtype=np.uint8)

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((3, 3), np.uint8)
    if close_iter > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=close_iter)
    if open_iter > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=open_iter)
    return mask


# ---------------------------------------------------------------------------
#  Data discovery
# ---------------------------------------------------------------------------

def discover_structure(work_root: Path) -> tuple[list[str], dict[str, list[str]], dict[str, dict[str, list[str]]]]:
    categories: list[str] = []
    instances: dict[str, list[str]] = {}
    images: dict[str, dict[str, list[str]]] = {}

    if not work_root.exists():
        return categories, instances, images

    for cat_dir in sorted(p for p in work_root.iterdir() if p.is_dir()):
        cat = cat_dir.name
        categories.append(cat)
        instances[cat] = []
        images[cat] = {}

        for inst_dir in sorted(p for p in cat_dir.iterdir() if p.is_dir()):
            inst = inst_dir.name
            instances[cat].append(inst)

            pngs = sorted(p.name for p in inst_dir.glob("*.png") if p.is_file())
            jpgs = sorted(p.name for p in inst_dir.glob("*.jpg") if p.is_file())
            images[cat][inst] = pngs + jpgs

    return categories, instances, images


# ---------------------------------------------------------------------------
#  API routes
# ---------------------------------------------------------------------------

@app.route("/api/structure")
def api_structure():
    return jsonify({
        "categories": CATEGORIES,
        "instances": INSTANCES,
        "image_count": {
            cat: {inst: len(files) for inst, files in inst_map.items()}
            for cat, inst_map in IMAGES.items()
        },
    })


@app.route("/api/images/<category>/<instance>")
def api_instance_images(category: str, instance: str):
    files = IMAGES.get(category, {}).get(instance, [])
    return jsonify({"category": category, "instance": instance, "images": files})


@app.route("/api/image-file")
def api_image_file():
    cat = request.args.get("category", "")
    inst = request.args.get("instance", "")
    name = request.args.get("name", "")
    path = WORK_ROOT / cat / inst / name
    if not path.exists():
        return jsonify({"error": "file not found"}), 404
    return send_file(str(path), mimetype="image/png")


@app.route("/api/detect-red", methods=["POST"])
def api_detect_red():
    data = request.get_json(silent=True) or {}
    cat = data.get("category", "")
    inst = data.get("instance", "")
    name = data.get("name", "")
    h_span = int(data.get("h_span", 12))
    s_min = int(data.get("s_min", 50))
    v_min = int(data.get("v_min", 50))
    close_iter = int(data.get("close_iter", 1))
    open_iter = int(data.get("open_iter", 1))
    mode = data.get("mode", "contour")
    ratio_w = float(data.get("ratio_w", 1.0))
    ratio_h = float(data.get("ratio_h", 0.42))
    offset_ratio = float(data.get("offset_ratio", 0.417))

    path = WORK_ROOT / cat / inst / name
    if not path.exists():
        return jsonify({"error": "file not found"}), 404

    image = cv2.imread(str(path))
    if image is None:
        return jsonify({"error": "failed to read image"}), 500

    if mode == "bottom":
        result = detect_red_roi_v2(
            image, h_span=h_span, s_min=s_min, v_min=v_min,
            close_iter=close_iter, open_iter=open_iter,
            ratio_w=ratio_w, ratio_h=ratio_h, offset_ratio=offset_ratio,
        )
    else:
        result = detect_red_roi(
            image, h_span=h_span, s_min=s_min, v_min=v_min,
            close_iter=close_iter, open_iter=open_iter,
        )
    h, w = image.shape[:2]
    return jsonify({"roi": result, "image_width": w, "image_height": h})


@app.route("/api/morph-mask", methods=["POST"])
def api_morph_mask():
    data = request.get_json(silent=True) or {}
    cat = data.get("category", "")
    inst = data.get("instance", "")
    name = data.get("name", "")
    h_span = int(data.get("h_span", 12))
    s_min = int(data.get("s_min", 50))
    v_min = int(data.get("v_min", 50))
    close_iter = int(data.get("close_iter", 1))
    open_iter = int(data.get("open_iter", 1))

    path = WORK_ROOT / cat / inst / name
    if not path.exists():
        return jsonify({"error": "file not found"}), 404

    image = cv2.imread(str(path))
    if image is None:
        return jsonify({"error": "failed to read image"}), 500

    morph = make_morph_mask(image, h_span=h_span, s_min=s_min, v_min=v_min, close_iter=close_iter, open_iter=open_iter)
    ok, buf = cv2.imencode(".png", morph)
    if not ok:
        return jsonify({"error": "encode failed"}), 500
    return send_file(io.BytesIO(buf.tobytes()), mimetype="image/png")


# ---------------------------------------------------------------------------
#  Batch export
# ---------------------------------------------------------------------------

_batch_state: dict = {"running": False, "total": 0, "done": 0, "failed": 0, "status": "idle"}


def _process_image(img_path: Path, out_dir: Path, params: dict) -> bool:
    """Process one image: detect ROI, crop, save. Returns True on success."""
    image = cv2.imread(str(img_path))
    if image is None:
        return False

    mode = params.get("mode", "contour")
    if mode == "bottom":
        result = detect_red_roi_v2(
            image,
            h_span=params.get("h_span", 12),
            s_min=params.get("s_min", 50),
            v_min=params.get("v_min", 50),
            close_iter=params.get("close_iter", 1),
            open_iter=params.get("open_iter", 1),
            ratio_w=params.get("ratio_w", 1.0),
            ratio_h=params.get("ratio_h", 0.42),
            offset_ratio=params.get("offset_ratio", 0.417),
        )
    else:
        result = detect_red_roi(
            image,
            h_span=params.get("h_span", 12),
            s_min=params.get("s_min", 50),
            v_min=params.get("v_min", 50),
            close_iter=params.get("close_iter", 1),
            open_iter=params.get("open_iter", 1),
        )

    if result is None:
        return False

    x, y, w, h = result["x"], result["y"], result["w"], result["h"]
    if x < 0 or y < 0 or w <= 0 or h <= 0:
        return False

    x = max(0, x)
    y = max(0, y)
    w = min(w, image.shape[1] - x)
    h = min(h, image.shape[0] - y)
    if w <= 0 or h <= 0:
        return False

    crop = image[y:y + h, x:x + w]
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = img_path.stem
    out_path = out_dir / f"{stem}__fine.png"
    cv2.imwrite(str(out_path), crop)
    return True


def _batch_worker(params: dict) -> None:
    global _batch_state, WORK_ROOT
    _batch_state["running"] = True
    _batch_state["done"] = 0
    _batch_state["failed"] = 0
    _batch_state["total"] = 0

    output_root = WORK_ROOT.parent / "roi_crop"

    try:
        # Count total
        for cat in CATEGORIES:
            for inst in INSTANCES.get(cat, []):
                _batch_state["total"] += len(IMAGES.get(cat, {}).get(inst, []))

        _batch_state["status"] = "processing"

        for cat in CATEGORIES:
            for inst in INSTANCES.get(cat, []):
                img_files = IMAGES.get(cat, {}).get(inst, [])
                out_dir = output_root / cat / inst
                for fname in img_files:
                    img_path = WORK_ROOT / cat / inst / fname
                    try:
                        ok = _process_image(img_path, out_dir, params)
                        if ok:
                            _batch_state["done"] += 1
                        else:
                            _batch_state["failed"] += 1
                    except Exception:
                        _batch_state["failed"] += 1
        _batch_state["status"] = "done"
    except Exception as e:
        _batch_state["status"] = f"error: {e}"
    finally:
        _batch_state["running"] = False


@app.route("/api/batch-export", methods=["POST"])
def api_batch_export():
    global _batch_state
    if _batch_state["running"]:
        return jsonify({"error": "batch already running"}), 409

    data = request.get_json(silent=True) or {}
    params = {
        "mode": data.get("mode", "contour"),
        "h_span": int(data.get("h_span", 12)),
        "s_min": int(data.get("s_min", 50)),
        "v_min": int(data.get("v_min", 50)),
        "close_iter": int(data.get("close_iter", 1)),
        "open_iter": int(data.get("open_iter", 1)),
        "ratio_w": float(data.get("ratio_w", 1.0)),
        "ratio_h": float(data.get("ratio_h", 0.42)),
        "offset_ratio": float(data.get("offset_ratio", 0.417)),
    }

    thread = threading.Thread(target=_batch_worker, args=(params,), daemon=True)
    thread.start()
    return jsonify({"status": "started"})


@app.route("/api/batch-progress")
def api_batch_progress():
    return jsonify(_batch_state)


# ---------------------------------------------------------------------------
#  Frontend (embedded HTML)
# ---------------------------------------------------------------------------

INDEX_HTML = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Fine Crop Debug</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', sans-serif; background: #1a1a2e; color: #eee; height: 100vh; display: flex; flex-direction: column; }

  /* ── header bar ── */
  #topbar { background: #16213e; padding: 6px 12px; display: flex; align-items: center; gap: 10px; flex-shrink: 0; flex-wrap: wrap; }
  #topbar h1 { font-size: 15px; color: #e94560; white-space: nowrap; }
  #topbar label { font-size: 11px; color: #999; }
  #topbar select, #topbar button { padding: 3px 7px; border-radius: 3px; border: 1px solid #444; background: #0f3460; color: #eee; font-size: 12px; cursor: pointer; }
  #topbar button:hover { background: #e94560; }
  .nav-group { display: flex; gap: 4px; align-items: center; }
  .nav-group button { min-width: 28px; }

  /* ── hsv sliders row ── */
  #slider-bar { background: #1a2744; padding: 4px 12px; display: flex; align-items: center; gap: 14px; flex-shrink: 0; flex-wrap: wrap; border-bottom: 1px solid #2a3a5c; }
  #slider-bar label { font-size: 11px; color: #ccc; white-space: nowrap; }
  #slider-bar input[type=range] { width: 100px; accent-color: #e94560; }
  #slider-bar .val { display: inline-block; width: 32px; text-align: right; font-size: 11px; color: #e94560; }
  #slider-bar button { padding: 3px 10px; border-radius: 3px; border: 1px solid #444; background: #0f3460; color: #eee; font-size: 12px; cursor: pointer; }
  #slider-bar button:hover { background: #e94560; }

  /* ── 2x2 panel grid ── */
  main { display: grid; grid-template-columns: 1fr 1fr; grid-template-rows: 1fr 1fr; flex: 1; gap: 3px; padding: 3px; overflow: hidden; min-height: 0; }
  .panel { display: flex; flex-direction: column; background: #16213e; border-radius: 5px; overflow: hidden; min-width: 0; }
  .panel-header { padding: 4px 10px; font-size: 12px; font-weight: 600; background: #0f3460; flex-shrink: 0; display: flex; justify-content: space-between; align-items: center; }
  .panel-header .info { font-size: 10px; color: #888; font-weight: 400; }
  .canvas-wrap { flex: 1; position: relative; overflow: auto; display: flex; justify-content: center; align-items: center; padding: 6px; }
  canvas { border: 1px solid #333; max-width: 100%; max-height: 100%; }
  #orig-canvas { cursor: crosshair; }
  .status-bar { background: #0f3460; padding: 3px 10px; font-size: 11px; color: #999; flex-shrink: 0; }
</style>
</head>
<body>

<!-- ── top bar: navigation ── -->
<div id="topbar">
  <h1>Fine Crop Debug</h1>
  <label>Category</label><select id="cat-select"></select>
  <label>Instance</label><select id="inst-select"></select>
  <label>Image</label><select id="img-select" style="min-width:180px;"></select>
  <div class="nav-group">
    <button id="btn-prev" title="prev">&lt;</button>
    <button id="btn-next" title="next">&gt;</button>
    <span id="img-counter" style="font-size:11px;color:#888;">0/0</span>
  </div>
  <label style="margin-left:6px;"><input type="checkbox" id="chk-auto" checked> Auto</label>
  <button id="btn-detect">Detect</button>
  <button id="btn-clear-roi">Clear ROI</button>
  <button id="btn-save-roi">Save Crop</button>
  <button id="btn-batch" style="background:#c0392b;">Batch Export</button>
</div>

<!-- ── HSV sliders ── -->
<div id="slider-bar">
  <label>H_span <input type="range" id="sl-hspan" min="2" max="40" value="12" step="1"><span class="val" id="val-hspan">12</span></label>
  <label>S_min <input type="range" id="sl-smin" min="0" max="255" value="50" step="1"><span class="val" id="val-smin">50</span></label>
  <label>V_min <input type="range" id="sl-vmin" min="0" max="255" value="50" step="1"><span class="val" id="val-vmin">50</span></label>
  <label>Close <input type="range" id="sl-close" min="0" max="4" value="1" step="1"><span class="val" id="val-close">1</span></label>
  <label>Open <input type="range" id="sl-open" min="0" max="4" value="1" step="1"><span class="val" id="val-open">1</span></label>
  <button id="btn-reset-hsv">Reset HSV</button>
  <span style="font-size:10px;color:#666;">| red ranges: [0,&thinsp;<span id="lbl-lo">12</span>] &amp; [<span id="lbl-hi">168</span>,&thinsp;180]</span>
  <label style="margin-left:8px;">Mode <select id="sel-mode"><option value="contour">Contour</option><option value="bottom">Bottom-Anchor</option></select></label>
  <label>R_W <input type="range" id="sl-rw" min="0.3" max="2.0" value="1.0" step="0.05"><span class="val" id="val-rw">1.0</span></label>
  <label>R_H <input type="range" id="sl-rh" min="0.1" max="1.5" value="0.42" step="0.01"><span class="val" id="val-rh">0.42</span></label>
  <label>Lift <input type="range" id="sl-lift" min="0.0" max="1.0" value="0.417" step="0.005"><span class="val" id="val-lift">0.417</span></label>
</div>

<!-- ── 2x2 panels ── -->
<main>
  <div class="panel">
    <div class="panel-header">Original <span class="info" id="info-orig"></span></div>
    <div class="canvas-wrap"><canvas id="orig-canvas"></canvas></div>
  </div>
  <div class="panel">
    <div class="panel-header">ROI Crop <span class="info" id="info-roi"></span></div>
    <div class="canvas-wrap"><canvas id="roi-canvas"></canvas></div>
  </div>
  <div class="panel">
    <div class="panel-header">Raw HSV Mask <span class="info" id="info-mask"></span></div>
    <div class="canvas-wrap"><canvas id="mask-canvas"></canvas></div>
  </div>
  <div class="panel">
    <div class="panel-header">Morph Mask (after close+open) <span class="info" id="info-morph"></span></div>
    <div class="canvas-wrap"><canvas id="morph-canvas"></canvas></div>
  </div>
</main>

<div class="status-bar" id="status">Loading...</div>
<div id="batch-progress" style="display:none; background:#c0392b; padding:4px 12px; font-size:11px; color:#fff; flex-shrink:0;"><span id="batch-msg"></span> <span id="batch-pct"></span></div>

<script>
// ═══════════════════════════════════════════════════════════════════════
//  State
// ═══════════════════════════════════════════════════════════════════════
let categories = [], instances = {}, imageMap = {};
let curCat = '', curInst = '', curImg = '';
let curImgIdx = 0, curImgList = [];
let origImg = null;
let roi = null;
let drawing = false, drawStart = {x:0, y:0};

// HSV params (synced to sliders)
let hsv = { h_span: 12, s_min: 50, v_min: 50, close_iter: 1, open_iter: 1, mode: "contour", ratio_w: 1.0, ratio_h: 0.42, offset_ratio: 0.417 };

// ═══════════════════════════════════════════════════════════════════════
//  DOM refs
// ═══════════════════════════════════════════════════════════════════════
const $ = id => document.getElementById(id);
const catSel = $('cat-select'), instSel = $('inst-select'), imgSel = $('img-select');
const btnPrev = $('btn-prev'), btnNext = $('btn-next'), imgCounter = $('img-counter');
const btnDetect = $('btn-detect'), btnClear = $('btn-clear-roi'), btnSave = $('btn-save-roi');
const chkAuto = $('chk-auto');
const slHspan = $('sl-hspan'), slSmin = $('sl-smin'), slVmin = $('sl-vmin'), slClose = $('sl-close'), slOpen = $('sl-open'), slRw = $('sl-rw'), slRh = $('sl-rh'), slLift = $('sl-lift');
const selMode = $('sel-mode');
const valHspan = $('val-hspan'), valSmin = $('val-smin'), valVmin = $('val-vmin'), valClose = $('val-close'), valOpen = $('val-open'), valRw = $('val-rw'), valRh = $('val-rh'), valLift = $('val-lift');
const lblLo = $('lbl-lo'), lblHi = $('lbl-hi');
const origCanvas = $('orig-canvas'), roiCanvas = $('roi-canvas'), maskCanvas = $('mask-canvas'), morphCanvas = $('morph-canvas');
const origCtx = origCanvas.getContext('2d'), roiCtx = roiCanvas.getContext('2d'), maskCtx = maskCanvas.getContext('2d'), morphCtx = morphCanvas.getContext('2d');
const infoOrig = $('info-orig'), infoRoi = $('info-roi'), infoMask = $('info-mask'), infoMorph = $('info-morph');
const statusBar = $('status');
const btnBatch = $('btn-batch'), batchProg = $('batch-progress'), batchMsg = $('batch-msg'), batchPct = $('batch-pct');

// ═══════════════════════════════════════════════════════════════════════
//  RGB → HSV (client-side, for real-time mask)
// ═══════════════════════════════════════════════════════════════════════
function rgbToHsv(r, g, b) {
  r /= 255; g /= 255; b /= 255;
  const mx = Math.max(r, g, b), mn = Math.min(r, g, b), d = mx - mn;
  let h = 0;
  if (d !== 0) {
    if (mx === r) h = ((g - b) / d) % 6;
    else if (mx === g) h = (b - r) / d + 2;
    else h = (r - g) / d + 4;
  }
  h = Math.round(h * 30);           // 0..180  (OpenCV range)
  if (h < 0) h += 180;
  const s = mx === 0 ? 0 : Math.round((d / mx) * 255);
  const v = Math.round(mx * 255);
  return [h, s, v];
}

function inRedRange(h, s, v) {
  return (h <= hsv.h_span || h >= 180 - hsv.h_span) && s >= hsv.s_min && v >= hsv.v_min;
}

function renderMask() {
  if (!origImg) { maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height); return; }
  const w = origImg.naturalWidth, h = origImg.naturalHeight;

  // Draw image onto temp canvas to read pixels
  const tmp = document.createElement('canvas');
  tmp.width = w; tmp.height = h;
  const tctx = tmp.getContext('2d');
  tctx.drawImage(origImg, 0, 0);
  const imgData = tctx.getImageData(0, 0, w, h);
  const px = imgData.data;

  // Build mask
  const maskData = new Uint8ClampedArray(w * h * 4);
  let redCount = 0;
  for (let i = 0; i < px.length; i += 4) {
    const [hh, ss, vv] = rgbToHsv(px[i], px[i+1], px[i+2]);
    const isRed = inRedRange(hh, ss, vv);
    const val = isRed ? 255 : 0;
    maskData[i] = val;
    maskData[i+1] = val;
    maskData[i+2] = val;
    maskData[i+3] = 255;
    if (isRed) redCount++;
  }

  // Show mask, scaled to fit panel
  const wrap = maskCanvas.parentElement;
  const maxW = wrap.clientWidth - 12, maxH = wrap.clientHeight - 12;
  const scale = Math.min(maxW / w, maxH / h, 4.0);  // allow upscale for small images
  maskCanvas.width = Math.floor(w * scale);
  maskCanvas.height = Math.floor(h * scale);

  const tmp2 = document.createElement('canvas');
  tmp2.width = w; tmp2.height = h;
  const tctx2 = tmp2.getContext('2d');
  const id2 = new ImageData(maskData, w, h);
  tctx2.putImageData(id2, 0, 0);

  maskCtx.imageSmoothingEnabled = false;
  maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
  maskCtx.drawImage(tmp2, 0, 0, maskCanvas.width, maskCanvas.height);

  const pct = (100 * redCount / (w * h)).toFixed(1);
  infoMask.textContent = `${w}x${h}  red: ${redCount} px (${pct}%)`;

  // Update slider display labels
  valHspan.textContent = hsv.h_span;
  valSmin.textContent = hsv.s_min;
  valVmin.textContent = hsv.v_min;
  valClose.textContent = hsv.close_iter;
  valOpen.textContent = hsv.open_iter;
  valRw.textContent = hsv.ratio_w.toFixed(2);
  valRh.textContent = hsv.ratio_h.toFixed(2);
  valLift.textContent = hsv.offset_ratio.toFixed(3);
  lblLo.textContent = hsv.h_span;
  lblHi.textContent = 180 - hsv.h_span;
}

// ── Morph mask (server-rendered, after morphology) ──
function renderMorphMask() {
  if (!origImg) { morphCtx.clearRect(0, 0, morphCanvas.width, morphCanvas.height); return; }
  const url = '/api/morph-mask';
  fetch(url, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      category: curCat, instance: curInst, name: curImg,
      h_span: hsv.h_span, s_min: hsv.s_min, v_min: hsv.v_min, close_iter: hsv.close_iter, open_iter: hsv.open_iter, mode: hsv.mode, ratio_w: hsv.ratio_w, ratio_h: hsv.ratio_h, offset_ratio: hsv.offset_ratio,
    }),
  }).then(r => r.blob()).then(blob => {
    const img = new Image();
    img.onload = () => {
      const w = img.naturalWidth, h = img.naturalHeight;
      const wrap = morphCanvas.parentElement;
      const maxW = wrap.clientWidth - 12, maxH = wrap.clientHeight - 12;
      const scale = Math.min(maxW / w, maxH / h, 4.0);
      morphCanvas.width = Math.floor(w * scale);
      morphCanvas.height = Math.floor(h * scale);
      morphCtx.imageSmoothingEnabled = false;
      morphCtx.clearRect(0, 0, morphCanvas.width, morphCanvas.height);
      morphCtx.drawImage(img, 0, 0, morphCanvas.width, morphCanvas.height);
      infoMorph.textContent = w + 'x' + h;
    };
    img.src = URL.createObjectURL(blob);
  });
}

// ═══════════════════════════════════════════════════════════════════════
//  Init
// ═══════════════════════════════════════════════════════════════════════
async function init() {
  status('Loading...');
  const resp = await fetch('/api/structure');
  const data = await resp.json();
  categories = data.categories;
  instances = data.instances;
  imageMap = data.image_count;

  catSel.innerHTML = categories.map(c => `<option value="${c}">${c}</option>`).join('');
  if (categories.length > 0) {
    curCat = categories[0]; catSel.value = curCat;
    populateInstances();
  }
  status('Ready.');
}

function populateInstances() {
  const insts = instances[curCat] || [];
  instSel.innerHTML = insts.map(i => `<option value="${i}">${i}</option>`).join('');
  if (insts.length > 0) {
    curInst = insts[0]; instSel.value = curInst;
    populateImages();
  }
}

function populateImages() {
  fetch(`/api/images/${encodeURIComponent(curCat)}/${encodeURIComponent(curInst)}`)
    .then(r => r.json())
    .then(data => {
      curImgList = data.images || [];
      imgSel.innerHTML = curImgList.map((f, i) => `<option value="${i}">${f}</option>`).join('');
      if (curImgList.length > 0) {
        curImgIdx = 0; imgSel.value = '0'; curImg = curImgList[0];
        loadImage();
      }
    });
}

// ═══════════════════════════════════════════════════════════════════════
//  Events
// ═══════════════════════════════════════════════════════════════════════
catSel.addEventListener('change', () => { curCat = catSel.value; populateInstances(); });
instSel.addEventListener('change', () => { curInst = instSel.value; populateImages(); });
imgSel.addEventListener('change', () => {
  curImgIdx = parseInt(imgSel.value); curImg = curImgList[curImgIdx];
  loadImage();
  if (chkAuto.checked) runDetect();
});
btnPrev.addEventListener('click', () => { if (curImgIdx > 0) { curImgIdx--; selectImage(); } });
btnNext.addEventListener('click', () => { if (curImgIdx < curImgList.length - 1) { curImgIdx++; selectImage(); } });
btnDetect.addEventListener('click', runDetect);
btnClear.addEventListener('click', () => { roi = null; renderAll(); });
btnSave.addEventListener('click', saveRoiCrop);
$('btn-reset-hsv').addEventListener('click', () => {
  slHspan.value = 12; slSmin.value = 50; slVmin.value = 50; slClose.value = 1; slOpen.value = 1; slRw.value = 1.0; slRh.value = 0.42; slLift.value = 0.417; selMode.value = 'contour';
  readSliders();
});

// HSV sliders → re-render mask + optionally re-detect
[slHspan, slSmin, slVmin, slClose, slOpen, slRw, slRh, slLift].forEach(sl => {
  sl.addEventListener('input', () => {
    readSliders();
    renderMask();
  });
  sl.addEventListener('change', () => {
    readSliders();
    renderMask();
    renderMorphMask();
    if (chkAuto.checked && origImg) runDetect();
  });
});
selMode.addEventListener('change', () => {
  readSliders();
  if (chkAuto.checked && origImg) runDetect();
});

function readSliders() {
  hsv.h_span = parseInt(slHspan.value);
  hsv.s_min = parseInt(slSmin.value);
  hsv.v_min = parseInt(slVmin.value);
  hsv.close_iter = parseInt(slClose.value);
  hsv.open_iter = parseInt(slOpen.value);
  hsv.mode = selMode.value;
  hsv.ratio_w = parseFloat(slRw.value);
  hsv.ratio_h = parseFloat(slRh.value);
  hsv.offset_ratio = parseFloat(slLift.value);
}

function selectImage() {
  imgSel.value = String(curImgIdx); curImg = curImgList[curImgIdx];
  loadImage();
  if (chkAuto.checked) runDetect();
}

// keyboard
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'SELECT' || e.target.tagName === 'INPUT') return;
  if (e.key === 'ArrowLeft' && curImgIdx > 0) { curImgIdx--; selectImage(); }
  if (e.key === 'ArrowRight' && curImgIdx < curImgList.length - 1) { curImgIdx++; selectImage(); }
  if (e.key === 'd' || e.key === 'D') runDetect();
  if (e.key === 'Escape') { roi = null; renderAll(); }
});

// ═══════════════════════════════════════════════════════════════════════
//  Load & render
// ═══════════════════════════════════════════════════════════════════════
function loadImage() {
  roi = null;
  if (!curImg) { clearAll(); return; }
  const url = `/api/image-file?category=${encodeURIComponent(curCat)}&instance=${encodeURIComponent(curInst)}&name=${encodeURIComponent(curImg)}`;
  const img = new Image();
  img.onload = () => {
    origImg = img;
    renderAll();
    updateCounter();
  };
  img.src = url;
}

function renderAll() {
  renderOriginal();
  renderRoiCrop();
  renderMask();
  renderMorphMask();
}

function clearAll() {
  origCtx.clearRect(0, 0, origCanvas.width, origCanvas.height);
  roiCtx.clearRect(0, 0, roiCanvas.width, roiCanvas.height);
  maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
  morphCtx.clearRect(0, 0, morphCanvas.width, morphCanvas.height);
  origImg = null; roi = null;
}

function renderOriginal() {
  if (!origImg) return;
  const w = origImg.naturalWidth, h = origImg.naturalHeight;
  const wrap = origCanvas.parentElement;
  const maxW = wrap.clientWidth - 12, maxH = wrap.clientHeight - 12;
  const scale = Math.min(maxW / w, maxH / h, 4.0);
  origCanvas.width = Math.floor(w * scale);
  origCanvas.height = Math.floor(h * scale);

  origCtx.imageSmoothingEnabled = false;
  origCtx.clearRect(0, 0, origCanvas.width, origCanvas.height);
  origCtx.drawImage(origImg, 0, 0, origCanvas.width, origCanvas.height);

  if (roi) {
    // Show contour bbox (red dashed) + anchor line in bottom-anchor mode
    if (roi.bbox) {
      const bx = roi.bbox.x * scale, by = roi.bbox.y * scale, bw = roi.bbox.w * scale, bh = roi.bbox.h * scale;
      origCtx.strokeStyle = '#ff4444'; origCtx.lineWidth = 1;
      origCtx.setLineDash([3, 3]);
      origCtx.strokeRect(bx, by, bw, bh);
      origCtx.setLineDash([]);
      // Anchor line (cyan) at the lifted bottom
      if (roi.anchor_y !== undefined) {
        const ay = roi.anchor_y * scale;
        origCtx.strokeStyle = '#00ffff'; origCtx.lineWidth = 1;
        origCtx.setLineDash([2, 4]);
        origCtx.beginPath();
        origCtx.moveTo(0, ay);
        origCtx.lineTo(origCanvas.width, ay);
        origCtx.stroke();
        origCtx.setLineDash([]);
      }
    }
    const rx = roi.x * scale, ry = roi.y * scale, rw = roi.w * scale, rh = roi.h * scale;
    origCtx.strokeStyle = '#00ff00'; origCtx.lineWidth = 2;
    origCtx.strokeRect(rx, ry, rw, rh);
    origCtx.fillStyle = 'rgba(0,255,0,0.15)'; origCtx.fillRect(rx, ry, rw, rh);
  }
  infoOrig.textContent = `${w}x${h}`;
}

function renderRoiCrop() {
  if (!origImg) { roiCtx.clearRect(0, 0, roiCanvas.width, roiCanvas.height); return; }
  if (roi) {
    roiCanvas.width = roi.w; roiCanvas.height = roi.h;
    roiCtx.imageSmoothingEnabled = false;
    roiCtx.drawImage(origImg, roi.x, roi.y, roi.w, roi.h, 0, 0, roi.w, roi.h);
  } else {
    roiCanvas.width = 100; roiCanvas.height = 100;
    roiCtx.clearRect(0, 0, 100, 100);
  }
  infoRoi.textContent = roi ? `${roi.w}x${roi.h} @ (${roi.x},${roi.y})` + (roi.bbox ? ' [bottom]' : ' [contour]') : 'no ROI';
}

function updateCounter() {
  imgCounter.textContent = `${curImgIdx + 1}/${curImgList.length}`;
}

// ═══════════════════════════════════════════════════════════════════════
//  Manual ROI drawing
// ═══════════════════════════════════════════════════════════════════════
origCanvas.addEventListener('mousedown', e => {
  if (!origImg) return;
  const rect = origCanvas.getBoundingClientRect();
  const sx = origImg.naturalWidth / origCanvas.width;
  const sy = origImg.naturalHeight / origCanvas.height;
  drawStart = { x: Math.round((e.clientX - rect.left) * sx), y: Math.round((e.clientY - rect.top) * sy) };
  drawing = true;
});

origCanvas.addEventListener('mousemove', e => {
  if (!drawing || !origImg) return;
  const rect = origCanvas.getBoundingClientRect();
  const sx = origImg.naturalWidth / origCanvas.width;
  const sy = origImg.naturalHeight / origCanvas.height;
  const cx = Math.round((e.clientX - rect.left) * sx), cy = Math.round((e.clientY - rect.top) * sy);
  const x = Math.min(drawStart.x, cx), y = Math.min(drawStart.y, cy);
  const w = Math.abs(cx - drawStart.x), h = Math.abs(cy - drawStart.y);
  renderOriginal();
  const scale = origCanvas.width / origImg.naturalWidth;
  origCtx.strokeStyle = '#ff0'; origCtx.lineWidth = 2;
  origCtx.setLineDash([4, 4]);
  origCtx.strokeRect(x * scale, y * scale, w * scale, h * scale);
  origCtx.setLineDash([]);
});

origCanvas.addEventListener('mouseup', e => {
  if (!drawing || !origImg) return;
  drawing = false;
  const rect = origCanvas.getBoundingClientRect();
  const sx = origImg.naturalWidth / origCanvas.width;
  const sy = origImg.naturalHeight / origCanvas.height;
  const cx = Math.round((e.clientX - rect.left) * sx), cy = Math.round((e.clientY - rect.top) * sy);
  const x = Math.min(drawStart.x, cx), y = Math.min(drawStart.y, cy);
  const w = Math.abs(cx - drawStart.x), h = Math.abs(cy - drawStart.y);
  if (w > 3 && h > 3) { roi = {x, y, w, h}; }
  renderAll();
});

// ═══════════════════════════════════════════════════════════════════════
//  Detect red (server-side, with current HSV params)
// ═══════════════════════════════════════════════════════════════════════
async function runDetect() {
  if (!curImg) return;
  status('Detecting...');
  const resp = await fetch('/api/detect-red', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      category: curCat, instance: curInst, name: curImg,
      h_span: hsv.h_span, s_min: hsv.s_min, v_min: hsv.v_min, close_iter: hsv.close_iter, open_iter: hsv.open_iter, mode: hsv.mode, ratio_w: hsv.ratio_w, ratio_h: hsv.ratio_h, offset_ratio: hsv.offset_ratio,
    }),
  });
  const data = await resp.json();
  if (data.roi) {
    roi = data.roi;
    status(`Detected: (${roi.x},${roi.y}) ${roi.w}x${roi.h}  area=${roi.area.toFixed(0)}`);
  } else {
    status('No red region found.');
  }
  renderAll();
}

// ═══════════════════════════════════════════════════════════════════════
//  Save ROI crop
// ═══════════════════════════════════════════════════════════════════════
function saveRoiCrop() {
  if (!origImg || !roi) { status('No ROI to save.'); return; }
  const c = document.createElement('canvas');
  c.width = roi.w; c.height = roi.h;
  c.getContext('2d').drawImage(origImg, roi.x, roi.y, roi.w, roi.h, 0, 0, roi.w, roi.h);
  c.toBlob(blob => {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = curImg.replace(/\.\w+$/, '') + '__fine.png';
    a.click(); URL.revokeObjectURL(url);
    status(`Saved: ${a.download}`);
  }, 'image/png');
}

function status(msg) { statusBar.textContent = msg; }

// ═══════════════════════════════════════════════════════════════════════
//  Start
// ═══════════════════════════════════════════════════════════════════════
// batch export
function startBatch() {
  if (!confirm('Export ROI crops for ALL ' + categories.length + ' categories, all instances? This may take a while.')) return;
  batchProg.style.display = 'block';
  batchMsg.textContent = 'Starting batch export...';
  batchPct.textContent = '';
  btnBatch.disabled = true;
  fetch('/api/batch-export', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      mode: hsv.mode, h_span: hsv.h_span, s_min: hsv.s_min, v_min: hsv.v_min,
      close_iter: hsv.close_iter, open_iter: hsv.open_iter,
      ratio_w: hsv.ratio_w, ratio_h: hsv.ratio_h, offset_ratio: hsv.offset_ratio,
    }),
  }).then(r => r.json()).then(data => {
    if (data.error) { batchMsg.textContent = data.error; btnBatch.disabled = false; return; }
    pollProgress();
  });
}
function pollProgress() {
  fetch('/api/batch-progress').then(r => r.json()).then(data => {
    if (!data.running && data.status === 'done') {
      batchMsg.textContent = 'Done: ' + data.done + ' OK, ' + data.failed + ' failed, ' + data.total + ' total.';
      batchPct.textContent = '';
      btnBatch.disabled = false;
      batchProg.style.background = '#27ae60';
      return;
    }
    if (!data.running && data.status !== 'processing') {
      batchMsg.textContent = data.status;
      btnBatch.disabled = false;
      return;
    }
    const pct = data.total > 0 ? (100 * (data.done + data.failed) / data.total).toFixed(1) : 0;
    batchMsg.textContent = data.done + ' OK / ' + data.failed + ' failed / ' + data.total + ' total';
    batchPct.textContent = pct + '%';
    setTimeout(pollProgress, 1000);
  }).catch(() => { setTimeout(pollProgress, 2000); });
}
btnBatch.addEventListener('click', startBatch);

init();
</script>
</body>
</html>"""


@app.route("/")
def index():
    return INDEX_HTML


# ---------------------------------------------------------------------------
#  CLI entry
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fine crop debugging platform.")
    parser.add_argument("--work-root", default="work/rough_crop", help="Path to rough_crop directory.")
    parser.add_argument("--port", type=int, default=5000, help="Server port.")
    parser.add_argument("--host", default="127.0.0.1", help="Server host.")
    return parser


def main(argv: list[str] | None = None) -> int:
    global WORK_ROOT, CATEGORIES, INSTANCES, IMAGES

    parser = build_parser()
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parent.parent
    WORK_ROOT = (project_root / args.work_root).resolve()
    if not WORK_ROOT.exists():
        print(f"ERROR: work root not found: {WORK_ROOT}", file=sys.stderr)
        return 1

    CATEGORIES, INSTANCES, IMAGES = discover_structure(WORK_ROOT)
    total_instances = sum(len(v) for v in INSTANCES.values())
    total_images = sum(len(files) for inst_map in IMAGES.values() for files in inst_map.values())
    print(f"Data root : {WORK_ROOT}")
    print(f"Categories: {len(CATEGORIES)}")
    print(f"Instances : {total_instances}")
    print(f"Images    : {total_images}")
    print(f"Server    : http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")

    app.run(host=args.host, port=args.port, debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
