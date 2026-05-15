from __future__ import annotations

import argparse
import sys
import threading
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request

app = Flask(__name__)

WORK_ROOT: Path | None = None
EXPECTED_CLASSES = {"ambulance", "armored_car", "bomb", "gun", "medicine", "telescope"}

_batch_state: dict = {"running": False, "total": 0, "done": 0, "failed": 0, "status": "idle"}

# Pad color: neutral gray (BGR)
PAD_COLOR = (128, 128, 128)


# ---------------------------------------------------------------------------
#  Discovery (same logic as sampler)
# ---------------------------------------------------------------------------

def find_source_folders(work_root: Path) -> list[dict]:
    results: list[dict] = []
    if not work_root.exists():
        return results
    for d in sorted(work_root.iterdir()):
        if not d.is_dir():
            continue
        subdirs = {p.name for p in d.iterdir() if p.is_dir()}
        if EXPECTED_CLASSES.issubset(subdirs):
            total = 0
            for cls in sorted(EXPECTED_CLASSES):
                cls_dir = d / cls
                for inst_dir in sorted(cls_dir.iterdir()):
                    if inst_dir.is_dir():
                        for f in inst_dir.glob("*"):
                            if f.is_file() and f.suffix.lower() in (".png", ".jpg", ".jpeg"):
                                total += 1
            results.append({"name": d.name, "path": str(d.resolve()), "total_images": total})
    return results


# ---------------------------------------------------------------------------
#  Image processing
# ---------------------------------------------------------------------------

def resize_to_fit(image: np.ndarray, target: int) -> np.ndarray:
    """Scale image to fit within target×target, keeping aspect ratio.
    Pad with gray to exactly target×target. Returns BGR image."""
    h, w = image.shape[:2]
    scale = target / max(h, w)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create gray canvas and paste resized image centered
    canvas = np.full((target, target, 3), PAD_COLOR, dtype=np.uint8)
    x_off = (target - new_w) // 2
    y_off = (target - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


# ---------------------------------------------------------------------------
#  API routes
# ---------------------------------------------------------------------------

@app.route("/api/sources")
def api_sources():
    sources = find_source_folders(WORK_ROOT)
    return jsonify([{
        "name": s["name"],
        "total_images": s["total_images"],
    } for s in sources])


@app.route("/api/execute", methods=["POST"])
def api_execute():
    global _batch_state
    if _batch_state["running"]:
        return jsonify({"error": "batch already running"}), 409

    data = request.get_json(silent=True) or {}
    source_name = data.get("source", "")
    target_size = int(data.get("target_size", 64))

    sources = find_source_folders(WORK_ROOT)
    source = next((s for s in sources if s["name"] == source_name), None)
    if source is None:
        return jsonify({"error": f"source not found: {source_name}"}), 404

    timestamp = datetime.now().strftime("%m%d")
    out_name = f"{source_name}_size_{timestamp}_{target_size}"
    out_root = WORK_ROOT / out_name

    thread = threading.Thread(
        target=_execute_worker,
        args=(source, out_root, target_size),
        daemon=True,
    )
    thread.start()
    return jsonify({"status": "started", "output": out_name, "total": source["total_images"]})


def _execute_worker(source: dict, out_root: Path, target_size: int) -> None:
    global _batch_state
    total = source["total_images"]
    _batch_state = {"running": True, "total": total, "done": 0, "failed": 0, "status": "processing"}

    try:
        src_path = Path(source["path"])
        for cls in sorted(EXPECTED_CLASSES):
            cls_dir = src_path / cls
            if not cls_dir.is_dir():
                continue
            for inst_dir in sorted(cls_dir.iterdir()):
                if not inst_dir.is_dir():
                    continue
                out_dir = out_root / cls / inst_dir.name
                out_dir.mkdir(parents=True, exist_ok=True)
                for f in sorted(inst_dir.glob("*")):
                    if not f.is_file() or f.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                        continue
                    try:
                        img = cv2.imread(str(f))
                        if img is None:
                            _batch_state["failed"] += 1
                            continue
                        result = resize_to_fit(img, target_size)
                        out_path = out_dir / f.name
                        cv2.imwrite(str(out_path), result)
                        _batch_state["done"] += 1
                    except Exception:
                        _batch_state["failed"] += 1
        _batch_state["status"] = "done"
    except Exception as e:
        _batch_state["status"] = f"error: {e}"
    finally:
        _batch_state["running"] = False


@app.route("/api/progress")
def api_progress():
    return jsonify(_batch_state)


# ---------------------------------------------------------------------------
#  Frontend
# ---------------------------------------------------------------------------

INDEX_HTML = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Image Resizer</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', sans-serif; background: #1a1a2e; color: #eee; padding: 20px; max-width: 800px; margin: 0 auto; }
  h1 { color: #e94560; margin-bottom: 16px; }
  .row { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; margin-bottom: 12px; }
  select, input, button { padding: 6px 12px; border-radius: 4px; border: 1px solid #444; background: #0f3460; color: #eee; font-size: 13px; cursor: pointer; }
  button:hover { background: #e94560; }
  button.primary { background: #c0392b; font-weight: 600; }
  button.primary:hover { background: #e74c3c; }
  label { font-size: 13px; color: #aaa; }
  .info-box { background: #16213e; border-radius: 6px; padding: 14px; margin-bottom: 12px; }
  #progress-bar { display: none; margin-top: 12px; }
  #progress-bar .track { background: #0f3460; border-radius: 4px; height: 20px; overflow: hidden; }
  #progress-bar .fill { background: #27ae60; height: 100%; transition: width 0.3s; border-radius: 4px; }
  #progress-bar .text { font-size: 11px; color: #aaa; margin-top: 4px; }
  .error { color: #e74c3c; }
  .success { color: #27ae60; }
</style>
</head>
<body>

<h1>Image Resizer &mdash; Scale to Fit + Gray Pad</h1>

<div class="row">
  <label>Source folder:</label>
  <select id="sel-source"></select>
  <button id="btn-refresh">Refresh</button>
</div>

<div class="row">
  <label>Target size:</label>
  <input type="number" id="inp-size" value="64" min="8" max="1024" step="8" style="width:80px;">
  <span style="color:#888;font-size:12px;">px &times; px</span>
  <button id="btn-execute" class="primary">Execute Resize</button>
</div>

<div class="info-box">
  <p style="font-size:13px;color:#aaa;">
    Scales each image to <b>fit within</b> the target square, keeping aspect ratio.
    Empty space is filled with neutral gray (128,128,128).
    Output folder: <code>work/&lt;source&gt;_size_&lt;MMDD&gt;_&lt;N&gt;</code>
  </p>
</div>

<div id="progress-bar">
  <div class="track"><div class="fill" id="prog-fill" style="width:0%"></div></div>
  <div class="text" id="prog-text"></div>
</div>

<div id="msg"></div>

<script>
const $ = id => document.getElementById(id);
const selSource = $('sel-source'), inpSize = $('inp-size');
const btnRefresh = $('btn-refresh'), btnExecute = $('btn-execute');
const progBar = $('progress-bar'), progFill = $('prog-fill'), progText = $('prog-text'), msg = $('msg');

async function loadSources() {
  const resp = await fetch('/api/sources');
  const data = await resp.json();
  selSource.innerHTML = data.map(s =>
    `<option value="${s.name}">${s.name}  (${s.total_images.toLocaleString()} imgs)</option>`
  ).join('');
  if (data.length > 0) selSource.value = data[0].name;
}
btnRefresh.addEventListener('click', loadSources);

btnExecute.addEventListener('click', async () => {
  const size = parseInt(inpSize.value);
  if (!confirm(`Resize all images in "${selSource.value}" to fit ${size}x${size} with gray padding?`)) return;
  btnExecute.disabled = true;
  const resp = await fetch('/api/execute', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({source: selSource.value, target_size: size}),
  });
  const data = await resp.json();
  if (data.error) { msg.innerHTML = `<span class="error">${data.error}</span>`; btnExecute.disabled = false; return; }
  msg.innerHTML = `<span class="success">Output: work/${data.output} (${data.total.toLocaleString()} images)</span>`;
  progBar.style.display = 'block';
  pollProgress();
});

function pollProgress() {
  fetch('/api/progress').then(r => r.json()).then(data => {
    if (!data.running) {
      progText.textContent = data.status === 'done'
        ? `Done: ${data.done.toLocaleString()} resized, ${data.failed} failed.`
        : data.status;
      progFill.style.width = '100%';
      btnExecute.disabled = false;
      return;
    }
    const pct = data.total > 0 ? (100 * (data.done + data.failed) / data.total).toFixed(1) : 0;
    progFill.style.width = pct + '%';
    progText.textContent = `${data.done.toLocaleString()} / ${data.total.toLocaleString()}  (${pct}%)`;
    setTimeout(pollProgress, 1000);
  }).catch(() => setTimeout(pollProgress, 2000));
}

loadSources();
</script>
</body>
</html>"""


@app.route("/")
def index():
    return INDEX_HTML


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Image resize + gray-pad tool.")
    parser.add_argument("--work-root", default="work", help="Path to work directory.")
    parser.add_argument("--port", type=int, default=5002, help="Server port.")
    parser.add_argument("--host", default="127.0.0.1", help="Server host.")
    return parser


def main(argv: list[str] | None = None) -> int:
    global WORK_ROOT
    parser = build_parser()
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parent.parent
    WORK_ROOT = (project_root / args.work_root).resolve()
    if not WORK_ROOT.exists():
        print(f"ERROR: work root not found: {WORK_ROOT}", file=sys.stderr)
        return 1

    sources = find_source_folders(WORK_ROOT)
    print(f"Work root : {WORK_ROOT}")
    print(f"Sources found: {len(sources)}")
    for s in sources:
        print(f"  {s['name']}: {s['total_images']:,} images")
    print(f"Server    : http://{args.host}:{args.port}")

    app.run(host=args.host, port=args.port, debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
