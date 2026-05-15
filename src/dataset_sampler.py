from __future__ import annotations

import argparse
import shutil
import sys
import threading
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, request

app = Flask(__name__)

WORK_ROOT: Path | None = None
EXPECTED_CLASSES = {"ambulance", "armored_car", "bomb", "gun", "medicine", "telescope"}

_batch_state: dict = {"running": False, "total": 0, "done": 0, "status": "idle"}


# ---------------------------------------------------------------------------
#  Discovery
# ---------------------------------------------------------------------------

def find_source_folders(work_root: Path) -> list[dict]:
    """Find subfolders of work_root that contain the 6 expected class subdirs."""
    results: list[dict] = []
    if not work_root.exists():
        return results
    for d in sorted(work_root.iterdir()):
        if not d.is_dir():
            continue
        subdirs = {p.name for p in d.iterdir() if p.is_dir()}
        if EXPECTED_CLASSES.issubset(subdirs):
            total = 0
            instances = {}
            for cls in sorted(EXPECTED_CLASSES):
                cls_dir = d / cls
                inst_map = {}
                for inst_dir in sorted(cls_dir.iterdir()):
                    if inst_dir.is_dir():
                        files = sorted(
                            p.name for p in inst_dir.glob("*")
                            if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg")
                        )
                        if files:
                            inst_map[inst_dir.name] = len(files)
                            total += len(files)
                instances[cls] = inst_map
            results.append({
                "name": d.name,
                "path": str(d.resolve()),
                "total_images": total,
                "instances": instances,
            })
    return results


# ---------------------------------------------------------------------------
#  Sampling logic
# ---------------------------------------------------------------------------

def interval_sample(items: list[str], quota: int) -> list[str]:
    """Pick `quota` items from sorted list by stride, ensuring even temporal spread."""
    if quota >= len(items):
        return list(items)
    if quota <= 0:
        return []
    step = len(items) / quota
    picked: list[str] = []
    for i in range(quota):
        idx = int(round(i * step))
        if idx >= len(items):
            idx = len(items) - 1
        picked.append(items[idx])
    # Deduplicate (may happen if step < 1)
    seen: set[str] = set()
    unique: list[str] = []
    for item in picked:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def distribute_quota(instance_counts: dict[str, int], total_quota: int) -> dict[str, int]:
    """Distribute total_quota evenly across instances, proportional to their counts."""
    inst_names = sorted(instance_counts.keys())
    if not inst_names:
        return {}

    # Step 1: give each instance min(count, floor(quota / N))
    base = total_quota // len(inst_names)
    quotas: dict[str, int] = {}
    remaining = total_quota
    for name in inst_names:
        q = min(base, instance_counts[name])
        quotas[name] = q
        remaining -= q

    # Step 2: distribute remainder to instances that still have room
    if remaining > 0:
        candidates = [n for n in inst_names if quotas[n] < instance_counts[n]]
        while remaining > 0 and candidates:
            for name in list(candidates):
                if quotas[name] < instance_counts[name]:
                    quotas[name] += 1
                    remaining -= 1
                    if remaining == 0:
                        break
            candidates = [n for n in candidates if quotas[n] < instance_counts[n]]

    return quotas


# ---------------------------------------------------------------------------
#  API routes
# ---------------------------------------------------------------------------

@app.route("/api/sources")
def api_sources():
    sources = find_source_folders(WORK_ROOT)
    return jsonify([{
        "name": s["name"],
        "total_images": s["total_images"],
        "class_count": len(s["instances"]),
    } for s in sources])


@app.route("/api/preview", methods=["POST"])
def api_preview():
    data = request.get_json(silent=True) or {}
    source_name = data.get("source", "")
    per_class = int(data.get("per_class", 5000))

    sources = find_source_folders(WORK_ROOT)
    source = next((s for s in sources if s["name"] == source_name), None)
    if source is None:
        return jsonify({"error": f"source not found: {source_name}"}), 404

    preview: dict[str, dict] = {}
    grand_total = 0
    for cls in sorted(EXPECTED_CLASSES):
        inst_counts = source["instances"].get(cls, {})
        quotas = distribute_quota(inst_counts, per_class)
        cls_total = sum(quotas.values())
        grand_total += cls_total
        preview[cls] = {
            "instance_count": len(inst_counts),
            "total_available": sum(inst_counts.values()),
            "quota": per_class,
            "sampled": cls_total,
            "per_instance": {k: v for k, v in sorted(quotas.items())},
        }

    return jsonify({"per_class": per_class, "grand_total": grand_total, "by_class": preview})


@app.route("/api/execute", methods=["POST"])
def api_execute():
    global _batch_state
    if _batch_state["running"]:
        return jsonify({"error": "batch already running"}), 409

    data = request.get_json(silent=True) or {}
    source_name = data.get("source", "")
    per_class = int(data.get("per_class", 5000))

    sources = find_source_folders(WORK_ROOT)
    source = next((s for s in sources if s["name"] == source_name), None)
    if source is None:
        return jsonify({"error": f"source not found: {source_name}"}), 404

    timestamp = datetime.now().strftime("%m%d")
    grand_total = 0
    for cls in sorted(EXPECTED_CLASSES):
        inst_counts = source["instances"].get(cls, {})
        quotas = distribute_quota(inst_counts, per_class)
        grand_total += sum(quotas.values())

    out_name = f"{source_name}_sampled_{timestamp}_{grand_total}"
    out_root = WORK_ROOT / out_name

    thread = threading.Thread(
        target=_execute_worker,
        args=(source, out_root, per_class, grand_total),
        daemon=True,
    )
    thread.start()
    return jsonify({"status": "started", "output": out_name, "grand_total": grand_total})


def _execute_worker(source: dict, out_root: Path, per_class: int, grand_total: int) -> None:
    global _batch_state
    _batch_state = {"running": True, "total": grand_total, "done": 0, "failed": 0, "status": "processing"}

    try:
        src_path = Path(source["path"])
        for cls in sorted(EXPECTED_CLASSES):
            inst_counts = source["instances"].get(cls, {})
            if not inst_counts:
                continue
            quotas = distribute_quota(inst_counts, per_class)
            for inst_name, quota in sorted(quotas.items()):
                if quota <= 0:
                    continue
                inst_dir = src_path / cls / inst_name
                files = sorted(
                    p.name for p in inst_dir.glob("*")
                    if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg")
                )
                picked = interval_sample(files, quota)
                out_dir = out_root / cls / inst_name
                out_dir.mkdir(parents=True, exist_ok=True)
                for fname in picked:
                    try:
                        shutil.copy2(str(inst_dir / fname), str(out_dir / fname))
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
<title>Dataset Sampler</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', sans-serif; background: #1a1a2e; color: #eee; padding: 20px; max-width: 1100px; margin: 0 auto; }
  h1 { color: #e94560; margin-bottom: 16px; }
  .row { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; margin-bottom: 12px; }
  select, input, button { padding: 6px 12px; border-radius: 4px; border: 1px solid #444; background: #0f3460; color: #eee; font-size: 13px; cursor: pointer; }
  button:hover { background: #e94560; }
  button.primary { background: #c0392b; font-weight: 600; }
  button.primary:hover { background: #e74c3c; }
  label { font-size: 13px; color: #aaa; }
  .info-box { background: #16213e; border-radius: 6px; padding: 14px; margin-bottom: 12px; }
  .info-box table { width: 100%; border-collapse: collapse; font-size: 12px; }
  .info-box th, .info-box td { padding: 4px 8px; text-align: right; border-bottom: 1px solid #1e2d4a; }
  .info-box th:first-child, .info-box td:first-child { text-align: left; }
  .info-box th { color: #999; font-weight: 600; }
  #progress-bar { display: none; margin-top: 12px; }
  #progress-bar .track { background: #0f3460; border-radius: 4px; height: 20px; overflow: hidden; }
  #progress-bar .fill { background: #27ae60; height: 100%; transition: width 0.3s; border-radius: 4px; }
  #progress-bar .text { font-size: 11px; color: #aaa; margin-top: 4px; }
  .error { color: #e74c3c; }
  .success { color: #27ae60; }
</style>
</head>
<body>

<h1>Dataset Sampler</h1>

<div class="row">
  <label>Source folder:</label>
  <select id="sel-source"></select>
  <button id="btn-refresh">Refresh</button>
</div>

<div class="row">
  <label>Per-class target:</label>
  <input type="number" id="inp-quota" value="5000" min="1" max="99999" step="100" style="width:100px;">
  <button id="btn-preview">Preview</button>
  <button id="btn-execute" class="primary">Execute Sample</button>
</div>

<div id="preview-box" class="info-box" style="display:none;"></div>

<div id="progress-bar">
  <div class="track"><div class="fill" id="prog-fill" style="width:0%"></div></div>
  <div class="text" id="prog-text"></div>
</div>

<div id="msg"></div>

<script>
const $ = id => document.getElementById(id);
const selSource = $('sel-source'), inpQuota = $('inp-quota');
const btnRefresh = $('btn-refresh'), btnPreview = $('btn-preview'), btnExecute = $('btn-execute');
const previewBox = $('preview-box'), progBar = $('progress-bar'), progFill = $('prog-fill');
const progText = $('prog-text'), msg = $('msg');

async function loadSources() {
  const resp = await fetch('/api/sources');
  const data = await resp.json();
  selSource.innerHTML = data.map(s =>
    `<option value="${s.name}">${s.name}  (${s.total_images.toLocaleString()} imgs, ${s.class_count} classes)</option>`
  ).join('');
  if (data.length > 0) selSource.value = data[0].name;
}
btnRefresh.addEventListener('click', loadSources);

btnPreview.addEventListener('click', async () => {
  const resp = await fetch('/api/preview', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({source: selSource.value, per_class: parseInt(inpQuota.value)}),
  });
  const data = await resp.json();
  if (data.error) { msg.innerHTML = `<span class="error">${data.error}</span>`; return; }
  let html = `<p style="margin-bottom:8px;">Per-class quota: <b>${data.per_class.toLocaleString()}</b> &rarr; Grand total: <b>${data.grand_total.toLocaleString()}</b> images across 6 classes</p>`;
  html += '<table><tr><th>Class</th><th>Instances</th><th>Available</th><th>Sampled</th></tr>';
  for (const [cls, info] of Object.entries(data.by_class)) {
    html += `<tr><td>${cls}</td><td>${info.instance_count}</td><td>${info.total_available.toLocaleString()}</td><td>${info.sampled.toLocaleString()}</td></tr>`;
  }
  html += '</table>';
  previewBox.innerHTML = html;
  previewBox.style.display = 'block';
  msg.innerHTML = '';
});

btnExecute.addEventListener('click', async () => {
  const quota = parseInt(inpQuota.value);
  if (!confirm(`Sample ${quota.toLocaleString()} per class from "${selSource.value}"?`)) return;
  btnExecute.disabled = true;
  const resp = await fetch('/api/execute', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({source: selSource.value, per_class: quota}),
  });
  const data = await resp.json();
  if (data.error) { msg.innerHTML = `<span class="error">${data.error}</span>`; btnExecute.disabled = false; return; }
  msg.innerHTML = `<span class="success">Exporting to: work/${data.output} (${data.grand_total.toLocaleString()} images)</span>`;
  progBar.style.display = 'block';
  pollProgress();
});

function pollProgress() {
  fetch('/api/progress').then(r => r.json()).then(data => {
    if (!data.running) {
      progText.textContent = data.status === 'done'
        ? `Done: ${data.done.toLocaleString()} copied, ${data.failed} failed.`
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
    parser = argparse.ArgumentParser(description="Dataset sampling tool.")
    parser.add_argument("--work-root", default="work", help="Path to work directory.")
    parser.add_argument("--port", type=int, default=5001, help="Server port.")
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
        print(f"  {s['name']}: {s['total_images']:,} images, {len(s['instances'])} classes")
    print(f"Server    : http://{args.host}:{args.port}")

    app.run(host=args.host, port=args.port, debug=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
