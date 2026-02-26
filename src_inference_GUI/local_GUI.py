#!/usr/bin/env python3
import argparse
import cgi
import hashlib
import json
import os
import shutil
import subprocess
import threading
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

REPO = Path(__file__).resolve().parents[1]

# Relative to repo root (as you requested)
PREPROC_IMG_REL = Path("assets/containers/singularity_preprocessing.simg")
TRAIN_EVAL_IMG_REL = Path("assets/containers/singularity_train_eval.simg")
DEFAULT_CKPT_REL = Path("assets/inference_model_checkpoint/TOAD_ResNet_harmonized.pt")

# CLAM code directory (mounted read-only)
CLAM_CODE_REL = Path("src_preprocessing/")

# Output location (relative to repo root)
OUT_ROOT_REL = Path("output/inference_GUI_runs")

# Preprocessing settings
FORCED_ENCODER = "resnet50_trunc"
FORCED_TARGET_PATCH_SIZE = "224"

# If your CLAM wrapper has a different name/path, change here:
CLAM_RUNNER = "run_clam_single_input.sh"

ID2LABEL = {
    0: "Adrenal",
    1: "Bladder",
    2: "Breast",
    3: "Cervix",
    4: "Colorectal",
    5: "Endometrial",
    6: "Esophagogastric",
    7: "Germ cell",
    8: "Glioma",
    9: "Head and Neck",
    10: "Liver",
    11: "Lung adeno",
    12: "Ovarian",
    13: "Pancreatic",
    14: "Prostate",
    15: "Renal",
    16: "Skin",
    17: "Thyroid",
}

JOBS = {}


def abs_in_repo(rel: Path) -> Path:
    return (REPO / rel).resolve()


def mk_job_id(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def job_log(job, msg: str):
    job["log"].append(msg)
    job["log"] = job["log"][-220:]


def set_step(job, step: str, state: str):
    # state in {"pending","running","done","error"}
    job["steps"][step] = state


def write_text(path: Path, s: str):
    path.write_text(s, encoding="utf-8")


def write_json(path: Path, obj):
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _safe_int(s: str, default: int) -> int:
    try:
        v = int(str(s).strip())
        return v
    except Exception:
        return default


def _validate_batch_size(batch_size_str: str) -> str:
    """
    Return a safe batch size string. Enforce >0.
    """
    bs = _safe_int(batch_size_str, 800)
    if bs <= 0:
        bs = 800
    # allow very large if user wants; the model/container will fail if too large anyway
    return str(bs)


def run_cmd_stream(job, cmd, env=None, slide_stem=None, meta_dir=None, feat_root=None):
    """
    Streams container stdout to GUI log.
    Updates step states using:
      (1) explicit 'echo' markers
      (2) file existence heuristics (robust fallback)
    """
    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=0,
        env=env,
    )

    buf = ""
    while True:
        ch = p.stdout.read(1)
        if not ch:
            break
        try:
            ch = ch.decode("utf-8", errors="ignore")
        except Exception:
            continue

        if ch in ("\n", "\r"):
            line = buf.strip()
            if line:
                job_log(job, line)
                low = line.lower()

                # -------------- Step markers (from your bash echo lines) --------------
                if "starting patch creation" in low:
                    set_step(job, "patching", "running")

                elif "starting feature extraction" in low:
                    set_step(job, "patching", "done")
                    set_step(job, "feature_extraction", "running")

                # -------------- Robust fallback: if outputs exist, update states --------------
                # If patch h5 exists, patching is effectively done.
                if slide_stem and meta_dir:
                    h5_path = Path(meta_dir) / "patches" / f"{slide_stem}.h5"
                    if h5_path.exists():
                        # Only flip to done if it was pending/running
                        if job["steps"].get("patching") in ("pending", "running"):
                            set_step(job, "patching", "done")

                # If pt exists, feature extraction is done.
                if slide_stem and feat_root:
                    pt_path = Path(feat_root) / "pt_files" / f"{slide_stem}.pt"
                    if pt_path.exists():
                        if job["steps"].get("feature_extraction") in ("pending", "running"):
                            set_step(job, "feature_extraction", "done")

            buf = ""
        else:
            buf += ch

    p.wait()
    return p.returncode


def collect_slide_summary(preproc_img: Path, svs_path: Path) -> dict:
    cmd = [
        "singularity", "exec", "--cleanenv", "--containall",
        "--bind", f"{svs_path}:/app/slide.svs:ro",
        str(preproc_img),
        "bash", "-lc",
        (
            "python3 - <<'PY'\n"
            "import json\n"
            "import openslide\n"
            "s = openslide.open_slide('/app/slide.svs')\n"
            "d = {\n"
            "  'dimensions': list(s.dimensions),\n"
            "  'level_count': int(s.level_count),\n"
            "  'level_dimensions': [list(x) for x in s.level_dimensions],\n"
            "  'level_downsamples': [float(x) for x in s.level_downsamples],\n"
            "  'properties': dict(s.properties),\n"
            "}\n"
            "print(json.dumps(d))\n"
            "PY"
        ),
    ]
    out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
    lines = [l.strip() for l in out.splitlines() if l.strip()]
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].startswith("{") and lines[i].endswith("}"):
            return json.loads(lines[i])
    raise RuntimeError("Could not parse slide summary output.")


def make_thumbnail(preproc_img: Path, svs_path: Path, out_png: Path):
    """
    Create a thumbnail ONCE. The GUI will serve it as a static file thereafter.
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "singularity", "exec", "--cleanenv", "--containall",
        "--bind", f"{svs_path}:/app/slide.svs:ro",
        "--bind", f"{out_png.parent}:/app/out",
        str(preproc_img),
        "bash", "-lc",
        (
            "python3 - <<'PY'\n"
            "import openslide\n"
            "s = openslide.open_slide('/app/slide.svs')\n"
            "thumb = s.get_thumbnail((1400, 1400))\n"
            "thumb.save('/app/out/thumbnail.png')\n"
            "print('ok')\n"
            "PY"
        )
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)


def run_preprocessing(job, preproc_img: Path, clam_code: Path, svs_path: Path, run_dir: Path,
                      gpu: str, batch_size: str):
    """
    Runs your CLAM single-input wrapper inside the preprocessing container.

    Guaranteed behaviors:
    - batch_size coming from GUI is passed into bash wrapper
    - step states are updated using BOTH log markers and file existence
    """
    meta_root = run_dir / "logs_and_metadata"
    feat_root = run_dir / "FEATURES"
    meta_root.mkdir(parents=True, exist_ok=True)
    feat_root.mkdir(parents=True, exist_ok=True)

    slide_file = svs_path.name
    slide_stem = svs_path.stem
    h = hashlib.sha1(str(svs_path).encode("utf-8")).hexdigest()[:8]
    meta_dir = meta_root / f"{slide_stem}__{h}"
    meta_dir.mkdir(parents=True, exist_ok=True)

    # ephemeral dir mounted to /app/input_data; we link the slide into it
    tmp_in = Path(subprocess.check_output(["mktemp", "-d"]).decode().strip())

    try:
        env = os.environ.copy()
        env["SINGULARITYENV_CUDA_VISIBLE_DEVICES"] = gpu
        env["APPTAINERENV_CUDA_VISIBLE_DEVICES"] = gpu
        env["PYTHONUNBUFFERED"] = "1"

        # Make GUI reflect what is really happening
        set_step(job, "patching", "running")
        set_step(job, "feature_extraction", "pending")
        job_log(job, f"Preprocessing started (encoder={FORCED_ENCODER}, batch_size={batch_size})...")

        # Important: do NOT bind the slide file itself to /app/input_data because some systems create temp copies.
        # Instead bind the parent read-only and symlink inside container -> fast and consistent.
        cmd = [
            "singularity", "exec", "--nv", "--cleanenv", "--containall",
            "--bind", f"{clam_code}:/app/CLAM:ro",
            "--bind", f"{tmp_in}:/app/input_data",
            "--bind", f"{svs_path.parent}:/app/slide_parent:ro",
            "--bind", f"{meta_dir}:/app/meta_out",
            "--bind", f"{feat_root}:/app/features_out",
            str(preproc_img),
            "bash", "-lc",
            (
                # link slide into /app/input_data
                f"ln -sf /app/slide_parent/{slide_file} /app/input_data/{slide_file} && "
                f"cd /app/CLAM && "
                f"test -f {CLAM_RUNNER} || (echo 'ERROR: missing {CLAM_RUNNER} in /app/CLAM' && exit 1) && "
                # pass GUI batch size here (CONFIRMED)
                f"bash {CLAM_RUNNER} "
                f"/app/input_data /app/meta_out /app/features_out "
                f"'{slide_file}' '{batch_size}' '{FORCED_ENCODER}' '{FORCED_TARGET_PATCH_SIZE}'"
            )
        ]

        rc = run_cmd_stream(
            job,
            cmd,
            env=env,
            slide_stem=slide_stem,
            meta_dir=meta_dir,
            feat_root=feat_root
        )
        if rc != 0:
            raise RuntimeError(f"Preprocessing failed (exit {rc}).")

        # final authoritative states: if command ended successfully, both must be done
        set_step(job, "patching", "done")
        set_step(job, "feature_extraction", "done")

        pt_path = feat_root / "pt_files" / f"{slide_stem}.pt"
        if not pt_path.exists():
            raise RuntimeError(f"Expected features not found: {pt_path}")

        return pt_path, meta_dir, feat_root

    finally:
        shutil.rmtree(tmp_in, ignore_errors=True)


def run_inference(job, train_eval_img: Path, ckpt_path: Path, pt_path: Path, out_json: Path,
                  gpu: str, sex: str):
    out_json.parent.mkdir(parents=True, exist_ok=True)

    pt_real = Path(pt_path).resolve()
    ckpt_real = Path(ckpt_path).resolve()

    env = os.environ.copy()
    env["SINGULARITYENV_CUDA_VISIBLE_DEVICES"] = gpu
    env["APPTAINERENV_CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONUNBUFFERED"] = "1"

    set_step(job, "inference", "running")
    job_log(job, "Inference started...")

    cmd = [
        "singularity", "exec", "--nv", "--cleanenv", "--containall",
        "--bind", f"{REPO}:/app/TOAD_repo:ro",
        "--bind", f"{pt_real}:/app/input.pt:ro",
        "--bind", f"{ckpt_real}:/app/model.pt:ro",
        "--bind", f"{out_json.parent}:/app/out",
        str(train_eval_img),
        "bash", "-lc",
        (
            f"export CUDA_VISIBLE_DEVICES={gpu} ; "
            f"python3 /app/TOAD_repo/src_inference_GUI/infer_from_pt.py "
            f"--pt /app/input.pt "
            f"--ckpt /app/model.pt "
            f"--sex {sex} "
            f"--out /app/out/{out_json.name}"
        )
    ]

    rc = run_cmd_stream(job, cmd, env=env)
    if rc != 0:
        raise RuntimeError(f"Inference failed (exit {rc}).")

    set_step(job, "inference", "done")

    if not out_json.exists():
        raise RuntimeError("pred.json not produced.")


def postprocess_pred(pred: dict):
    probs_sorted = pred.get("probs_sorted", [])
    fixed = []
    for row in probs_sorted:
        idx = int(row["index"])
        prob = float(row["prob"])
        fixed.append({"index": idx, "label": ID2LABEL.get(idx, f"class_{idx}"), "prob": prob})
    fixed.sort(key=lambda r: r["prob"], reverse=True)
    return fixed


def write_run_summaries(run_dir: Path, job: dict, slide_summary: dict, pred_fixed: list, pt_path: Path):
    svs_p = Path(job["svs_path"])
    input_summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "svs_path": str(svs_p),
        "svs_filename": svs_p.name,
        "svs_size_bytes": int(svs_p.stat().st_size),
        "gpu": job["gpu"],
        "encoder": FORCED_ENCODER,
        "target_patch_size": int(FORCED_TARGET_PATCH_SIZE),
        "batch_size": int(job["batch_size"]),
        "sex": int(job["sex"]),
        "checkpoint_path": str(job["ckpt_path"]),
        "pt_path": str(pt_path),
        "slide": slide_summary,
    }
    write_json(run_dir / "input_summary.json", input_summary)

    txt_in = [
        f"timestamp: {input_summary['timestamp']}",
        f"svs_path: {input_summary['svs_path']}",
        f"svs_size_bytes: {input_summary['svs_size_bytes']}",
        f"gpu: {input_summary['gpu']}",
        f"encoder: {input_summary['encoder']}",
        f"target_patch_size: {input_summary['target_patch_size']}",
        f"batch_size: {input_summary['batch_size']}",
        f"sex: {input_summary['sex']}",
        f"checkpoint_path: {input_summary['checkpoint_path']}",
        f"pt_path: {input_summary['pt_path']}",
        f"dimensions: {slide_summary.get('dimensions')}",
        f"level_count: {slide_summary.get('level_count')}",
        "",
    ]
    write_text(run_dir / "input_summary.txt", "\n".join(txt_in))

    output_summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "top_prediction": pred_fixed[0] if pred_fixed else None,
        "predictions": pred_fixed,
    }
    write_json(run_dir / "output_summary.json", output_summary)

    txt_out = [f"timestamp: {output_summary['timestamp']}"]
    if pred_fixed:
        txt_out.append(f"top_prediction: {pred_fixed[0]['label']} ({pred_fixed[0]['prob']*100:.2f}%)")
    txt_out.append("")
    txt_out.append("all_predictions:")
    for r in pred_fixed:
        txt_out.append(f"- {r['label']}: {r['prob']*100:.2f}%")
    txt_out.append("")
    write_text(run_dir / "output_summary.txt", "\n".join(txt_out))


def worker(job_id: str):
    job = JOBS[job_id]
    try:
        # reset steps defensively
        job["steps"] = {"patching": "pending", "feature_extraction": "pending", "inference": "pending"}

        # Thumbnail
        job["status"] = "thumbnail"
        make_thumbnail(job["preproc_img"], job["svs_path"], job["run_dir"] / "thumbnail.png")
        job["thumb_ready"] = (job["run_dir"] / "thumbnail.png").exists()

        # Slide summary
        slide_summary = collect_slide_summary(job["preproc_img"], job["svs_path"])
        write_json(job["run_dir"] / "slide_summary.json", slide_summary)

        # Preprocessing
        job["status"] = "preprocessing"
        pt_path, meta_dir, feat_root = run_preprocessing(
            job,
            job["preproc_img"],
            job["clam_code"],
            job["svs_path"],
            job["run_dir"],
            job["gpu"],
            job["batch_size"],
        )
        job["pt_path"] = str(pt_path)

        # Inference
        job["status"] = "inference"
        out_json = job["run_dir"] / "pred.json"
        run_inference(job, job["train_eval_img"], job["ckpt_path"], pt_path, out_json, job["gpu"], job["sex"])

        pred = json.loads(out_json.read_text(encoding="utf-8"))
        pred_fixed = postprocess_pred(pred)
        write_json(job["run_dir"] / "pred_fixed.json", pred_fixed)

        write_run_summaries(job["run_dir"], job, slide_summary, pred_fixed, pt_path)

        job["pred_ready"] = True
        job["status"] = "done"

    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        job_log(job, f"ERROR: {e}")
        for k, v in job["steps"].items():
            if v == "running":
                job["steps"][k] = "error"


INDEX_HTML = r"""
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>TOAD inference</title>
<style>
  html, body { height: 100%; margin: 0; }
  body { font-family: sans-serif; padding: 10px 12px; box-sizing: border-box; overflow: hidden; }

  .topbar {
    display: flex;
    gap: 10px;
    align-items: flex-end;
    flex-wrap: wrap;
    border: 1px solid #ddd;
    border-radius: 10px;
    padding: 8px 10px;
  }
  .ctrl { display: flex; flex-direction: column; }
  .ctrl label { font-size: 12px; font-weight: 700; margin-bottom: 4px; }
  .ctrl input, .ctrl select { padding: 6px 8px; font-size: 12px; min-width: 140px; }
  .ctrl.small input, .ctrl.small select { min-width: 90px; }
  .ctrl.wide input { min-width: 520px; }
  button { padding: 7px 12px; font-size: 12px; cursor: pointer; border-radius: 8px; border: 1px solid #888; background: #f6f6f6; }

  .steps {
    display:flex; gap:12px; align-items:center;
    margin-top: 8px;
    border: 1px solid #eee;
    border-radius: 10px;
    padding: 8px 10px;
  }
  .step { display:flex; gap:8px; align-items:center; font-size:12px; font-weight:800; }
  .icon { width: 18px; text-align:center; }

  .main {
    display: grid;
    grid-template-columns: 1.3fr 0.7fr;
    gap: 10px;
    height: calc(100vh - 166px);
    margin-top: 10px;
  }
  .panel { border: 1px solid #ddd; border-radius: 10px; padding: 10px; box-sizing: border-box; overflow: hidden; }

  #statusLine { font-size: 12px; font-weight: 800; margin-bottom: 8px; }
  #err { color: #b00020; font-weight: 900; font-size: 12px; margin-left: 10px; }

  #thumb {
    width: 100%;
    height: calc(100% - 140px);
    object-fit: contain;
    border: 1px solid #eee;
    border-radius: 10px;
    background: #fafafa;
    display:none;
  }
  #placeholder {
    height: calc(100% - 140px);
    display:flex;
    align-items:center;
    justify-content:center;
    border: 1px dashed #bbb;
    border-radius: 10px;
    color:#444;
    font-size: 13px;
    font-weight: 700;
    background:#fcfcfc;
  }

  #log {
    height: 120px;
    overflow: hidden;
    white-space: pre-wrap;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 11px;
    background: #0b1020;
    color: #e6e6e6;
    border-radius: 10px;
    padding: 8px;
    margin-top: 8px;
  }

  #probList { height: calc(100% - 48px); display: flex; flex-direction: column; gap: 6px; overflow: hidden; }
  .row { display: grid; grid-template-columns: 1fr 0.95fr; gap: 8px; align-items: center; }
  .label { font-size: 12px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .barWrap { height: 14px; background: #eee; border-radius: 8px; overflow: hidden; position: relative; }
  .bar { height: 100%; background: #3b82f6; }
  .val { position: absolute; right: 6px; top: -1px; font-size: 11px; font-weight: 900; color: #111; }
  .hint { font-size: 11px; color: #444; margin-bottom: 8px; line-height: 1.2; }
</style>
</head>

<body>
  <div class="topbar">
    <div class="ctrl wide">
      <label>.svs path</label>
      <input id="svs_path" placeholder="/abs/path/to/slide.svs">
    </div>

    <div class="ctrl small">
      <label>GPU</label>
      <input id="gpu" value="0">
    </div>

    <div class="ctrl">
      <label>Checkpoint</label>
      <input id="ckpt" value="{{DEFAULT_CKPT}}">
    </div>

    <div class="ctrl small">
      <label>Batch</label>
      <input id="batch_size" value="800">
    </div>

    <div class="ctrl small">
      <label>Sex</label>
      <select id="sex">
        <option value="0">F (0)</option>
        <option value="1">M (1)</option>
      </select>
    </div>

    <button id="runBtn">Run</button>
  </div>

  <div class="steps">
    <div class="step"><div class="icon" id="ic_patching">⬜</div><div>Patching (~seconds)</div></div>
    <div class="step"><div class="icon" id="ic_feature">⬜</div><div>Feature extraction (~minutes)</div></div>
    <div class="step"><div class="icon" id="ic_infer">⬜</div><div>Inference (~seconds)</div></div>
  </div>

  <div class="main">
    <div class="panel">
      <div id="statusLine">
        Status: <span id="status">idle</span>
        <span id="err"></span>
      </div>

      <div id="placeholder">Run to see the slide and the predictions</div>
      <img id="thumb" src="" alt="thumbnail">

      <div id="log"></div>
    </div>

    <div class="panel">
      <div style="font-size:12px; font-weight:900; margin-bottom:6px;">Tumor site prediction</div>
      <div class="hint">
        If results look strange, verify the checkpoint was trained with ResNet features
      </div>
      <div id="probList"></div>
    </div>
  </div>

<script>
  function setStatus(s){ document.getElementById("status").textContent = s; }
  function setErr(e){ document.getElementById("err").textContent = e ? (" — " + e) : ""; }
  function setLog(lines){
    document.getElementById("log").textContent = (lines || []).slice(-18).join("\\n");
  }

  function iconFor(state){
    if(state === "running") return "⏳";
    if(state === "done") return "✅";
    if(state === "error") return "❌";
    return "⬜";
  }

  function setSteps(steps){
    if(!steps) return;
    document.getElementById("ic_patching").textContent = iconFor(steps.patching);
    document.getElementById("ic_feature").textContent = iconFor(steps.feature_extraction);
    document.getElementById("ic_infer").textContent = iconFor(steps.inference);
  }

  function pct(prob){
    const p = Math.max(0, Math.min(1, prob)) * 100.0;
    return p.toFixed(2) + "%";
  }

  function renderProbs(rows){
    const box = document.getElementById("probList");
    box.innerHTML = "";
    if(!rows || rows.length === 0){
      box.innerHTML = "<div style='font-size:12px;'>No results yet.</div>";
      return;
    }
    rows.forEach(r => {
      const row = document.createElement("div");
      row.className = "row";

      const lab = document.createElement("div");
      lab.className = "label";
      lab.textContent = r.label;

      const bw = document.createElement("div");
      bw.className = "barWrap";

      const bar = document.createElement("div");
      bar.className = "bar";
      bar.style.width = (Math.max(0, Math.min(1, r.prob)) * 100).toFixed(2) + "%";

      const val = document.createElement("div");
      val.className = "val";
      val.textContent = pct(r.prob);

      bw.appendChild(bar);
      bw.appendChild(val);

      row.appendChild(lab);
      row.appendChild(bw);
      box.appendChild(row);
    });
  }

  async function poll(job_id){
    let thumbShown = false;

    while(true){
      const r = await fetch(`/status?job_id=${job_id}`);
      const s = await r.json();

      setStatus(s.status);
      setErr(s.error || "");
      setLog(s.log || []);
      setSteps(s.steps || {});

      if(s.thumb_ready && !thumbShown){
        document.getElementById("placeholder").style.display = "none";
        const img = document.getElementById("thumb");
        img.style.display = "block";
        img.src = `/thumb?job_id=${job_id}`;   // fetch ONCE
        thumbShown = true;
      }

      if(s.pred_ready){
        const pr = await fetch(`/results?job_id=${job_id}`);
        const rows = await pr.json();
        renderProbs(rows);
      }

      if(s.status === "done" || s.status === "error") break;
      await new Promise(res => setTimeout(res, 900));
    }
  }

  document.getElementById("runBtn").onclick = async () => {
    setErr("");
    setStatus("starting");
    renderProbs([]);

    setSteps({patching:"pending", feature_extraction:"pending", inference:"pending"});

    const fd = new FormData();
    fd.append("svs_path", document.getElementById("svs_path").value.trim());
    fd.append("gpu", document.getElementById("gpu").value.trim());
    fd.append("checkpoint_path", document.getElementById("ckpt").value.trim());
    fd.append("batch_size", document.getElementById("batch_size").value.trim());
    fd.append("sex", document.getElementById("sex").value);

    const r = await fetch("/start", { method:"POST", body: fd });
    const j = await r.json();
    if(j.error){
      setErr(j.error);
      setStatus("idle");
      return;
    }
    await poll(j.job_id);
  };
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def _send(self, code=200, content_type="application/json", body=b""):
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)

        if u.path == "/":
            html = INDEX_HTML.replace("{{DEFAULT_CKPT}}", str(DEFAULT_CKPT_REL))
            self._send(200, "text/html; charset=utf-8", html.encode("utf-8"))
            return

        if u.path == "/status":
            q = parse_qs(u.query)
            job_id = (q.get("job_id") or [""])[0]
            job = JOBS.get(job_id)
            if not job:
                self._send(404, "application/json", json.dumps({"error": "job not found"}).encode())
                return
            payload = {
                "status": job["status"],
                "error": job.get("error", ""),
                "thumb_ready": job.get("thumb_ready", False),
                "pred_ready": job.get("pred_ready", False),
                "log": job.get("log", []),
                "steps": job.get("steps", {}),
            }
            self._send(200, "application/json", json.dumps(payload).encode("utf-8"))
            return

        if u.path == "/thumb":
            q = parse_qs(u.query)
            job_id = (q.get("job_id") or [""])[0]
            job = JOBS.get(job_id)
            if not job:
                self._send(404, "text/plain", b"job not found")
                return
            p = job["run_dir"] / "thumbnail.png"
            if not p.exists():
                self._send(404, "text/plain", b"thumbnail not ready")
                return
            self._send(200, "image/png", p.read_bytes())
            return

        if u.path == "/results":
            q = parse_qs(u.query)
            job_id = (q.get("job_id") or [""])[0]
            job = JOBS.get(job_id)
            if not job:
                self._send(404, "application/json", json.dumps({"error": "job not found"}).encode())
                return
            p = job["run_dir"] / "pred_fixed.json"
            if not p.exists():
                self._send(404, "application/json", json.dumps({"error": "results not ready"}).encode())
                return
            self._send(200, "application/json", p.read_bytes())
            return

        self._send(404, "text/plain; charset=utf-8", b"not found")

    def do_POST(self):
        u = urlparse(self.path)
        if u.path != "/start":
            self._send(404, "text/plain", b"not found")
            return

        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": self.headers.get("Content-Type"),
            },
        )

        svs_path = (form.getfirst("svs_path") or "").strip()
        gpu = (form.getfirst("gpu") or "0").strip()
        batch_size = _validate_batch_size(form.getfirst("batch_size") or "800")
        sex = (form.getfirst("sex") or "0").strip()

        ckpt_path_in = (form.getfirst("checkpoint_path") or "").strip()
        if not ckpt_path_in:
            ckpt_path_in = str(DEFAULT_CKPT_REL)

        svs_p = Path(svs_path).resolve() if svs_path else None
        if svs_p is None or not svs_p.exists():
            self._send(200, "application/json", json.dumps({"error": "Provide a valid .svs path"}).encode())
            return

        ckpt_p = (REPO / ckpt_path_in).resolve() if not os.path.isabs(ckpt_path_in) else Path(ckpt_path_in).resolve()
        if not ckpt_p.exists():
            self._send(200, "application/json", json.dumps({"error": f"Checkpoint not found: {ckpt_path_in}"}).encode())
            return

        if sex not in ("0", "1"):
            self._send(200, "application/json", json.dumps({"error": "Sex must be 0 (F) or 1 (M)"}).encode())
            return

        preproc_img = abs_in_repo(PREPROC_IMG_REL)
        train_eval_img = abs_in_repo(TRAIN_EVAL_IMG_REL)
        clam_code = abs_in_repo(CLAM_CODE_REL)
        out_root = abs_in_repo(OUT_ROOT_REL)

        if not preproc_img.exists():
            self._send(200, "application/json", json.dumps({"error": f"Missing: {PREPROC_IMG_REL}"}).encode())
            return
        if not train_eval_img.exists():
            self._send(200, "application/json", json.dumps({"error": f"Missing: {TRAIN_EVAL_IMG_REL}"}).encode())
            return
        if not clam_code.exists():
            self._send(200, "application/json", json.dumps({"error": f"Missing: {CLAM_CODE_REL}"}).encode())
            return

        out_root.mkdir(parents=True, exist_ok=True)

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = (out_root / run_id).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)

        job_id = mk_job_id(f"{svs_p}|{ckpt_p}|{run_id}|{batch_size}|{sex}|{gpu}")
        job = {
            "job_id": job_id,
            "status": "queued",
            "error": "",
            "thumb_ready": False,
            "pred_ready": False,
            "log": [],
            "steps": {"patching": "pending", "feature_extraction": "pending", "inference": "pending"},
            "svs_path": svs_p,
            "ckpt_path": ckpt_p,
            "gpu": gpu,
            "batch_size": batch_size,
            "sex": sex,
            "preproc_img": preproc_img,
            "train_eval_img": train_eval_img,
            "clam_code": clam_code,
            "run_dir": run_dir,
        }

        # Save a small run metadata immediately (so debugging is easy even if it crashes)
        write_json(run_dir / "run_meta.json", {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "svs_path": str(svs_p),
            "gpu": gpu,
            "batch_size": batch_size,
            "sex": sex,
            "forced_encoder": FORCED_ENCODER,
            "forced_target_patch_size": FORCED_TARGET_PATCH_SIZE,
            "checkpoint": str(ckpt_p),
        })

        JOBS[job_id] = job
        t = threading.Thread(target=worker, args=(job_id,), daemon=True)
        t.start()

        self._send(200, "application/json", json.dumps({"job_id": job_id}).encode("utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    httpd = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"GUI running at: http://127.0.0.1:{args.port}/")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
