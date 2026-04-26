import os
import subprocess
import sys
import threading
import time
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file

PROJECT = Path(__file__).parent
PYTHON = sys.executable
LOG_FILE = PROJECT / "webapp_run.log"
INPUT_VIDEO = PROJECT / "input.mp4"
OUTPUT_VIDEO = PROJECT / "output.mp4"
MAX_UPLOAD_MB = 500

JOB_LOCK = threading.Lock()
JOB = {"status": "idle", "message": "", "started_at": 0.0, "finished_at": 0.0}


def _set(**kw):
    JOB.update(kw)


def _reencode_for_browser(path: Path) -> None:
    import imageio_ffmpeg
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    tmp = path.with_name(path.stem + ".h264.mp4")
    subprocess.run(
        [ffmpeg, "-y", "-i", str(path), "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", "-an", str(tmp)],
        check=True, capture_output=True,
    )
    tmp.replace(path)


def _run_pipeline():
    with JOB_LOCK:
        _set(status="running", message="Processing...", started_at=time.time(), finished_at=0.0)
        try:
            with open(LOG_FILE, "w", encoding="utf-8") as lf:
                proc = subprocess.Popen(
                    [str(PYTHON), "single_video.py", str(INPUT_VIDEO), str(OUTPUT_VIDEO)],
                    cwd=str(PROJECT),
                    stdout=lf,
                    stderr=subprocess.STDOUT,
                )
                proc.wait()
            if proc.returncode != 0:
                _set(status="error", message=f"Pipeline exited with code {proc.returncode}.", finished_at=time.time())
                return
            _set(status="running", message="Re-encoding for browser playback...")
            if OUTPUT_VIDEO.exists():
                _reencode_for_browser(OUTPUT_VIDEO)
            _set(status="done", message="Pipeline finished.", finished_at=time.time())
        except subprocess.CalledProcessError as e:
            stderr = e.stderr.decode("utf-8", errors="replace")[-500:] if e.stderr else ""
            _set(status="error", message=f"ffmpeg failed: {stderr}", finished_at=time.time())
        except Exception as e:
            _set(status="error", message=f"{type(e).__name__}: {e}", finished_at=time.time())


STAGE_PROGRESS = [
    ("Loading models",    5,  "Loading models..."),
    ("Running YOLO",      15, "Detecting players in each frame..."),
    ("Extracting ResNet", 60, "Computing visual fingerprints..."),
    ("Matching players",  80, "Tracking players across frames..."),
    ("Visualizing",       88, "Drawing bounding boxes and IDs..."),
    ("Saved video",       94, "Exporting video..."),
    ("Re-encoding",       97, "Re-encoding for browser playback..."),
]


def _tail_stage() -> tuple[str, str, int]:
    if not LOG_FILE.exists():
        return ("", "", 0)
    raw = ""
    try:
        with open(LOG_FILE, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                s = line.strip()
                for marker, _, _ in STAGE_PROGRESS:
                    if marker in s:
                        raw = s
                        break
    except OSError:
        pass
    friendly, pct = "", 0
    for marker, p, label in STAGE_PROGRESS:
        if marker in raw:
            friendly, pct = label, p
    return (raw, friendly, pct)


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024


@app.route("/")
def index():
    cache_bust = int(OUTPUT_VIDEO.stat().st_mtime) if OUTPUT_VIDEO.exists() else 0
    return render_template(
        "index.html",
        cache_bust=cache_bust,
        output_ready=OUTPUT_VIDEO.exists(),
    )


@app.route("/status")
def status():
    data = dict(JOB)
    raw, friendly, pct = _tail_stage()
    data["stage_raw"] = raw
    data["stage"] = friendly
    data["progress"] = 100 if JOB["status"] == "done" else (0 if JOB["status"] in ("idle", "error") else pct)
    if JOB["status"] == "running" and JOB["started_at"]:
        data["elapsed"] = int(time.time() - JOB["started_at"])
    return jsonify(data)


@app.route("/upload", methods=["POST"])
def upload():
    if JOB["status"] == "running":
        return jsonify(error="Still processing the previous video — wait a moment."), 409
    v = request.files.get("video")
    if not v or not v.filename:
        return jsonify(error="Please choose a video file to upload."), 400
    v.save(str(INPUT_VIDEO))
    threading.Thread(target=_run_pipeline, daemon=True).start()
    return jsonify(ok=True)


@app.route("/video/result")
def video_result():
    if not OUTPUT_VIDEO.exists():
        return "not rendered yet", 404
    return send_file(str(OUTPUT_VIDEO), mimetype="video/mp4", conditional=True)


if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "5000"))
    app.run(host=host, port=port, debug=False, use_reloader=False)
