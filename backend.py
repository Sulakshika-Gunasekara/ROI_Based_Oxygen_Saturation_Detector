import os
import time
import gc
import tempfile
import threading
from collections import deque

import cv2
import numpy as np
import torch
import uvicorn
import psutil

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse

import final  # must be in same folder as backend.py


# ===================== Paths (portable) =====================
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

WEIGHTS = os.path.join(ROOT_DIR, "best_run.pt")
FEAT_MU = os.path.join(ROOT_DIR, "feat_mu.pt")
FEAT_STD = os.path.join(ROOT_DIR, "feat_std.pt")

if not os.path.exists(WEIGHTS):
    raise FileNotFoundError(f"Missing weights: {WEIGHTS}")
if not os.path.exists(FEAT_MU):
    raise FileNotFoundError(f"Missing feat_mu: {FEAT_MU}")
if not os.path.exists(FEAT_STD):
    raise FileNotFoundError(f"Missing feat_std: {FEAT_STD}")


# ===================== App =====================
app = FastAPI(title="Non-contact SpO2 Demo API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # demo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===================== Device =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===================== Memory helpers =====================
def get_mem_info():
    p = psutil.Process(os.getpid())
    rss_bytes = p.memory_info().rss
    ram_gb = rss_bytes / (1024 ** 3)

    if torch.cuda.is_available():
        alloc_gb = torch.cuda.memory_allocated(0) / (1024 ** 3)
        res_gb = torch.cuda.memory_reserved(0) / (1024 ** 3)
    else:
        alloc_gb = 0.0
        res_gb = 0.0

    return {
        "ram_gb": round(float(ram_gb), 3),
        "gpu_mem_allocated_gb": round(float(alloc_gb), 3),
        "gpu_mem_reserved_gb": round(float(res_gb), 3),
    }


# ===================== Load normalization =====================
feat_mu = torch.load(FEAT_MU, map_location="cpu").float()
feat_std = torch.load(FEAT_STD, map_location="cpu").float()

N_FEATS = int(feat_mu.numel())
MAX_T = int(getattr(final, "MAX_T", 200))


# ===================== Load model =====================
model = final.TinyTransformer(
    n_feats=N_FEATS,
    d=int(getattr(final, "D_MODEL", 32)),
    h=int(getattr(final, "N_HEADS", 2)),
    L=int(getattr(final, "N_LAYERS", 4)),
    drop=float(getattr(final, "DROP", 0.25)),
    drop_path=float(getattr(final, "DROP_PATH", 0.10)),
    max_t=MAX_T,
).to(device)

state = torch.load(WEIGHTS, map_location="cpu")
model.load_state_dict(state)
model.eval()


# ===================== Face detector =====================
CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


# ===================== Shared Camera Manager (for preview + analysis together) =====================
class SharedCamera:
    """
    Keeps ONE VideoCapture open per selected cam index.
    - latest_frame: for streaming preview
    - frame_buffer: recent frames for analyze_live
    """
    def __init__(self, max_buffer_frames=600):
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)

        self.cam_index = None
        self.cap = None
        self.running = False
        self.thread = None

        self.latest_frame = None
        self.latest_time = 0.0

        self.frame_buffer = deque(maxlen=max_buffer_frames)  # (t, frame_bgr)

    def start(self, cam_index: int):
        with self.lock:
            if self.running and self.cam_index == cam_index and self.cap is not None:
                return True, None

            # stop previous
            self._stop_locked()

            self.cam_index = int(cam_index)

            # try DSHOW first (Windows)
            cap = cv2.VideoCapture(self.cam_index, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(self.cam_index)
                if not cap.isOpened():
                    cap.release()
                    self.cap = None
                    self.running = False
                    return False, f"Could not open camera index {self.cam_index}. Try cam=0/1/2..."

            self.cap = cap
            self.running = True
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()
            return True, None

    def _stop_locked(self):
        self.running = False
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
        self.cap = None
        self.latest_frame = None
        self.frame_buffer.clear()
        self.cond.notify_all()

    def stop(self):
        with self.lock:
            self._stop_locked()

    def _loop(self):
        # capture loop
        while True:
            with self.lock:
                if not self.running or self.cap is None:
                    break
                cap = self.cap

            ok, fr = cap.read()
            if not ok or fr is None:
                # small wait then retry
                time.sleep(0.02)
                continue

            tnow = time.time()
            with self.lock:
                self.latest_frame = fr
                self.latest_time = tnow
                self.frame_buffer.append((tnow, fr))
                self.cond.notify_all()

            time.sleep(0.001)

    def get_latest_jpeg(self, width=640, quality=80):
        with self.lock:
            fr = self.latest_frame.copy() if self.latest_frame is not None else None

        if fr is None:
            return None

        if width is not None:
            h, w = fr.shape[:2]
            if w > width:
                new_h = int(h * (width / w))
                fr = cv2.resize(fr, (width, new_h))

        ok, jpg = cv2.imencode(".jpg", fr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            return None
        return jpg.tobytes()

    def get_frames_last_seconds(self, duration_s: float, min_frames=30):
        """
        Returns a list of frames from the last duration_s seconds.
        Waits until enough frames are available (up to duration_s + small margin).
        """
        end_t = time.time()
        start_t = end_t - float(duration_s)

        # Wait briefly to accumulate frames
        deadline = time.time() + max(0.5, float(duration_s) * 0.3)
        while True:
            with self.lock:
                frames = [fr for (t, fr) in self.frame_buffer if t >= start_t]
                if len(frames) >= min_frames:
                    break
                if time.time() > deadline:
                    break
                self.cond.wait(timeout=0.1)

        with self.lock:
            frames = [fr for (t, fr) in self.frame_buffer if t >= start_t]

        return frames


shared_cam = SharedCamera(max_buffer_frames=900)


# ===================== Signal / Model helpers =====================
def forehead_roi(frame_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

    if len(faces) == 0:
        h, w = frame_bgr.shape[:2]
        x1, x2 = int(w * 0.30), int(w * 0.70)
        y1, y2 = int(h * 0.10), int(h * 0.30)
        return frame_bgr[y1:y2, x1:x2]

    x, y, w, h = faces[0]
    fx1 = x + int(0.20 * w)
    fx2 = x + int(0.80 * w)
    fy1 = y + int(0.10 * h)
    fy2 = y + int(0.30 * h)

    fx1, fy1 = max(fx1, 0), max(fy1, 0)
    fx2, fy2 = min(fx2, frame_bgr.shape[1]), min(fy2, frame_bgr.shape[0])

    roi = frame_bgr[fy1:fy2, fx1:fx2]
    if roi.size == 0:
        H, W = frame_bgr.shape[:2]
        return frame_bgr[int(H * 0.10):int(H * 0.30), int(W * 0.30):int(W * 0.70)]
    return roi


def resample_to_T(arr: np.ndarray, T: int) -> np.ndarray:
    idx = np.linspace(0, len(arr) - 1, T).astype(int)
    return arr[idx]


def bandpass_like(sig: np.ndarray) -> np.ndarray:
    s = sig.astype(np.float32)
    s = s - s.mean()
    s = s / (s.std() + 1e-6)
    bp = np.zeros_like(s)
    bp[1:-1] = s[1:-1] - 0.5 * (s[:-2] + s[2:])
    return bp


def make_17_features_from_signal(sig: np.ndarray) -> np.ndarray:
    T = sig.shape[0]
    bp = bandpass_like(sig)
    fft = np.abs(np.fft.rfft(bp))

    def fbin(i):
        return float(fft[i]) if i < len(fft) else 0.0

    feats = np.zeros((T, 17), dtype=np.float32)
    for t in range(T):
        win = bp[max(0, t - 20): t + 1]
        if len(win) == 0:
            m = sd = mx = mn = rng = 0.0
            delta = 0.0
            last5 = first5 = 0.0
        else:
            m = float(win.mean())
            sd = float(win.std())
            mx = float(win.max())
            mn = float(win.min())
            rng = mx - mn
            delta = float(win[-1] - win[0])
            last5 = float(win[-5:].mean()) if len(win) >= 5 else float(win.mean())
            first5 = float(win[:5].mean()) if len(win) >= 5 else float(win.mean())

        fft_max = float(fft.max()) if len(fft) else 0.0
        fft_min = float(fft.min()) if len(fft) else 0.0
        fft_ratio = float(fft_max / (fft_min + 1e-6)) if len(fft) else 0.0

        feats[t] = np.array(
            [
                m, sd, delta,
                fft_max, fft_min, fft_ratio,
                fbin(1), fbin(2), fbin(3), fbin(4), fbin(5),
                last5, first5,
                mx, mn, rng,
                float(m / (sd + 1e-6)),
            ],
            dtype=np.float32,
        )
    return feats


def infer_spo2_from_frames(frames_bgr: list[np.ndarray]) -> dict:
    green = []
    for fr in frames_bgr:
        roi = forehead_roi(fr)
        if roi.size == 0:
            green.append(0.0)
            continue
        g = roi[:, :, 1].astype(np.float32)
        green.append(float(g.mean()))
    green = np.array(green, dtype=np.float32)

    if len(green) != MAX_T:
        green = resample_to_T(green, MAX_T)

    rppg = bandpass_like(green)

    feats = make_17_features_from_signal(green)
    if feats.shape[1] != N_FEATS:
        out = {"error": f"Feature mismatch: got {feats.shape[1]} expected {N_FEATS}"}
        out.update(get_mem_info())
        return out

    feats = (feats - feat_mu.numpy()[None, :]) / (feat_std.numpy()[None, :] + 1e-6)

    x = torch.from_numpy(feats).unsqueeze(0).to(device)
    mask = torch.zeros((1, MAX_T), dtype=torch.bool, device=device)

    with torch.no_grad():
        spo2 = model(x.float(), mask).item()

    out = {
        "spo2": float(spo2),
        "rppg": rppg.astype(np.float32).tolist(),
        "device": str(device),
        "cuda": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "max_t": MAX_T,
        "n_feats": N_FEATS,
    }
    out.update(get_mem_info())
    return out


def read_video_file_frames(path: str, max_t: int):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None, "Could not open uploaded video."

    frames = []
    while True:
        ok, fr = cap.read()
        if not ok or fr is None:
            break
        frames.append(fr)
    cap.release()

    if len(frames) < 30:
        return None, "Not enough frames in video. Try a longer / clearer video."

    if len(frames) > max_t:
        idxs = np.linspace(0, len(frames) - 1, max_t).astype(int)
        frames = [frames[i] for i in idxs]

    return frames, None


# ===================== Endpoints =====================
@app.get("/health")
def health():
    out = {
        "ok": True,
        "device": str(device),
        "cuda": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "max_t": MAX_T,
        "n_feats": N_FEATS,
    }
    out.update(get_mem_info())
    return out


@app.get("/list_cameras")
def list_cameras(max_index: int = 10):
    found = []
    for i in range(int(max_index) + 1):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        ok = cap.isOpened()
        if ok:
            ret, fr = cap.read()
            if ret and fr is not None:
                found.append(i)
        cap.release()
    out = {"cameras": found, "hint": "Use cam=<index> with /analyze_live and /mjpeg."}
    out.update(get_mem_info())
    return out


@app.get("/mjpeg")
def mjpeg(cam: int = 0):
    ok, err = shared_cam.start(int(cam))
    if not ok:
        return {"error": err}

    def gen():
        while True:
            jpg = shared_cam.get_latest_jpeg(width=640, quality=80)
            if jpg is None:
                time.sleep(0.05)
                continue
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n")
            time.sleep(0.07)  # ~14 fps

    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.post("/analyze_video")
async def analyze_video(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename or "")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = tmp.name
        content = await file.read()
        tmp.write(content)

    try:
        frames, err = read_video_file_frames(tmp_path, MAX_T)
        if err:
            out = {"error": err}
            out.update(get_mem_info())
            return out
        return infer_spo2_from_frames(frames)
    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass


@app.get("/analyze_live")
def analyze_live(cam: int = 0, duration: float = 5.0):
    ok, err = shared_cam.start(int(cam))
    if not ok:
        out = {"error": err}
        out.update(get_mem_info())
        return out

    frames = shared_cam.get_frames_last_seconds(float(duration), min_frames=30)
    if frames is None or len(frames) < 30:
        out = {"error": "Not enough frames from live camera. Improve lighting / check DroidCam connection."}
        out.update(get_mem_info())
        return out

    # downsample to MAX_T like before
    if len(frames) > MAX_T:
        idxs = np.linspace(0, len(frames) - 1, MAX_T).astype(int)
        frames = [frames[i] for i in idxs]

    out = infer_spo2_from_frames(frames)
    out.update({"cam_index": int(cam), "duration": float(duration)})
    return out


@app.on_event("shutdown")
def shutdown_event():
    print("Shutting down... releasing memory.")
    try:
        shared_cam.stop()
    except Exception:
        pass

    global model
    try:
        del model
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
