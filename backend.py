import os
import tempfile
import cv2
import numpy as np
import torch
import uvicorn as uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import final  # your final.py (TinyTransformer + constants)
import gc
# ---------------- Paths ----------------
BASE_DIR = r"D:\demo final"
LOG_DIR = os.path.join(BASE_DIR, final.LOG_DIR)

WEIGHTS = os.path.join(LOG_DIR, "best_run.pt")
FEAT_MU = os.path.join(LOG_DIR, "feat_mu.pt")
FEAT_STD = os.path.join(LOG_DIR, "feat_std.pt")

MAX_T = int(getattr(final, "MAX_T", 200))

# GLOBAL VARIABLES
model = None
device = None
CASCADE = None

# ---------------- FastAPI ----------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




@app.on_event("shutdown")
def shutdown_event():
    global model
    print("Shutting down... releasing GPU memory.")

    # 1. Remove the model from memory
    if 'model' in globals():
        del model

    # 2. Force Python to find and delete orphaned objects
    gc.collect()

    # 3. Clear the PyTorch 'Reserved' memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # Wait for all GPU kernels to finish
        print(f"Memory cleared. Current allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")


def forehead_roi(frame_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    faces = CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

    if len(faces) == 0:
        # fallback crop
        h, w = frame_bgr.shape[:2]
        x1 = int(w * 0.30);
        x2 = int(w * 0.70)
        y1 = int(h * 0.10);
        y2 = int(h * 0.30)
        return frame_bgr[y1:y2, x1:x2]

    x, y, w, h = max(faces, key=lambda r: r[2] * r[3])

    fx1 = x + int(0.25 * w)
    fx2 = x + int(0.75 * w)
    fy1 = y + int(0.10 * h)
    fy2 = y + int(0.30 * h)

    roi = frame_bgr[max(fy1, 0):max(fy2, 0), max(fx1, 0):max(fx2, 0)]
    if roi.size == 0:
        roi = frame_bgr[y:y + max(h // 3, 1), x:x + w]
    return roi


def resample_to_T(arr: np.ndarray, T: int) -> np.ndarray:
    idx = np.linspace(0, len(arr) - 1, T).astype(int)
    return arr[idx]


def bandpass_like(sig: np.ndarray) -> np.ndarray:
    """Simple detrend + diff filter (demo rPPG)."""
    s = sig.astype(np.float32)
    s = s - s.mean()
    s = s / (s.std() + 1e-6)
    bp = np.zeros_like(s)
    bp[1:-1] = s[1:-1] - 0.5 * (s[:-2] + s[2:])
    return bp


def make_17_features_from_signal(sig: np.ndarray) -> np.ndarray:
    """
    NOTE: This is a stable DEMO feature extractor.
    For dataset-level accuracy, replace this with YOUR exact preprocessing that created the NPZ features.
    Output: [T, 17]
    """
    T = sig.shape[0]
    bp = bandpass_like(sig)
    fft = np.abs(np.fft.rfft(bp))

    def fbin(i): return float(fft[i]) if i < len(fft) else 0.0

    feats = np.zeros((T, 17), dtype=np.float32)
    for t in range(T):
        win = bp[max(0, t - 20): t + 1]
        m = float(win.mean()) if len(win) else 0.0
        sd = float(win.std()) if len(win) else 0.0
        mx = float(win.max()) if len(win) else 0.0
        mn = float(win.min()) if len(win) else 0.0
        rng = mx - mn

        feats[t] = np.array([
            m, sd,
            float(win[-1] - win[0]) if len(win) else 0.0,
            float(fft.max()) if len(fft) else 0.0,
            float(fft.min()) if len(fft) else 0.0,
            float(fft.max() / (fft.min() + 1e-6)) if len(fft) else 0.0,
            fbin(1), fbin(2), fbin(3), fbin(4), fbin(5),
            float(win[-5:].mean()) if len(win) else 0.0,
            float(win[:5].mean()) if len(win) else 0.0,
            mx, mn, rng,
            float(m / (sd + 1e-6))
        ], dtype=np.float32)

    return feats


@app.get("/health")
def health():
    return {
        "ok": True,
        "device": str(device),
        "cuda": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "max_t": MAX_T,
        "n_feats": N_FEATS,
    }


@app.post("/analyze_video")
async def analyze_video(file: UploadFile = File(...)):
    # Save upload to temp file
    suffix = os.path.splitext(file.filename)[1].lower() or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        return {"error": "Could not open video file."}

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames = []

    if total > 0:
        idxs = np.linspace(0, total - 1, MAX_T).astype(int)
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, fr = cap.read()
            if ok:
                frames.append(fr)
    else:
        # fallback
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(fr)
        if len(frames) > MAX_T:
            frames = list(np.array(frames)[np.linspace(0, len(frames) - 1, MAX_T).astype(int)])

    cap.release()
    try:
        os.remove(tmp_path)
    except:
        pass

    if len(frames) < 30:
        return {"error": "Video too short or frames not readable. Use a clearer face video (>=3s)."}

    # Extract green mean from forehead ROI
    green = []
    for fr in frames:
        roi = forehead_roi(fr)
        if roi.size == 0:
            green.append(0.0)
            continue
        g = roi[:, :, 1].astype(np.float32)
        green.append(float(g.mean()))
    green = np.array(green, dtype=np.float32)

    # Ensure exactly MAX_T
    if len(green) != MAX_T:
        green = resample_to_T(green, MAX_T)

    # rPPG waveform
    rppg = bandpass_like(green)

    # Features -> normalize
    feats = make_17_features_from_signal(green)  # [T,17]
    if feats.shape[1] != N_FEATS:
        return {"error": f"Feature mismatch: got {feats.shape[1]} expected {N_FEATS}"}

    feats = (feats - feat_mu.numpy()[None, :]) / feat_std.numpy()[None, :]

    # Inference on GPU
    x = torch.from_numpy(feats).unsqueeze(0).to(device)  # [1,T,F]
    mask = torch.zeros((1, MAX_T), dtype=torch.bool, device=device)
    print("device = ",device)

    with torch.no_grad():
        spo2 = model(x.float(), mask).item()

    peak_memory = torch.cuda.max_memory_allocated(device) / 1024 ** 2
    print(f"Peak GPU Memory during Inference: {peak_memory:.2f} MB")

    return {
        "spo2": float(spo2),
        "rppg": rppg.astype(np.float32).tolist(),
        "device": str(device),
        "cuda": bool(torch.cuda.is_available()),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("model is loaded in to:", device)

    # ---------------- Load normalization stats ----------------
    feat_mu = torch.load(FEAT_MU, map_location="cpu").float()
    feat_std = torch.load(FEAT_STD, map_location="cpu").float().clamp_min(1e-6)
    N_FEATS = int(feat_mu.numel())

    # ---------------- Build + load model ----------------
    model = final.TinyTransformer(
        n_feats=N_FEATS,
        d=final.D_MODEL,
        h=final.N_HEADS,
        L=final.N_LAYERS,
        drop=final.DROP,
        drop_path=final.DROP_PATH,
        max_t=MAX_T,
    ).to(device)

    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
    model.eval()
    peak_memory = torch.cuda.max_memory_allocated(device) / 1024 ** 2
    print(f"Peak GPU Memory during modal loading: {peak_memory:.2f} MB")

    # ---------------- Face detector ----------------
    CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    uvicorn.run(app, host="0.0.0.0", port=8001)