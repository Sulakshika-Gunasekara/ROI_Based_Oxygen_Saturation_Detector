# final.py — Single dataset folder (70/30 train/test) + small val split for early stopping
import os, glob, math, csv, time, platform
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split

# Enable plotting by default
PLOT = True

# ==================== CONFIG ====================
DATA_DEFAULT_DIR = r"D:\New folder"   # <-- your dataset folder containing *.npz

LOG_DIR         = "training_logs_single"
os.makedirs(LOG_DIR, exist_ok=True)

EPOCHS          = 200
BATCH           = 32
LR              = 3e-4
WD              = 1e-4
PATIENCE        = 40
DROP            = 0.25
DROP_PATH       = 0.10
USE_Y_ZSCORE    = True
CLIP_NORM       = 1.0
SAVE_EPS        = 1e-4

# Cap sequence length (IMPORTANT: also used in ONNX export)
MAX_T           = 200

# Augmentations
TIME_MASK_P     = 0.30
TIME_MASK_MAX   = 20
FEAT_MASK_P     = 0.30
FEAT_MASK_MAX   = 4
JITTER_STD      = 0.01
SHIFT_MAX       = 5

# Model size (Tiny)
D_MODEL         = 32
N_HEADS         = 2
N_LAYERS        = 4


# ================== DATASET =====================
class NPZSeqDataset(Dataset):
    def __init__(self, root_dir: str, pattern: str = "*.npz"):
        self.files = sorted(
            glob.glob(os.path.join(root_dir, "**", pattern), recursive=True)
        )
        if not self.files:
            raise FileNotFoundError(f"No '{pattern}' under '{root_dir}'")

        self.Xs, self.ys, self.index = [], [], []
        for fi, p in enumerate(self.files):
            d = np.load(p)
            X = d["features"]  # [N, T, F]
            y = d["labels"]    # [N]
            self.Xs.append(X)
            self.ys.append(y)
            for si in range(X.shape[0]):
                self.index.append((fi, si))

        self.n_feats = int(self.Xs[0].shape[2])

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        fi, si = self.index[idx]
        x = torch.tensor(self.Xs[fi][si], dtype=torch.float32)
        y = torch.tensor(self.ys[fi][si], dtype=torch.float32)
        return x, y, fi


def collate_pad(batch):
    B = len(batch)
    xs, ys, fis, lens = [], [], [], []
    for (seq, tgt, fi) in batch:
        T = seq.shape[0]
        if T > MAX_T:
            start = (T - MAX_T)//2
            seq = seq[start:start+MAX_T]
            T = MAX_T
        xs.append(seq); ys.append(tgt); fis.append(fi); lens.append(T)

    Tm = max(lens); F = xs[0].shape[1]
    X   = torch.zeros(B, Tm, F, dtype=torch.float32)
    y   = torch.zeros(B, dtype=torch.float32)
    mask= torch.ones(B, Tm, dtype=torch.bool)
    fi_t= torch.zeros(B, dtype=torch.long)

    for i, (seq, tgt, fi) in enumerate(zip(xs, ys, fis)):
        T = seq.shape[0]
        X[i, :T] = seq
        mask[i, :T] = False
        y[i] = tgt
        fi_t[i] = fi
    return X, y, mask, fi_t


# --------- Tiny Transformer ----------
def sinusoid_pe(T, D, device):
    pe = torch.zeros(T, D, device=device)
    pos = torch.arange(T, device=device).float().unsqueeze(1)
    div = torch.exp(torch.arange(0, D, 2, device=device).float() * (-math.log(10000.0)/D))
    pe[:,0::2] = torch.sin(pos*div)
    pe[:,1::2] = torch.cos(pos*div)
    return pe

class DropPath(nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.p = float(p)
    def forward(self, x):
        if (not self.training) or self.p == 0.0:
            return x
        keep = 1.0 - self.p
        shape = (x.shape[0],) + (1,)*(x.ndim-1)
        mask = x.new_empty(shape).bernoulli_(keep) / keep
        return x * mask

class TinyBlock(nn.Module):
    def __init__(self, d=48, h=2, drop=0.2, drop_path=0.1):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, dropout=drop, batch_first=True)
        self.do1 = nn.Dropout(drop)
        self.dp1 = DropPath(drop_path)
        self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, int(1.5*d)*2), nn.GLU(),
            nn.Linear(int(1.5*d), d),
            nn.Dropout(drop)
        )
        self.dp2 = DropPath(drop_path)

    def forward(self, x, key_padding_mask):
        q = self.n1(x)
        z = self.attn(q, q, q, key_padding_mask=key_padding_mask, need_weights=False)[0]
        x = x + self.dp1(self.do1(z))
        x = x + self.dp2(self.ff(self.n2(x)))
        return x

class TinyTransformer(nn.Module):
    # ONNX-friendly: fixed positional encoding, no runtime buffer mutation
    def __init__(self, n_feats, d=48, h=2, L=2, drop=0.2, drop_path=0.1, max_t=MAX_T):
        super().__init__()
        self.proj = nn.Linear(n_feats, d)
        self.blocks = nn.ModuleList([TinyBlock(d, h, drop, drop_path) for _ in range(L)])
        self.head = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d//2),
            nn.GELU(),
            nn.Linear(d//2, 1)
        )
        self.d = d
        self.max_t = int(max_t)
        pe = sinusoid_pe(self.max_t, self.d, device=torch.device("cpu"))
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x, key_padding_mask=None):
        B, T, _ = x.shape
        if T > self.max_t:
            x = x[:, :self.max_t, :]
            T = self.max_t
            if key_padding_mask is not None:
                key_padding_mask = key_padding_mask[:, :self.max_t]

        x = self.proj(x)
        x = x + self.pe[:T].unsqueeze(0)

        for blk in self.blocks:
            x = blk(x, key_padding_mask)

        if key_padding_mask is not None:
            valid = (~key_padding_mask).sum(1).clamp(min=1).unsqueeze(-1)
            x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0).sum(1) / valid
        else:
            x = x.mean(1)

        return self.head(x).squeeze(-1)


# ---------------- metrics / efficiency ---------------
def compute_metrics(y_true, y_pred):
    y_true = y_true.astype(np.float32); y_pred = y_pred.astype(np.float32)
    mae  = float(np.mean(np.abs(y_pred - y_true)))
    mse  = float(np.mean((y_pred - y_true)**2))
    rmse = float(np.sqrt(mse))
    yt   = y_true - y_true.mean()
    yp   = y_pred - y_pred.mean()
    denom = (np.sqrt((yt**2).sum()) * np.sqrt((yp**2).sum()) + 1e-8)
    r    = float((yt * yp).sum() / denom)
    acc1 = float(np.mean(np.abs(y_pred - y_true) <= 1.0))
    acc2 = float(np.mean(np.abs(y_pred - y_true) <= 2.0))
    return dict(MAE=mae, RMSE=rmse, R=r, Acc1=acc1, Acc2=acc2)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def state_dict_size_mb(model, tmp_path=None):
    if tmp_path is None:
        tmp_path = os.path.join(LOG_DIR, "__tmp_model__.pt")
    torch.save(model.state_dict(), tmp_path)
    mb = os.path.getsize(tmp_path) / (1024*1024)
    try: os.remove(tmp_path)
    except: pass
    return mb

def cpu_latency_ms(model, n_feats, T=150, iters=50):
    device = torch.device("cpu")
    m = model.to(device).eval()
    x = torch.randn(1, T, n_feats)
    mask = torch.zeros(1, T, dtype=torch.bool)
    with torch.no_grad():
        for _ in range(10): m(x, mask)
        t0 = time.time()
        for _ in range(iters): m(x, mask)
        t1 = time.time()
    return (t1 - t0)*1000/iters


# ----------- normalization from TRAIN only -----------
@torch.no_grad()
def compute_feature_norm(train_loader, device):
    num = 0.0; mean = None; m2 = None
    for X, _, mask, _ in train_loader:
        X = X.to(device); mask = mask.to(device)
        valid = (~mask).unsqueeze(-1)
        Xv = X.masked_select(valid).view(-1, X.shape[-1])
        if Xv.numel() == 0:
            continue
        if mean is None:
            mean = Xv.mean(dim=0)
            m2   = ((Xv - mean)**2).sum(dim=0)
            num  = Xv.shape[0]
        else:
            num_new = num + Xv.shape[0]
            delta = Xv.mean(dim=0) - mean
            mean = mean + delta * (Xv.shape[0]/num_new)
            m2 = m2 + ((Xv - mean)**2).sum(dim=0) + (delta**2) * (num * Xv.shape[0] / num_new)
            num = num_new
    std = torch.sqrt(m2.clamp_min(1e-12) / max(num-1, 1))
    return mean, std.clamp_min(1e-6)

def normalize_batch(X, mean, std):
    return (X - mean.view(1,1,-1)) / std.view(1,1,-1)


# -------------------- augmentations --------------------
def augment_batch(X, mask,
                  time_mask_p=TIME_MASK_P, time_max=TIME_MASK_MAX,
                  feat_mask_p=FEAT_MASK_P, feat_max=FEAT_MASK_MAX,
                  jitter_std=JITTER_STD, shift_max=SHIFT_MAX):
    if (time_mask_p == 0 and feat_mask_p == 0 and jitter_std == 0 and shift_max == 0):
        return X, mask

    B,T,F = X.shape
    if jitter_std and jitter_std > 0:
        X = X + torch.randn_like(X) * jitter_std
    if shift_max and shift_max > 0:
        shifts = torch.randint(-shift_max, shift_max+1, (B,), device=X.device)
        for i, s in enumerate(shifts.tolist()):
            if s != 0:
                X[i]    = torch.roll(X[i], shifts=s, dims=0)
                mask[i] = torch.roll(mask[i], shifts=s, dims=0)
    if time_mask_p and time_mask_p > 0:
        for i in range(B):
            if torch.rand(1, device=X.device) < time_mask_p:
                w = int(torch.randint(1, min(time_max, T)+1, (1,), device=X.device))
                t0 = int(torch.randint(0, max(T - w + 1, 1), (1,), device=X.device))
                X[i, t0:t0+w] = 0.0
    if feat_mask_p and feat_mask_p > 0:
        for i in range(B):
            if torch.rand(1, device=X.device) < feat_mask_p:
                w = int(torch.randint(1, min(feat_max, F)+1, (1,), device=X.device))
                f0 = int(torch.randint(0, max(F - w + 1, 1), (1,), device=X.device))
                X[i, :, f0:f0+w] = 0.0
    return X, mask


# ---------------- Plotting ----------------
def plot_epoch_summary_single(log_dir=LOG_DIR, csv_name="history_run.csv"):
    csv_path = os.path.join(log_dir, csv_name)
    if not os.path.exists(csv_path):
        return
    df = pd.read_csv(csv_path)
    e = df["epoch"].values

    plt.figure(figsize=(7.5, 5.0))
    plt.plot(e, df["train_loss"], label="Train Loss (L1)")
    plt.plot(e, df["val_rmse"], label="Val RMSE")
    plt.title("Training Loss & Validation RMSE")
    plt.xlabel("Epoch"); plt.ylabel("Value")
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "epoch_loss_summary.png"), dpi=160); plt.close()

    if "train_acc1" in df.columns and "val_acc1" in df.columns:
        plt.figure(figsize=(7.5, 5.0))
        plt.plot(e, df["train_acc1"], label="Train Acc (±1)", color='blue')
        plt.plot(e, df["val_acc1"], label="Val Acc (±1)", color='orange')
        plt.title("Overfitting Check: Train vs Val Accuracy")
        plt.xlabel("Epoch"); plt.ylabel("Accuracy (0-1)")
        plt.legend()
        plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(log_dir, "epoch_accuracy_check.png"), dpi=160); plt.close()
        print("[plot] Saved Accuracy Check")

def plot_hardware_card(log_dir=LOG_DIR, csv_name="vv_pure_results.csv"):
    full_csv = os.path.join(log_dir, csv_name)
    if not os.path.exists(full_csv):
        return
    try:
        df = pd.read_csv(full_csv, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(full_csv, encoding="latin1")
    if df.empty:
        return

    params   = df["Params"].iloc[0]
    avg_size = df["ModelMB"].iloc[0]
    avg_lat  = df["CPUms"].iloc[0]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.axis('off')
    text_str = (
        f"HARDWARE REQUIREMENTS (Inference)\n"
        f"-----------------------------------\n"
        f"Parameters      : {int(params):,}\n"
        f"Disk Size       : {avg_size:.2f} MB\n"
        f"Latency (CPU)   : {avg_lat:.2f} ms/seq\n"
        f"-----------------------------------\n"
        f"Est. RAM (B=1)  : ~{avg_size*2.5:.1f} MB"
    )
    plt.text(0.5, 0.5, text_str, ha='center', va='center', fontsize=12,
             family='monospace',
             bbox=dict(boxstyle="round,pad=1", fc="#f0f0f0", ec="black"))
    plt.savefig(os.path.join(log_dir, "hardware_requirements.png"), dpi=160); plt.close()
    print("[plot] Saved Hardware Requirements")


# ---------------- training one run ------------------
def train_one_run(model, train_loader, val_loader, device, y_mu=None, y_std=None, run_id=0):
    crit = nn.L1Loss()
    opt  = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sch  = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.3, patience=6)
    best = 1e9; wait=0; best_metrics=None

    feat_mu, feat_std = compute_feature_norm(train_loader, device)

    hist = {"epoch":[], "train_loss":[], "train_acc1":[],
            "val_mae":[], "val_rmse":[], "val_r":[], "val_acc1":[], "val_acc2":[]}

    amp_enabled = torch.cuda.is_available()
    from torch import amp as _amp
    autocast = (lambda: _amp.autocast(device_type='cuda', dtype=torch.float16, enabled=amp_enabled))
    scaler   = _amp.GradScaler('cuda') if amp_enabled else None

    for ep in range(1, EPOCHS+1):
        model.train(); tr_losses=[]; tr_accs=[]
        t0 = time.time()
        for X,y,mask,_ in train_loader:
            X,y,mask = X.to(device, non_blocking=True), y.to(device, non_blocking=True), mask.to(device, non_blocking=True)
            X = normalize_batch(X, feat_mu, feat_std)
            X, mask = augment_batch(X, mask)

            y_n = (y - y_mu)/y_std if USE_Y_ZSCORE else y
            opt.zero_grad(set_to_none=True)

            if amp_enabled:
                with autocast():
                    pred = model(X, mask)
                    loss = crit((pred - y_mu)/y_std, y_n) if USE_Y_ZSCORE else crit(pred, y)
                scaler.scale(loss).backward()
                nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
                scaler.step(opt); scaler.update()
            else:
                pred = model(X, mask)
                loss = crit((pred - y_mu)/y_std, y_n) if USE_Y_ZSCORE else crit(pred, y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
                opt.step()

            tr_losses.append(loss.item())
            with torch.no_grad():
                acc = (torch.abs(pred - y) <= 1.0).float().mean()
                tr_accs.append(acc.item())

        model.eval(); vt, vp = [], []
        with torch.no_grad():
            for X,y,mask,_ in val_loader:
                X,y,mask = X.to(device, non_blocking=True), y.to(device, non_blocking=True), mask.to(device, non_blocking=True)
                X = normalize_batch(X, feat_mu, feat_std)
                out = model(X, mask)
                vt.append(y.detach().cpu().numpy()); vp.append(out.detach().cpu().numpy())
        y_true = np.concatenate(vt); y_pred = np.concatenate(vp)
        m = compute_metrics(y_true, y_pred)
        sch.step(m["RMSE"]**2)

        hist["epoch"].append(ep)
        hist["train_loss"].append(float(np.mean(tr_losses)))
        hist["train_acc1"].append(float(np.mean(tr_accs)))
        hist["val_mae"].append(m["MAE"])
        hist["val_rmse"].append(m["RMSE"])
        hist["val_r"].append(m["R"])
        hist["val_acc1"].append(m["Acc1"])
        hist["val_acc2"].append(m["Acc2"])

        print(f"[Run{run_id}] Ep{ep:03d} L={np.mean(tr_losses):.4f} TrAcc={np.mean(tr_accs)*100:.1f}% | "
              f"Val MAE={m['MAE']:.3f} RMSE={m['RMSE']:.3f} Acc±1={m['Acc1']*100:.1f}% | "
              f"T={time.time()-t0:.1f}s")

        if m["RMSE"] < best - SAVE_EPS:
            best = m["RMSE"]; wait=0; best_metrics = m.copy()
            torch.save(model.state_dict(), os.path.join(LOG_DIR, "best_run.pt"))
        else:
            wait += 1
            if wait >= PATIENCE:
                break

    pd.DataFrame(hist).to_csv(os.path.join(LOG_DIR, "history_run.csv"), index=False)
    return best_metrics, feat_mu, feat_std


def eval_on_loader(model, loader, device, feat_mu, feat_std):
    model.eval(); vt, vp = [], []
    with torch.no_grad():
        for X,y,mask,_ in loader:
            X,y,mask = X.to(device, non_blocking=True), y.to(device, non_blocking=True), mask.to(device, non_blocking=True)
            X = normalize_batch(X, feat_mu, feat_std)
            out = model(X, mask)
            vt.append(y.detach().cpu().numpy())
            vp.append(out.detach().cpu().numpy())
    y_true = np.concatenate(vt); y_pred = np.concatenate(vp)
    return compute_metrics(y_true, y_pred), y_true, y_pred


# ----------------------- main ------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=DATA_DEFAULT_DIR)
    parser.add_argument("--pattern", type=str, default="*.npz")
    parser.add_argument("--test_ratio", type=float, default=0.30)  # 30% test
    parser.add_argument("--val_ratio", type=float, default=0.10)   # 10% of remaining for val
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True

    if platform.system().lower().startswith("win"):
        num_workers = 0; pin_memory = False
    else:
        num_workers = max(2, (os.cpu_count() or 4)//2); pin_memory = True

    print(f"Device: {device}")
    print(f"Loading Dataset: {args.data_dir}")
    ds = NPZSeqDataset(args.data_dir, pattern=args.pattern)

    idx_all = np.arange(len(ds))

    # 70/30 split => train_idx, test_idx
    train_idx, test_idx = train_test_split(
        idx_all, test_size=args.test_ratio, random_state=42, shuffle=True
    )

    # from train, create a small val split for early stopping
    if len(train_idx) >= 5:
        train_idx, val_idx = train_test_split(
            train_idx, test_size=args.val_ratio, random_state=42, shuffle=True
        )
    else:
        val_idx = train_idx

    # y stats from TRAIN only
    y_train_all = [ds.ys[ds.index[i][0]][ds.index[i][1]] for i in train_idx]
    y_train_all = np.array(y_train_all, dtype=np.float32)
    y_mu = float(y_train_all.mean())
    y_std = float(y_train_all.std() + 1e-6)

    train_loader = DataLoader(Subset(ds, train_idx), batch_size=BATCH, shuffle=True,
                              collate_fn=collate_pad, num_workers=num_workers, pin_memory=pin_memory)
    val_loader   = DataLoader(Subset(ds, val_idx), batch_size=BATCH, shuffle=False,
                              collate_fn=collate_pad, num_workers=num_workers, pin_memory=pin_memory)
    test_loader  = DataLoader(Subset(ds, test_idx), batch_size=BATCH, shuffle=False,
                              collate_fn=collate_pad, num_workers=num_workers, pin_memory=pin_memory)

    model = TinyTransformer(
        n_feats=ds.n_feats,
        d=D_MODEL, h=N_HEADS, L=N_LAYERS,
        drop=DROP, drop_path=DROP_PATH,
        max_t=MAX_T
    ).to(device)

    params = count_params(model)
    sizeMB = state_dict_size_mb(model)
    lat_ms = cpu_latency_ms(model, n_feats=ds.n_feats, T=MAX_T, iters=50)

    best_val, feat_mu, feat_std = train_one_run(
        model, train_loader, val_loader, device, y_mu=y_mu, y_std=y_std, run_id=1
    )

    # SAVE NORMALIZATION STATS (needed for export_model.py)
    torch.save(feat_mu.detach().cpu(), os.path.join(LOG_DIR, "feat_mu.pt"))
    torch.save(feat_std.detach().cpu(), os.path.join(LOG_DIR, "feat_std.pt"))
    torch.save(torch.tensor(y_mu), os.path.join(LOG_DIR, "y_mu.pt"))
    torch.save(torch.tensor(y_std), os.path.join(LOG_DIR, "y_std.pt"))
    print("Saved normalization stats (feat_mu, feat_std, y_mu, y_std)")

    best_path = os.path.join(LOG_DIR, "best_run.pt")
    if os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))

    print("\nEvaluating...")
    train_metrics, _, _ = eval_on_loader(model, train_loader, device, feat_mu, feat_std)
    val_metrics, _, _   = eval_on_loader(model, val_loader, device, feat_mu, feat_std)
    test_metrics, t_true, t_pred = eval_on_loader(model, test_loader, device, feat_mu, feat_std)

    row = {
        "train_MAE": train_metrics["MAE"],
        "val_MAE": val_metrics["MAE"],
        "test_MAE": test_metrics["MAE"],
        "Params": float(params),
        "ModelMB": float(sizeMB),
        "CPUms": float(lat_ms),
    }
    out_csv = os.path.join(LOG_DIR, "results_70_30.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader()
        w.writerow(row)
    print(f"\nSaved summary to: {out_csv}")

    plot_epoch_summary_single(LOG_DIR)
    plot_hardware_card(LOG_DIR, csv_name="results_70_30.csv")

    if PLOT:
        plt.figure(figsize=(6,6))
        plt.scatter(t_true, t_pred, alpha=0.4, s=12)
        plt.xlabel("True"); plt.ylabel("Pred"); plt.title("Test Fit (70/30)")
        plt.grid(True)
        mn = min(t_true.min(), t_pred.min())
        mx = max(t_true.max(), t_pred.max())
        plt.plot([mn, mx], [mn, mx], 'r--', linewidth=2)
        plt.savefig(os.path.join(LOG_DIR, "test_scatter_70_30.png"), dpi=160)
        plt.close()

    print("\nDone. Results in:", LOG_DIR)


if __name__ == "__main__":
    main()
