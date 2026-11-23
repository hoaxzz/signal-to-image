import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import time
from scipy import signal as sp_signal

from hilbert_png_codec import hilbert_indices
from hilbert_png_denoiser import HilbertUNet, load_splits

def raster_indices(img_size=32):
    idx = np.arange(img_size * img_size, dtype=np.int64)
    return idx // img_size, idx % img_size

def _morton_d2xy(d: int):
    x, y = 0, 0
    for bit in range(16):
        x |= ((d >> (2 * bit)) & 1) << bit
        y |= ((d >> (2 * bit + 1)) & 1) << bit
    return x, y

def morton_indices(img_size=32):
    coords = [_morton_d2xy(d) for d in range(img_size * img_size)]
    xs, ys = zip(*coords)
    return np.asarray(ys, dtype=np.int64), np.asarray(xs, dtype=np.int64)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _sig_to_grid(xb, rows, cols, n=512):
    bsz = xb.shape[0]
    grid = torch.zeros(bsz, 2, 32, 32, device=xb.device, dtype=xb.dtype)
    grid[:, 0, rows[:n], cols[:n]] = xb[:, :n]
    grid[:, 1, rows[:n], cols[:n]] = 1.0
    return grid

def train_layout(layout_name, index_fn, x_tr, y_tr, x_va, y_va, x_te, y_te, epochs=5):
    rows_np, cols_np = index_fn(32)
    rows = torch.from_numpy(rows_np).long().to(device)
    cols = torch.from_numpy(cols_np).long().to(device)

    model = HilbertUNet(base=24).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loader = DataLoader(TensorDataset(torch.from_numpy(x_tr), torch.from_numpy(y_tr)), batch_size=256, shuffle=True)
    x_va_t = torch.from_numpy(x_va)

    best_loss = float("inf")
    best_epoch = 0
    best_state = None
    t0 = time.perf_counter()

    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            inp = _sig_to_grid(xb, rows, cols, n=512)
            pred = model(inp)[:, 0, rows[:512], cols[:512]]
            loss = F.mse_loss(pred, yb) + 0.8 * F.l1_loss(
                torch.fft.rfft(pred, dim=-1, norm="ortho").abs(),
                torch.fft.rfft(yb, dim=-1, norm="ortho").abs()
            )
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        # Val check
        model.eval()
        with torch.no_grad():
            val_preds = []
            for vs in range(0, len(x_va_t), 512):
                vb = x_va_t[vs:vs+512].to(device)
                vinp = _sig_to_grid(vb, rows, cols, n=512)
                val_preds.append(model(vinp)[:, 0, rows[:512], cols[:512]])
            vp = torch.cat(val_preds, dim=0)
            y_va_t = torch.from_numpy(y_va).to(device)
            val_mse = F.mse_loss(vp, y_va_t).item()
            if val_mse < best_loss:
                best_loss = val_mse
                best_epoch = ep
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    total_time = time.perf_counter() - t0
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()

    # Test eval on -5 dB
    x_te_t = torch.from_numpy(x_te).to(device)
    with torch.no_grad():
        test_inp = _sig_to_grid(x_te_t, rows, cols, n=512)
        test_pred = model(test_inp)[:, 0, rows[:512], cols[:512]].cpu().numpy()

    # Metrics
    clean = y_te
    est = test_pred
    diff = est - clean
    norm_clean = np.linalg.norm(clean, axis=1) + 1e-12
    norm_diff = np.linalg.norm(diff, axis=1)
    rrmse_t = float(np.mean(norm_diff / norm_clean))

    # Spectral RRMSE via Welch PSD
    def welch_psd(s):
        _, p = sp_signal.welch(s, fs=256, nperseg=64, axis=-1)
        return p
    psd_c = welch_psd(clean)
    psd_e = welch_psd(est)
    rrmse_s_welch = float(np.mean(np.linalg.norm(psd_c - psd_e, axis=-1) / (np.linalg.norm(psd_c, axis=-1) + 1e-12)))

    # CC
    c_cen = clean - clean.mean(axis=1, keepdims=True)
    e_cen = est - est.mean(axis=1, keepdims=True)
    cc_vec = np.sum(c_cen * e_cen, axis=1) / (np.linalg.norm(c_cen, axis=1) * np.linalg.norm(e_cen, axis=1) + 1e-12)
    cc = float(np.mean(cc_vec))

    # Delta SNR
    norm_noisy = np.linalg.norm(x_te - clean, axis=1) + 1e-12
    snr_in = float(np.mean(20.0 * np.log10(norm_clean / norm_noisy)))
    snr_out = float(np.mean(20.0 * np.log10(norm_clean / (norm_diff + 1e-12))))
    delta_snr = snr_out - snr_in

    print(f"[{layout_name}] Converged Ep: {best_epoch}/{epochs} | Time: {total_time:.1f}s | "
          f"RRMSE_t: {rrmse_t:.4f} | RRMSE_s_Welch: {rrmse_s_welch:.4f} | CC: {cc:.4f} | Delta_SNR: +{delta_snr:.2f} dB")
    return {
        "Layout": layout_name,
        "Epochs_to_Converge": best_epoch,
        "Training_Time_s": round(total_time, 1),
        "RRMSE_temporal": round(rrmse_t, 4),
        "RRMSE_spectral_welch": round(rrmse_s_welch, 4),
        "CC": round(cc, 4),
        "Delta_SNR_dB": round(delta_snr, 2),
    }

splits, test_by_snr = load_splits("EOG")
x_tr, y_tr = splits["train"]
x_va, y_va = splits["val"]
x_te, y_te = test_by_snr[-5]

# Fast test on 3 epochs
for name, fn in [("Hilbert Curve (Anda)", hilbert_indices), ("Morton / Z-Order", morton_indices), ("Raster-Scan", raster_indices)]:
    train_layout(name, fn, x_tr, y_tr, x_va, y_va, x_te, y_te, epochs=3)