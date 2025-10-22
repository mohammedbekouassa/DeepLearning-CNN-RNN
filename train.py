# train.py — CNN then RNN (LSTM) with key-press gates; logs to results.csv
import time, argparse, csv, torch, torch.nn as nn, torch.optim as optim
import torch.backends.cudnn as cudnn
from torch import amp  # new AMP API
from utils import load_csv_pair, wait_key
from models.cnn import SmallCNN
from models.rnn import RowLSTM

# speed: let cuDNN pick fastest convs for fixed shapes
cudnn.benchmark = True
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

def pick_device(device_arg: str) -> str:
    device_arg = device_arg.lower()
    if device_arg == "auto":
        if torch.cuda.is_available():
            print(f"🟢 Using CUDA GPU: {torch.cuda.get_device_name(0)}")
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("🟠 Using Apple Metal (MPS)")
            return "mps"
        print("🔵 Using CPU")

        return "cpu"
    if device_arg == "cuda":
        if torch.cuda.is_available():
            print(f"🟢 Using CUDA GPU: {torch.cuda.get_device_name(0)}")
            return "cuda"
        print("⚠️ CUDA requested but not available — falling back to CPU.")
        return "cpu"
    if device_arg == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("🟠 Using Apple Metal (MPS)")
            return "mps"
        print("⚠️ MPS requested but not available — falling back to CPU.")
        return "cpu"
    # cpu
    print("🔵 Using CPU")
    return "cpu"

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); crit = nn.CrossEntropyLoss()
    n = loss_sum = correct = 0
    for X, y in loader:
        X = X.to(device, non_blocking=(device=="cuda"))
        y = y.to(device, non_blocking=(device=="cuda"))
        logits = model(X); loss = crit(logits, y)
        loss_sum += loss.item() * X.size(0)
        correct  += (logits.argmax(1) == y).sum().item()
        n += X.size(0)
    return loss_sum / n, correct / n

def train_one(name, model, trL, teL, epochs, lr, device):
    use_amp = (device == "cuda")
    model.to(device)
    if use_amp:
        model = model.to(memory_format=torch.channels_last)

    opt = optim.AdamW(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    scaler = amp.GradScaler('cuda', enabled=use_amp)

    rows = []; t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        model.train(); ep_t = time.perf_counter()
        for X, y in trL:
            X = X.to(device, non_blocking=use_amp)
            y = y.to(device, non_blocking=use_amp)
            if use_amp:
                X = X.to(memory_format=torch.channels_last)

            opt.zero_grad(set_to_none=True)
            with amp.autocast('cuda', enabled=use_amp):
                logits = model(X)
                loss = crit(logits, y)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(opt); scaler.update()
            else:
                loss.backward(); opt.step()

        te_loss, te_acc = evaluate(model, teL, device)
        dt = time.perf_counter() - ep_t
        rows.append({"model": name, "epoch": ep, "test_acc": te_acc,
                     "test_loss": te_loss, "train_time_s": dt})
        print(f"[{name}] epoch {ep}: acc={te_acc:.4f} loss={te_loss:.4f} time={dt:.1f}s")
    return rows, time.perf_counter() - t0

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs",   type=int,   default=5)
    ap.add_argument("--bs",       type=int,   default=128)
    ap.add_argument("--lr_cnn",   type=float, default=1e-3)
    ap.add_argument("--lr_rnn",   type=float, default=1e-3)
    ap.add_argument("--out",      type=str,   default="results.csv")
    ap.add_argument("--device",   type=str,   default="auto",
                    choices=["auto","cpu","cuda","mps"],
                    help="Choose compute device (default: auto)")
    args = ap.parse_args()

    device = pick_device(args.device)
    print("PyTorch:", torch.__version__)
    if device == "cuda":
        print("CUDA (compiled):", torch.version.cuda)

    # On CUDA: try larger batch + enable pinned memory in loaders
    bs  = args.bs if device != "cuda" else max(args.bs, 256)
    pin = (device == "cuda")
    trL, teL = load_csv_pair(bs=bs, pin=pin, workers=2)

    print("\n# ===== CNN TRAINING =====")
    wait_key("▶ Press Enter to start CNN… ")
    cnn_rows, t_cnn = train_one("cnn", SmallCNN(), trL, teL, args.epochs, args.lr_cnn, device)

    print("\n# ===== RNN (LSTM) TRAINING =====")
    wait_key("▶ Press Enter to start RNN… ")
    rnn_rows, t_rnn = train_one("rnn", RowLSTM(), trL, teL, args.epochs, args.lr_rnn, device)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model","epoch","test_acc","test_loss","train_time_s"])
        w.writeheader(); w.writerows(cnn_rows); w.writerows(rnn_rows)

    ca, cl = cnn_rows[-1]["test_acc"], cnn_rows[-1]["test_loss"]
    ra, rl = rnn_rows[-1]["test_acc"], rnn_rows[-1]["test_loss"]
    print("\n# ===== SUMMARY (last epoch) =====")
    print(f"CNN       -> acc={ca:.4f} loss={cl:.4f} time={t_cnn:.1f}s")
    print(f"RNN(LSTM) -> acc={ra:.4f} loss={rl:.4f} time={t_rnn:.1f}s")
    print(f"Saved metrics to {args.out}")
