import os, csv, matplotlib.pyplot as plt

FIGSIZE = (10, 6)
MARGIN_FRAC = 0.05
MIN_SPAN = {"acc": 0.005, "loss": 0.02, "sec": 0.50}

def _tight_ylim(vals, kind):
    if not vals: return None
    vmin, vmax = min(vals), max(vals)
    span = vmax - vmin
    if span < MIN_SPAN[kind]:
        mid = 0.5*(vmin+vmax); half = 0.5*MIN_SPAN[kind]
        vmin, vmax = mid-half, mid+half
    pad = MARGIN_FRAC * max(1e-9, vmax - vmin)
    return (vmin - pad, vmax + pad)

def read_metrics(path):
    if not os.path.exists(path): return None
    rows=[]
    with open(path) as f:
        r=csv.DictReader(f)
        for row in r:
            rows.append({
                "epoch": int(row["epoch"]),
                "train_loss": float(row["train_loss"]),
                "train_acc": float(row["train_acc"]),
                "val_loss": float(row["val_loss"]),
                "val_acc": float(row["val_acc"]),
                "seconds": float(row["seconds"]),
            })
    return rows

def read_summary(path):
    if not os.path.exists(path): return None
    with open(path) as f:
        r=csv.DictReader(f)
        for row in r:
            return {
                "model": row["model"],
                "params": float(row["params"]),
                "epochs": int(row["epochs"]),
                "avg_sec_per_epoch": float(row["avg_sec_per_epoch"]),
                "total_sec": float(row["total_sec"]),
                "test_acc": float(row["test_acc"]),
                "test_loss": float(row["test_loss"]),
                "device": row.get("device",""),
            }
    return None

def _plot_line(xs, ys, label):
    plt.plot(xs, ys, marker="o", label=label)

def line_overlay(rows_a, rows_b, key, title, ylabel, kind, labels=("CNN","LSTM")):
    plt.figure(figsize=FIGSIZE); plt.title(title)
    all_vals=[]
    if rows_a:
        e=[r["epoch"] for r in rows_a]; ya=[r[key] for r in rows_a]
        _plot_line(e, ya, labels[0]); all_vals+=ya
    if rows_b:
        e=[r["epoch"] for r in rows_b]; yb=[r[key] for r in rows_b]
        _plot_line(e, yb, labels[1]); all_vals+=yb
    yl=_tight_ylim(all_vals, kind)
    if yl: plt.ylim(*yl)
    plt.xlabel("epoch"); plt.ylabel(ylabel); plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()

def bar_compare(d, title, ylabel, kind):
    if not d: return
    labels=list(d.keys()); vals=[d[k] for k in labels]
    plt.figure(figsize=FIGSIZE); plt.title(title)
    plt.bar(labels, vals)
    yl=_tight_ylim(vals, kind)
    if yl: plt.ylim(*yl)
    plt.ylabel(ylabel); plt.grid(axis="y", alpha=0.3); plt.tight_layout()

def main():
    cm = os.path.join("runs","cnn","metrics.csv")
    lm = os.path.join("runs","lstm","metrics.csv")
    cs = os.path.join("runs","cnn","summary.csv")
    ls = os.path.join("runs","lstm","summary.csv")

    cnn_rows = read_metrics(cm)
    lstm_rows = read_metrics(lm)
    cnn_sum  = read_summary(cs)
    lstm_sum = read_summary(ls)

    if not (cnn_rows or lstm_rows):
        print("No metrics found. Run train.py first."); return

    # KEEP: val_acc, val_loss, seconds overlays
    line_overlay(cnn_rows, lstm_rows, "val_acc",  "Validation Accuracy — CNN vs LSTM", "accuracy", "acc")
    line_overlay(cnn_rows, lstm_rows, "val_loss", "Validation Loss — CNN vs LSTM", "loss", "loss")
    line_overlay(cnn_rows, lstm_rows, "seconds",  "Seconds per epoch — CNN vs LSTM", "seconds", "sec")

    # KEEP: bars avg/sec and total/sec
    avg = {}; tot = {}
    if cnn_sum: avg["CNN"]=cnn_sum["avg_sec_per_epoch"]; tot["CNN"]=cnn_sum["total_sec"]
    if lstm_sum: avg["LSTM"]=lstm_sum["avg_sec_per_epoch"]; tot["LSTM"]=lstm_sum["total_sec"]
    bar_compare(avg, "Average seconds per epoch — CNN vs LSTM", "seconds", "sec")
    bar_compare(tot, "Total wall-clock seconds — CNN vs LSTM", "seconds", "sec")

    # KEEP: bars test acc and test loss
    test_acc = {}; test_loss = {}
    if cnn_sum: test_acc["CNN"]=cnn_sum["test_acc"]; test_loss["CNN"]=cnn_sum["test_loss"]
    if lstm_sum: test_acc["LSTM"]=lstm_sum["test_acc"]; test_loss["LSTM"]=lstm_sum["test_loss"]
    bar_compare(test_acc,  "Test accuracy — CNN vs LSTM", "accuracy", "acc")
    bar_compare(test_loss, "Test loss — CNN vs LSTM", "loss", "loss")

    # Console summary
    def ps(s):
        if not s: return
        print(f"{s['model'].upper():5s}  acc={s['test_acc']:.4f}  loss={s['test_loss']:.4f}  "
              f"avg_s/ep={s['avg_sec_per_epoch']:.2f}  total_s={s['total_sec']:.2f}  dev={s['device']}")
    ps(cnn_sum); ps(lstm_sum)

    plt.show()

if __name__ == "__main__":
    main()
