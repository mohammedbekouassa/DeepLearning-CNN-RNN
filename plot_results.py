# plot_results.py
import csv, matplotlib.pyplot as plt

rows = []
with open("results.csv", "r", newline="") as f:
    for d in csv.DictReader(f):
        rows.append({
            "model": d["model"],
            "epoch": int(d["epoch"]),
            "test_acc": float(d["test_acc"]),
            "test_loss": float(d["test_loss"]),
            "train_time_s": float(d["train_time_s"]),
        })

by = {}
for r in rows:
    by.setdefault(r["model"], []).append(r)
for m in by: by[m].sort(key=lambda r: r["epoch"])

# Acc
plt.figure()
for m,lst in by.items():
    plt.plot([r["epoch"] for r in lst], [r["test_acc"] for r in lst], label=m)
plt.xlabel("Epoch"); plt.ylabel("Test Acc"); plt.title("Accuracy vs Epoch"); plt.legend()
plt.tight_layout(); plt.savefig("acc_vs_epoch.png", dpi=160)

# Loss
plt.figure()
for m,lst in by.items():
    plt.plot([r["epoch"] for r in lst], [r["test_loss"] for r in lst], label=m)
plt.xlabel("Epoch"); plt.ylabel("Test Loss"); plt.title("Loss vs Epoch"); plt.legend()
plt.tight_layout(); plt.savefig("loss_vs_epoch.png", dpi=160)

# Time
plt.figure()
labels = list(by.keys())
times  = [sum(r["train_time_s"] for r in by[m]) for m in labels]
plt.bar(labels, times)
plt.title("Training time (sum of epochs)"); plt.ylabel("seconds")
plt.tight_layout(); plt.savefig("train_time_bar.png", dpi=160)

print("Saved: acc_vs_epoch.png, loss_vs_epoch.png, train_time_bar.png")
