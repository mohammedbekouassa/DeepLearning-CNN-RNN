import argparse, os, csv, time, torch
from torch import nn, optim
from pathlib import Path
from utils import set_seed, get_loaders, evaluate, now, MNIST_MEAN, MNIST_STD

def build_model(name):
    if name == "cnn":
        from models.cnn import SimpleCNN; return SimpleCNN()
    if name == "lstm":
        from models.lstm import RowLSTM;  return RowLSTM()
    raise ValueError("Unknown model")

def _autodetect_csv():
    tr, te = "mnist_train.csv", "mnist_test.csv"
    if not (os.path.exists(tr) and os.path.exists(te)): return None, None, False
    def has_header(p):
        with open(p, "r", encoding="utf-8") as f:
            first = f.readline().strip().split(",")
        if not first: return False
        if first[0].lower()=="label": return True
        try: float(first[0]); return len(first)!=785
        except ValueError: return True
    return tr, te, has_header(tr)

def train_one(tag, args):
    set_seed(args.seed)
    device = (torch.device("cuda") if (args.device=="auto" and torch.cuda.is_available())
              else torch.device("mps") if (args.device=="auto" and hasattr(torch.backends,"mps") and torch.backends.mps.is_available())
              else torch.device(args.device if args.device!="auto" else "cpu"))
    model = build_model(tag).to(device)
    train_ld, val_ld, test_ld = get_loaders(batch_size=args.batch_size, val_split=0.1, num_workers=2,
                                            train_csv=args.train_csv, test_csv=args.test_csv, has_header=args.has_header)

    outdir = Path(args.outdir) / tag
    outdir.mkdir(parents=True, exist_ok=True)
    params = sum(p.numel() for p in model.parameters())

    print(f"[{tag.upper()}] start | device={device} | params={params}")

    # Preuve one-hot (exigence)
    if args.dump_onehot:
        xb, yb = next(iter(train_ld))
        oh = torch.nn.functional.one_hot(yb, num_classes=10).float()
        torch.save({"y_indices": yb[:256], "y_onehot": oh[:256],
                    "note": f"Normalized mean={MNIST_MEAN}, std={MNIST_STD}"},
                   outdir/"onehot_sample.pt")
        with open(outdir/"onehot_preview.csv","w",newline="") as f:
            w = csv.writer(f); w.writerow(["label"]+[f"c{i}" for i in range(10)])
            for i in range(min(20,yb.size(0))):
                w.writerow([int(yb[i])]+[int(v) for v in oh[i].tolist()])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    metrics_path = outdir/"metrics.csv"
    with open(metrics_path,"w",newline="") as f:
        csv.writer(f).writerow(["epoch","train_loss","train_acc","val_loss","val_acc","seconds"])

    wall_start = time.perf_counter()
    epoch_times = []

    for ep in range(1, args.epochs+1):
        model.train()
        total=correct=0; loss_sum=0.0
        t0 = now()
        for x,y in train_ld:
            x,y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            if args.use_onehot_loss:
                oh = torch.nn.functional.one_hot(y, num_classes=10).float()
                loss = -(oh * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()
            else:
                loss = criterion(logits, y)
            loss.backward(); optimizer.step()
            loss_sum += loss.item()*y.size(0)
            correct += (logits.argmax(1)==y).sum().item()
            total += y.size(0)

        train_loss = loss_sum/total
        train_acc  = correct/total
        val_loss, val_acc = evaluate(model, val_ld, device, criterion)
        dt = now() - t0
        epoch_times.append(dt)

        with open(metrics_path,"a",newline="") as f:
            csv.writer(f).writerow([ep, f"{train_loss:.4f}", f"{train_acc:.4f}",
                                    f"{val_loss:.4f}", f"{val_acc:.4f}", f"{dt:.2f}"])
        torch.save(model.state_dict(), outdir/f"epoch{ep}.pt")

        # LIGNE UNIQUE, propre
        print(f"[{tag.upper()}] epoch {ep}/{args.epochs} | "
              f"train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | "
              f"val_acc {val_acc:.4f} | {dt:.2f}s")

    wall_total = time.perf_counter()-wall_start
    avg_epoch = sum(epoch_times)/len(epoch_times)
    test_loss, test_acc = evaluate(model, test_ld, device, criterion)

    with open(outdir/"summary.csv","w",newline="") as f:
        w = csv.writer(f)
        w.writerow(["model","params","epochs","avg_sec_per_epoch","total_sec","test_acc","test_loss","device"])
        w.writerow([tag, params, args.epochs, f"{avg_epoch:.2f}", f"{wall_total:.2f}",
                    f"{test_acc:.4f}", f"{test_loss:.4f}", str(device)])

    with open(outdir/"test.txt","w") as f:
        f.write(f"test_loss={test_loss:.4f}\n")
        f.write(f"test_acc={test_acc:.4f}\n")
        f.write(f"params={params}\n")
        f.write(f"avg_sec_per_epoch={avg_epoch:.2f}\n")
        f.write(f"total_sec={wall_total:.2f}\n")

    print(f"[{tag.upper()}] done | acc {test_acc:.4f} | loss {test_loss:.4f} | "
          f"avg_s/ep {avg_epoch:.2f} | total_s {wall_total:.2f}")

    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    return {"model":tag,"params":params,"epochs":args.epochs,
            "avg_sec_per_epoch":avg_epoch,"total_sec":wall_total,
            "test_acc":test_acc,"test_loss":test_loss,"device":str(device)}

def main():
    tr, te, hh = _autodetect_csv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["cnn","lstm","both"], default="both")  # CNN puis LSTM
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", type=str, default="runs")
    ap.add_argument("--device", type=str, default="auto")
    ap.add_argument("--train-csv", type=str, default=tr)
    ap.add_argument("--test-csv",  type=str, default=te)
    ap.add_argument("--has-header", action="store_true", default=bool(hh))
    ap.add_argument("--dump-onehot", action="store_true", default=True)
    ap.add_argument("--use-onehot-loss", action="store_true", default=False)
    args = ap.parse_args()

    order = ["cnn","lstm"] if args.model=="both" else [args.model]
    results = [train_one(m, args) for m in order]

    comb = Path(args.outdir)/"combined_summary.csv"
    with open(comb,"w",newline="") as f:
        w = csv.writer(f)
        w.writerow(["model","params","epochs","avg_sec_per_epoch","total_sec","test_acc","test_loss","device"])
        for r in results:
            w.writerow([r["model"], r["params"], r["epochs"],
                        f"{r['avg_sec_per_epoch']:.2f}", f"{r['total_sec']:.2f}",
                        f"{r['test_acc']:.4f}", f"{r['test_loss']:.4f}", r["device"]])
    print(f"[SUMMARY] {comb}")

if __name__ == "__main__":
    main()
