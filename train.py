import argparse, csv, time, torch
from torch import nn, optim
from pathlib import Path
from utils import set_seed, get_loaders, evaluate, MNIST_MEAN, MNIST_STD

def build_model(name):
    if name=="cnn":  from models.cnn import SimpleCNN; return SimpleCNN()
    if name=="lstm": from models.lstm import RowLSTM;  return RowLSTM()
    raise ValueError("Unknown model")

def _autodetect_csv():
    return "mnist_train.csv", "mnist_test.csv", True

def train_one(tag, args):
    set_seed(args.seed)
    dev = torch.device("cuda" if (args.device in ("auto","cuda") and torch.cuda.is_available()) else "cpu")
    model = build_model(tag).to(dev)
    train_ld, val_ld, test_ld = get_loaders(batch_size=args.batch_size, val_split=0.1, num_workers=2,
                                            train_csv=args.train_csv, test_csv=args.test_csv, has_header=args.has_header)
    outdir = Path(args.outdir)/tag; outdir.mkdir(parents=True, exist_ok=True)
    params = sum(p.numel() for p in model.parameters())
    print(f"[{tag.upper()}] start | device={dev} | params={params}")



    crit = nn.CrossEntropyLoss(); opt = optim.Adam(model.parameters(), lr=args.lr)
    mp = outdir/"metrics.csv"
    with open(mp,"w",newline="") as f:
        csv.writer(f).writerow(["epoch","train_loss","train_acc","val_loss","val_acc","seconds"])

    t_all = time.perf_counter(); epoch_times=[]
    for ep in range(1, args.epochs+1):
        model.train(); total=correct=0; loss_sum=0.0; t0=time.perf_counter()
        for x,y in train_ld:
            x,y = x.to(dev), y.to(dev); opt.zero_grad()
            logits = model(x)
            loss = (-(torch.nn.functional.one_hot(y,10).float()*torch.log_softmax(logits,1)).sum(1).mean()
                    if args.use_onehot_loss else crit(logits,y))
            loss.backward(); opt.step()
            n=y.size(0); loss_sum+=loss.item()*n; correct+=(logits.argmax(1)==y).sum().item(); total+=n
        tl, ta = loss_sum/total, correct/total
        vl, va = evaluate(model, val_ld, dev, crit)
        dt = time.perf_counter()-t0; epoch_times.append(dt)
        with open(mp,"a",newline="") as f:
            csv.writer(f).writerow([ep,f"{tl:.4f}",f"{ta:.4f}",f"{vl:.4f}",f"{va:.4f}",f"{dt:.2f}"])
        torch.save(model.state_dict(), outdir/f"epoch{ep}.pt")
        print(f"[{tag.upper()}] epoch {ep}/{args.epochs} | train_loss {tl:.4f} | val_loss {vl:.4f} | val_acc {va:.4f} | {dt:.2f}s")

    total_s = time.perf_counter()-t_all
    avg_s = sum(epoch_times)/len(epoch_times)
    test_l, test_a = evaluate(model, test_ld, dev, crit)

    with open(outdir/"summary.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["model","params","epochs","avg_sec_per_epoch","total_sec","test_acc","test_loss","device"])
        w.writerow([tag,params,args.epochs,f"{avg_s:.2f}",f"{total_s:.2f}",f"{test_a:.4f}",f"{test_l:.4f}",str(dev)])
    with open(outdir/"test.txt","w") as f:
        f.write(f"test_loss={test_l:.4f}\n"
                f"test_acc={test_a:.4f}\n"
                f"params={params}\n"
                f"avg_sec_per_epoch={avg_s:.2f}\n"
                f"total_sec={total_s:.2f}\n")
    print(f"[{tag.upper()}] done | acc {test_a:.4f} | loss {test_l:.4f} | avg_s/ep {avg_s:.2f} | total_s {total_s:.2f}")
    return {"model":tag,"params":params,"epochs":args.epochs,"avg_sec_per_epoch":avg_s,"total_sec":total_s,"test_acc":test_a,"test_loss":test_l,"device":str(dev)}

def main():
    tr,te,hh=_autodetect_csv()
    ap=argparse.ArgumentParser()
    ap.add_argument("--model", choices=["cnn","lstm","both"], default="both")
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
    args=ap.parse_args()
    order=["cnn","lstm"] if args.model=="both" else [args.model]
    results=[train_one(m,args) for m in order]
    comb=Path(args.outdir)/"combined_summary.csv"
    with open(comb,"w",newline="") as f:
        w=csv.writer(f); w.writerow(["model","params","epochs","avg_sec_per_epoch","total_sec","test_acc","test_loss","device"])
        [w.writerow([r["model"],r["params"],r["epochs"],f"{r['avg_sec_per_epoch']:.2f}",f"{r['total_sec']:.2f}",f"{r['test_acc']:.4f}",f"{r['test_loss']:.4f}",r["device"]]) for r in results]
    print(f"[SUMMARY] {comb}")

if __name__=="__main__": main()
