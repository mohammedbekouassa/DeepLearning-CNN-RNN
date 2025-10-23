# MNIST — Comparative Study (CNN vs LSTM)

Single-file README. Includes: objectives, data format, preprocessing, models, training, plots, **and your exact CSV results**.

---

## 1) Objective
Compare two architectures for MNIST classification:
- **CNN**
- **LSTM** (image as row sequence)

Deliverables per exercise: correct preprocessing (scale → normalize → reshape → **one-hot proof**), train and evaluate on test set, compare **accuracy/loss/time**, provide **plots**, **tables**, and **discussion**, publish with a clear README.

---

## 2) Dataset and Row Format
**Files**
- `mnist_train.csv`
- `mnist_test.csv`

**Row format**
```
label, pixel0, pixel1, ..., pixel783
```
- One sample per row
- `label` ∈ {0..9}
- 784 grayscale pixels (28×28), values ∈ [0,255]

**Dataset size**
- Train ≈ 60,000
- Test ≈ 10,000

---

## 3) Data Preparation (exact steps)
1. **Load** CSVs to tensors.
2. **Scale**: `X = X / 255.0`
3. **Normalize** (MNIST stats): `X = (X - 0.1307) / 0.3081`
4. **Reshape**
   - CNN: `[B, 1, 28, 28]`
   - LSTM: `[B, 28, 28]` (28 time steps × 28 features)
5. **Labels**
   - Train with integer labels (`CrossEntropyLoss`).
   - **One-hot requirement**: generate and save proof files under `runs/<model>/onehot_preview.csv` and `onehot_sample.pt`.
6. **Split**: 90% train / 10% val.
7. **Dataloaders**: batch=128, shuffle train=True; val/test not shuffled; X `float32`, y `long`.

---

## 4) Models
**CNN (SimpleCNN)**
```
Conv2d(1→32,3) → ReLU → MaxPool(2)
Conv2d(32→64,3) → ReLU → MaxPool(2)
Flatten → Linear(7*7*64→128) → ReLU → Linear(128→10)
```
Params measured: **421,642**.

**LSTM (RowLSTM)**
```
LSTM(input_size=28, hidden=128, num_layers=1)
Last hidden → Linear(128→10)
```
Params measured: **82,186**.

---

## 5) Training & Evaluation
- Optimizer: Adam(lr=1e-3)
- Loss: CrossEntropyLoss
- Epochs: 5 (configurable)
- Batch: 128
- Device: auto (cuda › mps › cpu)
- Fixed seed

**Timing**
- Per-epoch seconds → `runs/<model>/metrics.csv`
- `avg_sec_per_epoch` and `total_sec` → `runs/<model>/summary.csv`
- Combined table → `runs/combined_summary.csv`

---

## 6) How to Run
Environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
pip install -r requirements.txt
```
`requirements.txt`
```
torch
torchvision
numpy
matplotlib
tqdm
```

Train both (default; CNN then LSTM)
```bash
python train.py
```

Options
```bash
python train.py --model cnn
python train.py --model lstm
python train.py --epochs 10 --batch-size 256
python train.py --use-onehot-loss
```

Plots
```bash
python plot_results.py
```

---

## 7) Plots Produced
1) Validation accuracy — CNN vs LSTM  
2) Validation loss — CNN vs LSTM  
3) Seconds per epoch — CNN vs LSTM  
4) Bars: Average seconds/epoch, Total seconds, Test accuracy, Test loss  
(Axes auto-zoom for small differences.)

---

## 8) Results — Exact (from your CSVs)

### 8.1 combined_summary.csv
```
model,params,epochs,avg_sec_per_epoch,total_sec,test_acc,test_loss,device
cnn,421642,5,8.13,40.66,0.9890,0.0331,cuda
lstm,82186,5,7.54,37.73,0.9791,0.0644,cuda
```

### 8.2 cnn metrics.csv
```
epoch,train_loss,train_acc,val_loss,val_acc,seconds
1,0.1809,0.9464,0.0538,0.9855,9.23
2,0.0479,0.9852,0.0378,0.9877,7.36
3,0.0341,0.9896,0.0359,0.9897,8.54
4,0.0257,0.9918,0.0383,0.9880,7.08
5,0.0190,0.9940,0.0406,0.9878,8.42
```

### 8.3 lstm metrics.csv
```
epoch,train_loss,train_acc,val_loss,val_acc,seconds
1,0.5225,0.8320,0.1693,0.9503,8.23
2,0.1198,0.9659,0.1038,0.9703,6.92
3,0.0818,0.9763,0.0734,0.9793,7.77
4,0.0626,0.9810,0.0711,0.9800,6.89
5,0.0520,0.9844,0.0606,0.9817,7.90
```

**Comparison table**
| Model | Test Acc | Test Loss | Avg s/Epoch | Total s | Params | Device |
|:-----:|---------:|----------:|------------:|--------:|-------:|:------:|
| CNN   | 0.9890   | 0.0331    | 8.13        | 40.66   | 421,642 | cuda   |
| LSTM  | 0.9791   | 0.0644    | 7.54        | 37.73   | 82,186  | cuda   |

---

## 9) Discussion
- **Accuracy:** CNN > LSTM on MNIST; CNN exploits spatial locality and translation invariance.
- **Loss:** CNN converges faster and to lower validation loss.
- **Time:** LSTM slightly faster per epoch in this run; totals are close.
- **Device:** both trained on GPU (`cuda`).



---

## 10) Exercise Compliance
| Requirement | Implemented |
|:----------------------------------------------|:--:|
| Two distinct architectures (CNN & LSTM) .| OK
| Scaling + MNIST normalization .| OK
| Reshape per model .| OK
| One-hot label conversion .| OK
| Train and evaluate on test set .| OK
| Compare accuracy, loss, and time .| OK
| Plots and tables provided .| OK
| Discussion | OK
| Clear README on GitHub .| OK

---

## 11) Project Structure
```
.
├── train.py
├── plot_results.py
├── utils.py
├── models/
│   ├── cnn.py
│   └── lstm.py
├── runs/
│   ├── cnn/
│   ├── lstm/
│   └── combined_summary.csv
├── mnist_train.csv
├── mnist_test.csv
└── requirements.txt
```
