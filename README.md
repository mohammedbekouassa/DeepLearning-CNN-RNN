# MNIST – Comparative Study (CNN vs RNN/LSTM)

## Objective
Comparer **CNN** (2D convolutions) vs **RNN/LSTM** (séquence de lignes) pour la classification MNIST.

## Data
Fichiers fournis localement : `mnist_train.csv`, `mnist_test.csv` (colonne 0 = label, colonnes 1..784 = pixels 0–255).

## How to run
```bash
pip install -r requirements.txt
python train.py --model cnn --epochs 5 --out results.csv
python train.py --model rnn --epochs 5 --out results.csv
python plot_results.py

### Results
We trained both models for 5 epochs on the local CSV MNIST (train/test split).

- CNN (2D convolutions): Test Acc = 0.9927, Test Loss = 0.0207
- RNN/LSTM (rows as sequence): Test Acc = 0.9809, Test Loss = 0.0654
- Total times: CNN = 287.4 s, RNN = 128.0 s

See: `acc_vs_epoch.png`, `loss_vs_epoch.png`, `train_time_bar.png`.
