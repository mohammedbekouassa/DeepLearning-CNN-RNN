# MNIST – Comparison of CNN and LSTM Models

## Objective
Perform a comparative study between two neural network architectures (CNN and LSTM) for handwritten digit classification using the **MNIST** dataset.

The project fulfills all exercise requirements:
- Data preprocessing (normalization, reshaping)
- One-hot encoding of labels (saved as proof)
- Model training and testing
- Comparison of accuracy, loss, and training time
- Result visualization through clear plots

---

## Dataset
- **Files:** `mnist_train.csv`, `mnist_test.csv`  
  Format: `label,pixel0,…,pixel783`
- **Normalization:**  
  \[
  X = (X/255 - 0.1307) / 0.3081
  \]
- **Labels:** converted to one-hot vectors  
  (stored in `runs/<model>/onehot_preview.csv`)

---

## Environment Setup
```bash
python -m venv venv
venv\Scripts\activate         # Windows
source venv/bin/activate      # Linux / macOS
pip install -r requirements.txt
