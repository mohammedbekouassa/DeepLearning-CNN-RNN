import time, torch, random, numpy as np
from torch.utils.data import DataLoader, random_split, TensorDataset

MNIST_MEAN = 0.1307
MNIST_STD  = 0.3081

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def _load_csv(path, has_header=True):
    # row: label,p0,...,p783  with pixels 0..255
    skip = 1 if has_header else 0
    arr = np.loadtxt(path, delimiter=",", dtype=np.float32, skiprows=skip)
    if arr.ndim == 1:
        arr = np.expand_dims(arr, 0)
    y = arr[:, 0].astype(np.int64)
    X = arr[:, 1:] / 255.0
    if X.shape[1] != 784:
        raise ValueError(f"{path}: expected 784 pixels, got {X.shape[1]}")
    X = X.reshape((-1, 1, 28, 28))
    X = (X - MNIST_MEAN) / MNIST_STD  # same normalization as torchvision MNIST
    return TensorDataset(torch.from_numpy(X), torch.from_numpy(y))

def get_loaders(batch_size=128, val_split=0.1, num_workers=2,
                train_csv=None, test_csv=None, has_header=True):
    if train_csv and test_csv:
        train_full = _load_csv(train_csv, has_header)
        test_ds    = _load_csv(test_csv,  has_header)
    else:
        from torchvision import datasets, transforms
        tfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,))
        ])
        train_full = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
        test_ds    = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)

    val_len = int(len(train_full)*val_split)
    train_len = len(train_full) - val_len
    train_ds, val_ds = random_split(train_full, [train_len, val_len])

    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_ld   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_ld  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_ld, val_ld, test_ld

@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for x,y in loader:
        x,y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += loss.item()*y.size(0)
        pred = logits.argmax(1)
        correct += (pred==y).sum().item()
        total += y.size(0)
    return loss_sum/total, correct/total

def now():
    return time.perf_counter()
