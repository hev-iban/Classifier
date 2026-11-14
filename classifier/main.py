#check
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def data_load(seed=42):
    data = load_breast_cancer()
    X = data.data
    y = data.target

    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=seed)

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    scalar = StandardScaler()
    X_train = scalar.fit_transform(X_train)
    X_test = scalar.transform(X_test)

    x_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    x_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
    return x_train_tensor, x_test_tensor, y_train, y_test

def initialize_parameters(num_features, hidden_units=8, seed=0):
    torch.manual_seed(seed)
    W1 = torch.randn(num_features, hidden_units) * 0.01
    b1 = torch.zeros(hidden_units)
    W2 = torch.randn(hidden_units, 1) * 0.01
    b2 = torch.zeros(1)
    return W1, b1, W2, b2

def forward(X, W1, b1, W2, b2):
    z1 = X @ W1 + b1
    a1 = torch.relu(z1)
    y_hat = torch.sigmoid(a1 @ W2 + b2)
    return z1, a1, y_hat

def bce(y_hat, y, eps=1e-7):
    y_hat = torch.clamp(y_hat, eps, 1 - eps)
    log_y_hat = torch.log(y_hat)
    log_one_minus_y_hat = torch.log(1 - y_hat)
    term1 = y * log_y_hat
    term2 = (1 - y) * log_one_minus_y_hat
    bce_per_sample = -(term1 + term2)
    loss = torch.mean(bce_per_sample)
    return loss

def backward(X, y, W2, z1, a1, y_hat):
    dz2 = (y_hat - y) / y.shape[0]
    da1 = dz2 @ W2.T
    dz1 = da1 * (z1 > 0).float()
    dW2 = a1.T @ dz2
    db2 = torch.sum(dz2, dim=0)
    dW1 = X.T @ dz1
    db1 = torch.sum(dz1, dim=0)
    return dW1, db1, dW2, db2

def train(X_train, y_train, hidden_units=8, lr=0.1, epochs=100, seed=0, verbose=True):
    num_features = X_train.shape[1]
    W1, b1, W2, b2 = initialize_parameters(num_features, hidden_units, seed)

    best_loss = float("inf")
    best_params = None
    best_epoch = None
    losses = []
    for epoch in range(epochs):
        z1, a1, y_hat = forward(X_train, W1, b1, W2, b2)
        loss = bce(y_hat, y_train)
        losses.append(loss.item())

        if verbose:
            print("Epoch", epoch, "Loss", loss.item())
        if loss < best_loss:
            best_loss = loss.item()
            best_epoch = epoch+1
            best_params = {"W1": W1.clone(), "b1": b1.clone(), "W2": W2.clone(), "b2": b2.clone()}

        dW1, db1, dW2, db2 = backward(X_train, y_train, W2, z1, a1, y_hat)
        W1 = W1 - lr * dW1
        b1 = b1 - lr * db1
        W2 = W2 - lr * dW2
        b2 = b2 - lr * db2
        
    params = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    return params, losses, best_epoch, best_loss, best_params

def evaluation(X_test, y_test, params):
    _, _, test_y_hat = forward(X_test, **params)

    pred = (test_y_hat > 0.5).float()
    acc = (pred == y_test).float().mean()

    print("Test acc:", acc.item())
    return acc.item()

if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", type=str)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--metrics-out", type=str)
    args = parser.parse_args()

    if args.split:
        data = load_breast_cancer()
        indices = np.arange(len(data.target))
        train_idx, test_idx = train_test_split(indices, test_size=0.2, random_state=42)

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "train": train_idx.tolist(),
            "test": test_idx.tolist(),
        }, indent=2))

    elif args.train:
        X_train, _, y_train, _ = data_load()
        if args.smoke:
            X_train = X_train[:32]
            y_train = y_train[:32]
            epochs = 5
        else:
            epochs = 100

        _, _, _, _, best_params = train(
            X_train, y_train, hidden_units=8, lr=0.1, epochs=epochs, verbose=False
        )
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_params, out_path)

    elif args.evaluate:
        _, X_test, _, y_test = data_load()
        saved_params = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        acc = evaluation(X_test, y_test, saved_params)

        out_path = Path(args.metrics_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({"test_accuracy": acc}, indent=2))

    else:
        X_train, X_test, y_train, y_test = data_load()
        params, losses, best_epoch, best_loss, best_params = train(
            X_train, y_train, hidden_units=8, lr=0.1, epochs=100
        )
        test_acc = evaluation(X_test, y_test, best_params)

        print("Checkpoint:")
        print("Best loss:", best_loss)
        print("Best epoch", best_epoch)

        plt.plot(losses)
        plt.title("By Hand Calculation")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()