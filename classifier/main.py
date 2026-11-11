import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

torch.manual_seed(0)

# to load the dataset
data = load_breast_cancer()


df = pd.DataFrame(data=data.data, columns=data.feature_names)
df['target'] = data.target


x_data = torch.tensor(data.data, dtype=torch.float32)
y_data = torch.tensor(data.target, dtype=torch.float32).unsqueeze(1)


X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)

x_train_tensor = torch.tensor(X_train, dtype=torch.float32)
x_test_tensor = torch.tensor(X_test, dtype=torch.float32)


W1 = torch.randn(30, 8) * 0.01
b1 = torch.zeros(8)
W2 = torch.randn(8, 1) * 0.01
b2 = torch.zeros(1)

lr = 0.1
losses = []
for epoch in range(100):
    z1 = x_train_tensor @ W1 + b1
    a1 = torch.relu(z1)
    z2 = a1 @ W2 + b2
    y_hat = torch.sigmoid(z2)
  
    log_y_hat = torch.log(y_hat)
    log_one_minus_y_hat = torch.log(1 - y_hat)
  
    term1 = y_train * log_y_hat
    term2 = (1 - y_train) * log_one_minus_y_hat
    bce_per_sample = -(term1 + term2)
  
    loss = torch.mean(bce_per_sample)
    losses.append(loss.item())
    print("Epoch", epoch, "Loss", loss.item())

    dz2 = (y_hat - y_train)/y_train.shape[0]
    da1 = dz2 @ W2.T
    dz1 = da1 * (z1 > 0).float()

    dW2 = a1.T @ dz2
    db2 = torch.sum(dz2, dim=0)
    dW1 = x_train_tensor.T @ dz1
    db1 = torch.sum(dz1, dim=0)

    W1 = W1 - lr * dW1
    b1 = b1 - lr * db1
    W2 = W2 - lr * dW2
    b2 = b2 - lr * db2

test_z1 = x_test_tensor @ W1 + b1
test_a1 = torch.relu(test_z1)
test_z2 = test_a1 @ W2 + b2
test_y_hat = torch.sigmoid(test_z2)

pred = (test_y_hat > 0.5).float()
acc = (pred == y_test).float().mean()
print("Test acc:", acc.item())
print(pred.shape)
print(y_test.shape)

plt.plot(losses)
plt.title("Breast cancer train")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()