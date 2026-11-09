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

print("--- W1 ---")
print(W1)

print("\n--- b1 ---")
print(b1)

print("\n--- W2 ---")
print(W2)

print("\n--- b2 ---")
print(b2)