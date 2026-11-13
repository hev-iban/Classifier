import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class BreastCancerClassifier(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(BreastCancerClassifier, self).__init__()
        self.z1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.z2 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.loss_fn = nn.BCELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.1)
    
    def forward(self, x):
        x = self.z1(x)
        x = self.relu(x)
        x = self.z2(x)
        x = self.sigmoid(x)
        return x
    
    def train(self, x_train, y_train, epochs):
        losses = []
        for epoch in range(epochs):
            y_pred = self.forward(x_train)
            loss = self.loss_fn(y_pred, y_train)
            losses.append(loss.item())
            print("Epoch:", epoch, "Loss:", loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        return losses

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)

x_train_tensor = torch.tensor(X_train, dtype=torch.float32)
x_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

input_size = x_train_tensor.shape[1]
hidden_size = 8

model = BreastCancerClassifier(input_size, hidden_size)
losses = model.train(x_train_tensor, y_train, epochs=100)

eval_y_pred = model(x_test_tensor)
eval_pred = (eval_y_pred > 0.5).float()
eval_acc = (eval_pred == y_test).float().mean()
print("Test acc:", eval_acc.item())

plt.plot(losses)
plt.title("Breast cancer train")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()