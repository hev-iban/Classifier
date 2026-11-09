import pytest
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


@pytest.fixture
def data_pipeline():
    torch.manual_seed(0)
    data = load_breast_cancer()

    x_data = torch.tensor(data.data, dtype=torch.float32)
    y_data = torch.tensor(data.target, dtype=torch.float32).unsqueeze(1)

    X_train, X_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.2, random_state=42
    )

    scalar = StandardScaler()
    X_train_scaled = scalar.fit_transform(X_train)
    X_test_scaled = scalar.transform(X_test)

    x_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    x_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)

    W1 = torch.randn(30, 8) * 0.01
    b1 = torch.zeros(8)
    W2 = torch.randn(8, 1) * 0.01
    b2 = torch.zeros(1)

    return {
        "x_data": x_data,
        "y_data": y_data,
        "x_train_tensor": x_train_tensor,
        "x_test_tensor": x_test_tensor,
        "y_train": y_train,
        "y_test": y_test,
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2,
    }


def test_raw_dataset_shapes(data_pipeline):
    assert data_pipeline["x_data"].shape == (569, 30)
    assert data_pipeline["y_data"].shape == (569, 1)


def test_train_test_split_shapes(data_pipeline):
    assert data_pipeline["x_train_tensor"].shape == (455, 30)
    assert data_pipeline["x_test_tensor"].shape == (114, 30)
    assert data_pipeline["y_train"].shape == (455, 1)
    assert data_pipeline["y_test"].shape == (114, 1)


def test_weight_and_bias_shapes(data_pipeline):
    assert data_pipeline["W1"].shape == (30, 8)
    assert data_pipeline["b1"].shape == (8,)
    assert data_pipeline["W2"].shape == (8, 1)
    assert data_pipeline["b2"].shape == (1,)