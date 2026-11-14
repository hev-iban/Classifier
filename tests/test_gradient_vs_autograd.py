import torch
from classifier.main import forward, bce, initialize_parameters, backward, data_load

def test_gradients_vs_autograd():
    X_train, _, y_train, _ = data_load()
    X, y = X_train[:16], y_train[:16]
    
    W1, b1, W2, b2 = initialize_parameters(
        num_features=X.shape[1], hidden_units=8)

    W1_auto = W1.clone().detach().requires_grad_(True)
    b1_auto = b1.clone().detach().requires_grad_(True)
    W2_auto = W2.clone().detach().requires_grad_(True)
    b2_auto = b2.clone().detach().requires_grad_(True)
    
    z1, a1, y_hat = forward(X, W1, b1, W2, b2)
    dW1, db1, dW2, db2 = backward(X, y, W2, z1, a1, y_hat)

    z1_auto, a1_auto, y_hat_auto = forward(X, W1_auto, b1_auto, W2_auto, b2_auto)
    loss_auto = bce(y_hat_auto, y)
    loss_auto.backward()

    assert torch.allclose(dW1, W1_auto.grad, rtol=1e-5, atol=1e-8), f"dW1 mismatch! Max diff: {torch.max(torch.abs(dW1 - W1_auto.grad)).item()}"
    assert torch.allclose(db1, b1_auto.grad, rtol=1e-5, atol=1e-8), f"db1 mismatch! Max diff: {torch.max(torch.abs(db1 - b1_auto.grad)).item()}"
    assert torch.allclose(dW2, W2_auto.grad, rtol=1e-5, atol=1e-8), f"dW2 mismatch! Max diff: {torch.max(torch.abs(dW2 - W2_auto.grad)).item()}"
    assert torch.allclose(db2, b2_auto.grad, rtol=1e-5, atol=1e-8), f"db2 mismatch! Max diff: {torch.max(torch.abs(db2 - b2_auto.grad)).item()}"

    print("dW1 close:", torch.allclose(dW1, W1_auto.grad, rtol=1e-5, atol=1e-8))
    print("db1 close:", torch.allclose(db1, b1_auto.grad, rtol=1e-5, atol=1e-8))
    print("dW2 close:", torch.allclose(dW2, W2_auto.grad, rtol=1e-5, atol=1e-8))
    print("db2 close:", torch.allclose(db2, b2_auto.grad, rtol=1e-5, atol=1e-8))