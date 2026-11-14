import torch
from classifier.auto_main import BreastCancerClassifier
from classifier.main import backward, bce, data_load, forward, initialize_parameters

def test_starting_value():

    X_train, _, _, _ = data_load()
    num_features = X_train.shape[1]                                 
    W1, b1, W2, b2 = initialize_parameters(num_features)
    model = BreastCancerClassifier(num_features)

    assert torch.equal(model.z1.weight.T, W1), "W1 does not agree!"
    assert torch.equal(model.z1.bias, b1), "b1 does not agree!"
    assert torch.equal(model.z2.weight.T, W2), "W2 does not agree!"
    assert torch.equal(model.z2.bias, b2), "b2 does not agree!"

    print("Starting values agree!")
    
def test_5_epoch_run():
    lr = 0.1
    X_train, _, y_train, _ = data_load()
    num_features = X_train.shape[1]

    W1, b1, W2, b2 = initialize_parameters(num_features)
    model = BreastCancerClassifier(num_features)

    for epoch in range(5):
        z1, a1, y_hat_man = forward(X_train, W1, b1, W2, b2)
        loss_man = bce(y_hat_man, y_train)

        y_hat_mod = model(X_train)
        loss_mod = model.loss_fn(y_hat_mod, y_train)

        torch.testing.assert_close(y_hat_mod, y_hat_man)
        torch.testing.assert_close(loss_mod, loss_man)

        dW1, db1, dW2, db2 = backward(X_train, y_train, W2, z1, a1, y_hat_man)

        model.optimizer.zero_grad()
        loss_mod.backward()

        torch.testing.assert_close(model.z1.weight.grad.T, dW1)
        torch.testing.assert_close(model.z1.bias.grad, db1)
        torch.testing.assert_close(model.z2.weight.grad.T, dW2)
        torch.testing.assert_close(model.z2.bias.grad, db2)

        model.optimizer.step()
        with torch.no_grad():
            W1 -= lr * dW1
            b1 -= lr * db1
            W2 -= lr * dW2
            b2 -= lr * db2

        torch.testing.assert_close(model.z1.weight.T, W1)
        torch.testing.assert_close(model.z1.bias, b1)
        torch.testing.assert_close(model.z2.weight.T, W2)
        torch.testing.assert_close(model.z2.bias, b2)
if __name__ == "__main__":
    test_starting_value()
    test_5_epoch_run()