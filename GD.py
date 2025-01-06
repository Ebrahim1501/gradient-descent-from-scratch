import numpy as np

data_x = np.random.randn(10, 1)#random 10 samples dataset
data_y = 5 * data_x + np.random.rand(10, 1)

weight = 0.0
bias = 0.0
alpha = 0.01
tolerance = 1e-6

def model_prediction(data_x, weight, bias):
    return weight * data_x + bias

def calculate_loss(data_y, predictions, error_type="mse"):
    if error_type == "mse":#(y-y')^2
        return np.mean((data_y - predictions) ** 2)
    elif error_type == "mae":
        return np.mean(np.abs(data_y - predictions))
    elif error_type == "rmse":
        return np.sqrt(np.mean((data_y - predictions) ** 2))
    else:
        print("Invalid loss func entered null returned ")
        return None

def compute_gradients(data_x, data_y, weight, bias):
    gradient_w = 0.0
    gradient_b = 0.0
    num_samples = data_x.shape[0]
    for sample_x, sample_y in zip(data_x, data_y):
        pred_y = model_prediction(sample_x, weight, bias)
        gradient_w += -2 * sample_x * (sample_y - pred_y)
        gradient_b += -2 * (sample_y - pred_y)
    gradient_w /= num_samples
    gradient_b /= num_samples
    return gradient_w, gradient_b

def linear_regression_train(data_x, data_y, weight, bias, alpha, iterations, tolerance, error_type="mse",print_every=10):
    prev_loss = float('inf')#intial loss = +infinity
    for step in range(iterations):
        gradient_w, gradient_b = compute_gradients(data_x, data_y, weight, bias)
        weight -= alpha * gradient_w
        bias -= alpha * gradient_b
        predictions = model_prediction(data_x, weight, bias)
        loss_value = calculate_loss(data_y, predictions, error_type)

        if abs(prev_loss - loss_value) < tolerance:
            print(f"\nlocal minima reached at step {step}, Loss: {loss_value:.6f}, weight: {weight.item():.4f}, bias: {bias.item():.4f}")
            break

        prev_loss = loss_value

        if step % print_every == 0:
            print(f"Step {step}, Loss: {loss_value:.4f}, weight: {weight.item():.4f}, bias: {bias.item():.4f}")
    return weight, bias

iterations = 100
final_weight, final_bias = linear_regression_train(data_x, data_y, weight, bias, alpha, iterations, tolerance, error_type="mae",print_every=10)

print(f"\nFinal weight: {final_weight.item():.4f}, Final bias: {final_bias.item():.4f}")
