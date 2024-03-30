import numpy as np
import matplotlib.pyplot as plt 

# Sigmoid function
def sigmoid_func(x):
    val = 1 / (1 + np.exp(-x))
    return val 

# ReLU Function
def relu_func(x):
    val = np.maximum(0, x)
    return val 

# Leaky-ReLU Function
def leaky_relu(x):
    val = np.maximum(0.01 * x, x)
    return val 

# Tanh Function
def tanh_func(x):
    val = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    return val 

# Plot Activation Function
def plot(x, y, label):
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label=label)
    plt.xlabel("Input")
    plt.ylabel(f"{label} Activation Function")
    plt.legend()
    plt.show()

input_values = [-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6]
input_np = np.array(input_values)

if __name__ == "__main__":
    # Plot sigmoid function
    sigmoid_values = np.array([sigmoid_func(x) for x in input_values])
    plot(input_np, sigmoid_values, label="Sigmoid")

    # Plot relu function
    relu_values = np.array([relu_func(x) for x in input_values])
    plot(input_np, relu_values, label="ReLU")

    # Plot leaky-relu function
    leaky_relu_values = np.array([leaky_relu(x) for x in input_values])
    plot(input_np, leaky_relu_values, label="Leaky ReLU")

    # Plot tanh function
    tanh_values = np.array([tanh_func(x) for x in input_values])
    plot(input_np, tanh_values, label="Tanh")

    plt.close()
    
    print(f"ReLU Values: {[relu_func(x) for x in input_values]}")
    print(f"Leaky ReLU Values: {[leaky_relu(x) for x in input_values]}")
    print(f"Tanh Values: {[tanh_func(x) for x in input_values]}")
    # Bug of wrong function name