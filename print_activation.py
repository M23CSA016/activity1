from activations import sigmoid_func
import numpy as np 

input_values = [-3.5, -1.2, 0, 2.8, -4.1, 1.5, -0.7, 3.2, -2.4, 4.6]
input_np = np.array(input_values)

print(f"ReLU Values: {[sigmoid_func(x) for x in input_values]}")