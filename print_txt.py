import pickle
import numpy as np

with open("ANN_weights.txt", "rb") as f:
    weights, biases = pickle.load(f)

# Print weights
for i in sorted(weights.keys()):
    print(f"weights[{i}] = np.array([")
    for row in weights[i]:
        print("    [" + ", ".join(f"{x:.8f}" for x in row) + "],")
    print("])\n")

# Print biases
for i in sorted(biases.keys()):
    print(f"biases[{i}] = np.array([")
    for row in biases[i]:
        print("    [" + ", ".join(f"{x:.8f}" for x in row) + "],")
    print("])\n")