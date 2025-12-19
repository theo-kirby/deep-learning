from mlp import MLP

nn = MLP(inputs=2, depth=8, layers=1, outputs=1, learning_rate=0.1)

# test pred before training
result = nn.predict([0, 1])
print(f"Prediction for [0, 1]: {result}")

# create xor data
xor_data = [
    ([0, 0], [0]),
    ([0, 1], [1]),
    ([1, 0], [1]),
    ([1, 1], [0]),
]

# train for a 10k iterations
for _ in range(10000):
    for inputs, target in xor_data:
        nn.train(inputs, target)

# pred after training
print("\nAfter training on XOR:")
for inputs, _ in xor_data:
    print(f"{inputs} -> {nn.predict(inputs)}")

