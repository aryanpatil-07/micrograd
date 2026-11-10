"""
demo.py: Demonstrates end-to-end forward pass, loss calculation,
automatic differentiation (backward pass), and gradient descent optimization.
"""
from micrograd.engine import Value
from micrograd.nn import MLP

def main():
    print("=" * 60)
    print(" 1. SCALAR AUTOGRAD COMPUTATION GRAPH & BACKPROPAGATION")
    print("=" * 60)

    # Build an expression: L = ((a * b + c).tanh() * d + e) ** 2
    a = Value(2.0, label='a')
    b = Value(-3.0, label='b')
    c = Value(10.0, label='c')
    e = a * b; e.label = 'e'
    d = e + c; d.label = 'd'
    f = Value(-2.0, label='f')
    L = d * f; L.label = 'L'

    print(f"Forward pass output (L): {L.data:.4f}")
    
    # Run backpropagation
    L.backward()

    print(f"dL/da: {a.grad:.4f} (analytical gradient)")
    print(f"dL/db: {b.grad:.4f} (analytical gradient)")
    print(f"dL/dc: {c.grad:.4f} (analytical gradient)")
    print(f"dL/df: {f.grad:.4f} (analytical gradient)")

    print("\n" + "=" * 60)
    print(" 2. TRAINING A MULTI-LAYER PERCEPTRON (MLP)")
    print("=" * 60)

    # Initialize a 2-layer MLP (3 inputs -> 4 hidden -> 4 hidden -> 1 output)
    model = MLP(3, [4, 4, 1])
    print(f"Model Architecture:\n{model}")
    print(f"Total trainable parameters: {len(model.parameters())}")

    # Toy dataset: 4 samples with 3 features each
    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0]
    ]
    ys = [1.0, -1.0, -1.0, 1.0]  # Desired targets

    print("\nInitial predictions before training:")
    for x in xs:
        print(f"  Input: {x} -> Predicted: {model(x).data:.4f}")

    # Training loop
    epochs = 50
    lr = 0.05
    print(f"\nTraining for {epochs} epochs (Learning Rate = {lr})...")

    for k in range(epochs):
        # 1. Forward pass: compute predictions and Mean Squared Error loss
        ypred = [model(x) for x in xs]
        loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred))

        # 2. Zero gradients
        model.zero_grad()

        # 3. Backward pass: compute dLoss/dParameter
        loss.backward()

        # 4. Parameter update (Gradient Descent)
        for p in model.parameters():
            p.data -= lr * p.grad

        if (k + 1) % 10 == 0 or k == 0:
            print(f"  Epoch [{k+1:2d}/{epochs}] | MSE Loss: {loss.data:.6f}")

    print("\nFinal predictions after training:")
    for x, y in zip(xs, ys):
        pred = model(x).data
        print(f"  Input: {x} -> Target: {y:4.1f} | Predicted: {pred:6.4f}")

    print("\nmicrograd self-test completed successfully!")

if __name__ == "__main__":
    main()
