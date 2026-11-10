"""
test_engine.py: Unit tests verifying forward and backward passes,
operator overloads, and gradient correctness.
"""
from micrograd.engine import Value

def test_sanity_check():
    x = Value(-4.0)
    z = 2 * x + 2 + x
    q = z.relu() + z * x
    h = (z * z).relu()
    y = h + q + q * x
    y.backward()
    x_data, x_grad = x.data, x.grad

    # Analytical values:
    # z = 3x + 2 = -10
    # q = relu(-10) + (-10)*(-4) = 0 + 40 = 40
    # h = relu((-10)^2) = 100
    # y = 100 + 40 + 40*(-4) = -20
    assert y.data == -20.0
    # y = (3x+2)^2 + (3x+2)x + (3x+2)x^2
    # dy/dx = 2(3x+2)*3 + (6x+2) + (9x^2+4x) = 6(-10) + (-22) + (144 - 16) = -60 - 22 + 128 = 46
    assert x.grad == 46.0

def test_tanh_relu_pow():
    a = Value(0.5)
    b = Value(-1.5)
    c = a * b
    d = c.tanh()
    e = d + 2.0
    f = e ** 2
    f.backward()

    # Numerical gradient check via finite differences
    h = 1e-6
    # Perturb a
    a_plus = Value(0.5 + h)
    f_plus = ((a_plus * b).tanh() + 2.0) ** 2
    a_minus = Value(0.5 - h)
    f_minus = ((a_minus * b).tanh() + 2.0) ** 2
    num_grad_a = (f_plus.data - f_minus.data) / (2 * h)

    assert abs(a.grad - num_grad_a) < 1e-4

if __name__ == "__main__":
    test_sanity_check()
    test_tanh_relu_pow()
    print("All unit tests passed successfully!")
