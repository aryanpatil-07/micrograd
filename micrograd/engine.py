import math

class Value:
    """
    Stores a single scalar value and its gradient, dynamically constructing
    a directed acyclic graph (DAG) for reverse-mode automatic differentiation.
    """
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = float(data)
        self.grad = 0.0
        # Internal variables used for autograd DAG graph construction
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    # --- Addition (+) ---
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward

        return out

    def __radd__(self, other):
        return self + other

    # --- Multiplication (*) ---
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out

    def __rmul__(self, other):
        return self * other

    # --- Power (**) ---
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data ** other, (self,), f'**{other}')

        def _backward():
            self.grad += (other * (self.data ** (other - 1))) * out.grad
        out._backward = _backward

        return out

    # --- Negation (-) ---
    def __neg__(self):
        return self * -1.0

    # --- Subtraction (-) ---
    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return Value(other) + (-self)

    # --- Division (/) ---
    def __truediv__(self, other):
        return self * (other ** -1.0)

    def __rtruediv__(self, other):
        return Value(other) * (self ** -1.0)

    # --- Activation Functions ---
    def relu(self):
        out = Value(0.0 if self.data < 0.0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (1.0 if out.data > 0.0 else 0.0) * out.grad
        out._backward = _backward

        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1.0 - t ** 2) * out.grad
        out._backward = _backward

        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward

        return out

    # --- Reverse-Mode Autograd (Backpropagation) ---
    def backward(self):
        """
        Traverses the computational graph in topological order to compute
        gradients for all nodes via the multivariable chain rule.
        """
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # Base case: d(root)/d(root) = 1.0
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
