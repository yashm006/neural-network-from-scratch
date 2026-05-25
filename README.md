# Neural Network From Scratch

A fully connected Feed-Forward Network implemented completely from scratch using only `NumPy` and core mathematics (no `PyTorch` or `TensorFlow`).

This project was built to understand the mathematical foundations of deep learning by bypassing high-level wrappers and manually implementing vectorization, backpropagation, and weight optimization.

---

### Function Approximation & The Universal Approximation Theorem

This project explores the core thesis of the **Universal Approximation Theorem**, which states that:

> A sufficiently large feed-forward neural network with non-linear activation functions can approximate any continuous function on a bounded domain arbitrarily well.
> $$\|\hat{f}(x) - f(x)\| \le \epsilon$$

The model successfully approximates a variety of continuous mathematical mappings, including:
- Quadratic and cubic polynomial functions ($f(x) = x^2$, $f(x) = x^3$)
- Trigonometric functions ($\sin(x)$, $\cos(x)$, $\dots$)
- Square Wave function

---

### Key Architectural Features
- **Pure NumPy Implementation:** Leverages NumPy's optimized matrix operations (`@` and `.T`) for layer-by-layer forward and backward propagation passes.
- **Dynamic Optimization:** Implements He (Kaiming) Initialization to ensure variance scaling stability across deep layers.
- **Defensive Engineering:** Automated scaling and normalization blocks to prevent mathematical gradient explosions ($NaN$ states).
- **Manual Backpropagation:** Chain rule implementation utilizing negative layer tracking indices for modular depth.
- **Live Training Visualizations:** Built-in dynamic tracking using `Matplotlib` to display optimization convergence in real time.

---

### Mathematical Foundations

#### 1. Forward Propagation
For any hidden layer $l$:
$$z^{[l]} = W^{[l]} a^{[l-1]} + b^{[l]}$$
$$a^{[l]} = g(z^{[l]})$$

Where:
- $W^{[l]}$ = Weight matrix
- $b^{[l]}$ = Bias vector 
- $a^{[l-1]}$ = Activation array from the prior layer ($a^{[0]} = X$)
- $z^{[l]}$ = Pre-activation linear combination
- $g$ = Activation Function (LeakyReLU implemented to bypass the dying ReLU limitation)

#### 2. Loss Function
The network evaluates precision using Mean Squared Error (MSE) Loss across batch samples:
$$\mathcal{L} = \frac{1}{m} \sum_{i=1}^{m}(\hat{y}^{(i)} - y^{(i)})^2$$

#### 3. Backpropagation (The Chain Rule)
Gradients are manually tracked backwards through the network graph:
$$\nabla_{z^{[l]}} \mathcal{L} = \nabla_{a^{[l]}} \mathcal{L} \odot g'(z^{[l]})$$
$$\nabla_{W^{[l]}} \mathcal{L} = \frac{1}{m} \left(\nabla_{z^{[l]}} \mathcal{L} \cdot (a^{[l-1]})^T\right)$$
$$\nabla_{b^{[l]}} \mathcal{L} = \frac{1}{m} \sum \nabla_{z^{[l]}} \mathcal{L}$$
$$\nabla_{a^{[l-1]}} \mathcal{L} = (W^{[l]})^T \cdot \nabla_{z^{[l]}} \mathcal{L}$$

#### 4. Gradient Descent Optimization
Parameters are adjusted per epoch based on computed partial derivatives:
$$W^{[l]} := W^{[l]} - \eta \nabla_{W^{[l]}} \mathcal{L}$$
$$b^{[l]} := b^{[l]} - \eta \nabla_{b^{[l]}} \mathcal{L}$$

Where $\eta$ represents the learning rate.

---

### Visual Results

[Sin Wave Demo](Recordings/Sin%20Wave%20function.mp4)


[Quadratic function Demo](Recordings/Quadratic%20function.mp4)

[All demos...](Recordings)