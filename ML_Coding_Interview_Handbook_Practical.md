
# ML Coding Interview Handbook

This is a one-day practical preparation guide based on the ML Coding and Problem Solving interview guide. The PDF emphasizes clean, efficient, tensorized code, broadcasting/vectorization, testing, numerical stability, and the ability to communicate assumptions and trade-offs. It also explicitly lists core ML algorithms such as linear/logistic regression, KNN, decision trees, K-Means, PCA, softmax/cross-entropy, convolution, pooling, sampling, and optimizers.

## How to use this in the interview

For every problem, say this out loud before coding:

```text
Let me clarify the input/output shapes, constraints, and whether we want NumPy or PyTorch.
I will first write a clean vectorized implementation, then discuss edge cases and complexity.
```

For each solution, be ready to explain:

```text
1. Shapes
2. Mathematical formula
3. Numerical stability
4. Time complexity
5. Memory complexity
6. A tiny test case
```

---

# Problem 1 - Linear Regression with Gradient Descent

## Interview question

Implement linear regression from scratch using NumPy. Given `X: [N, D]` and `y: [N]`, learn `w: [D]` and scalar `b` by minimizing mean squared error with gradient descent.

## What is being asked

You need to implement the prediction equation, MSE loss, gradients with respect to weights and bias, and the training loop. The interviewer wants to see that you understand basic supervised learning and vectorized gradient computation.

## Clarifying questions

- Should I include an explicit bias term or absorb it into `X`?
- Should I standardize features?
- Should the solution use closed-form least squares or gradient descent?
- What stopping criterion should I use?

## Code solution

```python
import numpy as np


def mse_loss(y_pred, y):
    return np.mean((y_pred - y) ** 2)


def fit_linear_regression(X, y, lr=1e-2, num_steps=1000):
    """
    X: [N, D]
    y: [N]
    Returns:
        w: [D]
        b: scalar
        losses: list of floats
    """
    N, D = X.shape
    w = np.zeros(D)
    b = 0.0
    losses = []

    for _ in range(num_steps):
        # Forward pass: [N]
        y_pred = X @ w + b
        error = y_pred - y
        loss = np.mean(error ** 2)
        losses.append(loss)

        # Gradients of MSE = mean((Xw+b-y)^2)
        dw = (2.0 / N) * (X.T @ error)  # [D]
        db = (2.0 / N) * np.sum(error)  # scalar

        # Gradient descent update
        w -= lr * dw
        b -= lr * db

    return w, b, losses


# Tiny test
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 2))
    true_w = np.array([2.0, -3.0])
    true_b = 0.5
    y = X @ true_w + true_b

    w, b, losses = fit_linear_regression(X, y, lr=0.05, num_steps=500)
    print(w, b, losses[-1])
```

## Explanation

The model is:

```text
y_pred = Xw + b
```

The loss is:

```text
L = mean((y_pred - y)^2)
```

Let `error = y_pred - y`. Then:

```text
dL/dw = 2/N * X^T error
dL/db = 2/N * sum(error)
```

The important interview point is that all examples are processed at once using matrix multiplication. There is no loop over samples.

## Complexity

- Time per step: `O(ND)`
- Memory: `O(D + N)` if storing predictions/errors

## Common mistakes

- Forgetting the factor `2/N`
- Using a learning rate that is too large
- Not scaling features, causing unstable gradient descent
- Accidentally making `y` shape `[N,1]` while predictions are `[N]`

## Follow-ups

- How would you solve this in closed form? `w = (X^T X)^(-1) X^T y`
- What if `X^T X` is singular? Use ridge regression or pseudo-inverse.
- How would you add L2 regularization?

---

# Problem 2 - Logistic Regression with Stable BCE

## Interview question

Implement binary logistic regression from scratch. Given `X: [N, D]` and labels `y: [N]` in `{0,1}`, train a classifier using gradient descent.

## What is being asked

You need to implement logits, sigmoid probabilities, binary cross-entropy, and the gradient update. The important part is numerical stability.

## Clarifying questions

- Are labels binary or multi-class?
- Should I return probabilities or class labels?
- Should I use regularization?
- Is this NumPy or PyTorch?

## Code solution

```python
import numpy as np


def sigmoid(z):
    # Clipping avoids overflow in exp for very large negative values.
    z = np.clip(z, -50, 50)
    return 1.0 / (1.0 + np.exp(-z))


def binary_cross_entropy_from_logits(logits, y):
    """
    Stable BCE with logits.
    Equivalent to:
        -y*log(sigmoid(logits)) - (1-y)*log(1-sigmoid(logits))
    but avoids log(0) and overflow.
    """
    return np.mean(np.maximum(logits, 0) - logits * y + np.log1p(np.exp(-np.abs(logits))))


def fit_logistic_regression(X, y, lr=1e-2, num_steps=1000):
    N, D = X.shape
    w = np.zeros(D)
    b = 0.0
    losses = []

    for _ in range(num_steps):
        logits = X @ w + b      # [N]
        probs = sigmoid(logits) # [N]

        loss = binary_cross_entropy_from_logits(logits, y)
        losses.append(loss)

        # For BCE with sigmoid, gradient wrt logits is probs - y.
        dlogits = probs - y     # [N]
        dw = X.T @ dlogits / N  # [D]
        db = np.mean(dlogits)   # scalar

        w -= lr * dw
        b -= lr * db

    return w, b, losses


def predict_proba(X, w, b):
    return sigmoid(X @ w + b)


def predict(X, w, b, threshold=0.5):
    return (predict_proba(X, w, b) >= threshold).astype(int)
```

## Explanation

Logistic regression predicts:

```text
logits = Xw + b
p = sigmoid(logits)
```

The BCE loss can be unstable if you compute `log(sigmoid(x))` directly. For large positive or negative logits, sigmoid can become exactly 0 or 1 in floating point. The stable form avoids this.

The nice simplification is:

```text
dL/dlogits = sigmoid(logits) - y
```

So the weight gradient is just a matrix multiplication:

```text
dw = X^T (p - y) / N
```

## Complexity

- Time per step: `O(ND)`
- Memory: `O(N)` for logits/probs

## Common mistakes

- Computing `np.log(sigmoid(logits))` directly
- Forgetting to average over `N`
- Confusing labels in `{0,1}` with labels in `{-1,1}`
- Returning logits when probabilities are expected

## Follow-ups

- How would you extend this to multi-class classification? Use softmax regression.
- How do you handle class imbalance? Weighted BCE or sampling.
- How do you add L2 regularization? Add `lambda * w` to `dw`.

---

# Problem 3 - Stable Softmax and Cross-Entropy

## Interview question

Implement a numerically stable softmax and cross-entropy loss for multi-class classification. Inputs are `logits: [N, C]` and integer labels `y: [N]`.

## What is being asked

You need to compute probabilities and loss without numerical overflow. You should also know the gradient of softmax cross-entropy.

## Clarifying questions

- Are labels integer class IDs or one-hot vectors?
- Should I return per-example losses or the mean loss?
- Should I implement the gradient too?

## Code solution

```python
import numpy as np


def softmax(logits):
    """
    logits: [N, C]
    returns probabilities: [N, C]
    """
    z = logits - np.max(logits, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def cross_entropy_loss(logits, y):
    """
    logits: [N, C]
    y: [N] integer class labels
    """
    z = logits - np.max(logits, axis=1, keepdims=True)
    log_probs = z - np.log(np.sum(np.exp(z), axis=1, keepdims=True))
    return -np.mean(log_probs[np.arange(logits.shape[0]), y])


def softmax_cross_entropy_grad(logits, y):
    """
    Returns dL/dlogits for mean cross-entropy.
    """
    N = logits.shape[0]
    probs = softmax(logits)
    probs[np.arange(N), y] -= 1.0
    return probs / N


# Tiny test
logits = np.array([[10.0, 0.0, -10.0], [1.0, 2.0, 3.0]])
y = np.array([0, 2])
print(softmax(logits))
print(cross_entropy_loss(logits, y))
print(softmax_cross_entropy_grad(logits, y))
```

## Explanation

Softmax is:

```text
softmax_i = exp(logit_i) / sum_j exp(logit_j)
```

The problem is that `exp(1000)` overflows. But softmax is invariant to adding or subtracting the same constant from every class logit. So we subtract the row-wise maximum:

```text
softmax(logits) = softmax(logits - max(logits))
```

The cross-entropy for integer labels is:

```text
L = -mean(log p_correct)
```

The gradient is one of the most useful ML interview facts:

```text
dL/dlogits = (softmax(logits) - one_hot(y)) / N
```

## Complexity

- Time: `O(NC)`
- Memory: `O(NC)`

## Common mistakes

- Applying softmax over the wrong axis
- Forgetting `keepdims=True`
- Taking `log(softmax())` instead of using log-softmax
- Modifying `probs` in-place when the original probabilities are still needed

## Follow-ups

- How would you support one-hot labels?
- How do you implement label smoothing?
- Why is cross-entropy better than MSE for classification?

---

# Problem 4 - Vectorized KNN

## Interview question

Implement a K-Nearest Neighbor classifier. Given `X_train: [N, D]`, `y_train: [N]`, and `X_test: [M, D]`, predict one label for each test sample. Avoid nested Python loops.

## What is being asked

This is mainly a tensorization and broadcasting problem. The interviewer wants to see pairwise distance computation and clean handling of shapes.

## Clarifying questions

- Which distance metric: L2, L1, cosine?
- What should happen on ties?
- Are labels non-negative integers?
- Is memory an issue for large `N*M`?

## Code solution

```python
import numpy as np


def pairwise_l2_distances(X, Y):
    """
    X: [M, D]
    Y: [N, D]
    returns: [M, N] where output[i,j] = ||X[i] - Y[j]||_2
    """
    X2 = np.sum(X ** 2, axis=1, keepdims=True)       # [M, 1]
    Y2 = np.sum(Y ** 2, axis=1, keepdims=True).T     # [1, N]
    dist2 = X2 + Y2 - 2.0 * (X @ Y.T)                # [M, N]
    dist2 = np.maximum(dist2, 0.0)                   # avoid tiny negative values
    return np.sqrt(dist2)


def knn_predict(X_train, y_train, X_test, k=3):
    """
    Assumes y_train contains non-negative integer labels.
    """
    dists = pairwise_l2_distances(X_test, X_train)   # [M, N]
    nn_idx = np.argsort(dists, axis=1)[:, :k]        # [M, k]
    nn_labels = y_train[nn_idx]                      # [M, k]

    preds = []
    for row in nn_labels:
        preds.append(np.bincount(row).argmax())
    return np.array(preds)


# Tiny test
X_train = np.array([[0, 0], [1, 1], [10, 10], [11, 11]])
y_train = np.array([0, 0, 1, 1])
X_test = np.array([[0.2, 0.1], [10.5, 10.2]])
print(knn_predict(X_train, y_train, X_test, k=3))
```

## Explanation

The key identity is:

```text
||x - y||^2 = ||x||^2 + ||y||^2 - 2 x^T y
```

This computes all pairwise distances with one matrix multiplication. Shapes:

```text
X2:      [M, 1]
Y2.T:    [1, N]
X @ Y.T: [M, N]
```

Broadcasting gives the full `[M, N]` distance matrix.

The only small loop is over test examples for majority vote. That is usually acceptable, but you can vectorize it if the number of classes is small.

## Complexity

- Pairwise distances: `O(MND)`
- Sorting: `O(M N log N)`
- Memory: `O(MN)`

## Common mistakes

- Accidentally computing `[N, M]` instead of `[M, N]`
- Not clipping `dist2` before sqrt, causing `sqrt(-1e-12)`
- Using nested loops over test and train samples
- Ignoring memory blow-up when `M*N` is huge

## Follow-ups

- How would you avoid sorting all `N` distances? Use `np.argpartition`.
- How would you handle huge datasets? Process test samples in chunks or use approximate nearest neighbor search.
- How would you implement cosine KNN?

---

# Problem 5 - K-Means Clustering

## Interview question

Implement K-Means clustering from scratch. Given `X: [N, D]` and number of clusters `K`, return cluster assignments and centroids.

## What is being asked

You need to implement the two repeated steps of K-Means: assign points to nearest centroids and recompute centroids. The interviewer also expects edge-case handling, especially empty clusters.

## Clarifying questions

- How should centroids be initialized?
- How many iterations should I run?
- What should I do with empty clusters?
- Should I stop when centroids converge?

## Code solution

```python
import numpy as np


def pairwise_l2_distances(X, Y):
    X2 = np.sum(X ** 2, axis=1, keepdims=True)
    Y2 = np.sum(Y ** 2, axis=1, keepdims=True).T
    return np.sqrt(np.maximum(X2 + Y2 - 2 * X @ Y.T, 0.0))


def kmeans(X, K, num_steps=100, seed=0, tol=1e-6):
    """
    X: [N, D]
    Returns:
        labels: [N]
        centroids: [K, D]
    """
    N, D = X.shape
    rng = np.random.default_rng(seed)

    # Initialize using K random data points.
    init_idx = rng.choice(N, size=K, replace=False)
    centroids = X[init_idx].copy()

    for _ in range(num_steps):
        # Assignment step
        dists = pairwise_l2_distances(X, centroids)  # [N, K]
        labels = np.argmin(dists, axis=1)            # [N]

        # Update step
        new_centroids = np.zeros_like(centroids)
        for k in range(K):
            mask = labels == k
            if np.any(mask):
                new_centroids[k] = X[mask].mean(axis=0)
            else:
                # Empty cluster: keep old centroid.
                new_centroids[k] = centroids[k]

        shift = np.linalg.norm(new_centroids - centroids)
        centroids = new_centroids
        if shift < tol:
            break

    return labels, centroids
```

## Explanation

K-Means minimizes within-cluster squared distance:

```text
sum_i ||x_i - centroid_label_i||^2
```

Each iteration alternates between:

1. Assignment: choose nearest centroid for every point.
2. Update: replace each centroid with the mean of assigned points.

The assignment step is vectorized using pairwise distances. The centroid update uses a small loop over clusters, which is usually acceptable because `K` is much smaller than `N`.

## Complexity

- Time per iteration: `O(NKD)`
- Memory: `O(NK)` for distance matrix

## Common mistakes

- Not handling empty clusters
- Bad initialization causing poor local minima
- Forgetting convergence check
- Using loops over all data points instead of vectorized distance computation

## Follow-ups

- What is K-Means++ initialization?
- How would you scale to millions of points? Mini-batch K-Means.
- Why can K-Means fail on non-spherical clusters?

---

# Problem 6 - PCA with SVD

## Interview question

Implement PCA. Given `X: [N, D]`, return the top `num_components` principal components and the projected data.

## What is being asked

You need to center the data, find directions of maximum variance, and project the data onto those directions. SVD is usually the most stable implementation.

## Clarifying questions

- Should I use covariance eigendecomposition or SVD?
- Should outputs be principal directions, projected data, or reconstructed data?
- Are features already standardized?

## Code solution

```python
import numpy as np


def pca_svd(X, num_components):
    """
    X: [N, D]
    Returns:
        X_proj: [N, num_components]
        components: [num_components, D]
        explained_variance: [num_components]
        mean: [D]
    """
    mean = X.mean(axis=0, keepdims=True)      # [1, D]
    X_centered = X - mean                     # [N, D]

    # X_centered = U S Vt
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

    components = Vt[:num_components]         # [C, D]
    X_proj = X_centered @ components.T        # [N, C]

    explained_variance = (S[:num_components] ** 2) / (X.shape[0] - 1)
    return X_proj, components, explained_variance, mean.squeeze(0)


def pca_reconstruct(X_proj, components, mean):
    return X_proj @ components + mean
```

## Explanation

PCA finds orthogonal directions with maximum variance. The first principal component is the direction along which the data varies most.

SVD gives:

```text
X_centered = U S V^T
```

Rows of `V^T` are principal directions. Projecting onto the first `C` directions gives:

```text
X_proj = X_centered @ Vt[:C].T
```

SVD is preferred because forming the covariance matrix can be less stable and more expensive when `D` is large.

## Complexity

For full SVD:

- Time: roughly `O(min(ND^2, N^2D))`
- Memory: `O(ND)`

## Common mistakes

- Forgetting to center the data
- Using columns instead of rows as samples
- Returning eigenvectors in ascending rather than descending variance
- Confusing components shape `[C, D]` vs `[D, C]`

## Follow-ups

- How do you choose the number of components? Explained variance ratio.
- What if `D` is huge? Randomized PCA or truncated SVD.
- What is whitening?

---

# Problem 7 - 2D Convolution

## Interview question

Implement 2D convolution for a single-channel image using NumPy. Support padding and stride.

## What is being asked

You need to correctly handle indexing, output shape, padding, and stride. This is a classic deep learning operations question.

## Clarifying questions

- Is this convolution or cross-correlation? Deep learning libraries usually implement cross-correlation.
- Single channel or multi-channel?
- Should I support batches?
- Should padding be zero padding?

## Code solution

```python
import numpy as np


def conv2d_single_channel(x, kernel, stride=1, padding=0):
    """
    x: [H, W]
    kernel: [KH, KW]
    returns: [OH, OW]

    This implements cross-correlation, which is what most DL libraries call conv2d.
    """
    if padding > 0:
        x = np.pad(x, ((padding, padding), (padding, padding)), mode="constant")

    H, W = x.shape
    KH, KW = kernel.shape

    OH = (H - KH) // stride + 1
    OW = (W - KW) // stride + 1

    out = np.zeros((OH, OW), dtype=x.dtype)

    for i in range(OH):
        for j in range(OW):
            h0 = i * stride
            w0 = j * stride
            patch = x[h0:h0 + KH, w0:w0 + KW]
            out[i, j] = np.sum(patch * kernel)

    return out


# Optional vectorized version for stride=1, padding=0
from numpy.lib.stride_tricks import sliding_window_view


def conv2d_vectorized_stride1(x, kernel):
    windows = sliding_window_view(x, kernel.shape)  # [OH, OW, KH, KW]
    return np.sum(windows * kernel, axis=(-1, -2))
```

## Explanation

The output size is:

```text
OH = floor((H + 2P - KH) / stride) + 1
OW = floor((W + 2P - KW) / stride) + 1
```

At each output location, we multiply the kernel with a local image patch and sum the result.

Deep learning frameworks usually do not flip the kernel, so technically they implement cross-correlation. In interviews, mention this distinction and then proceed with the DL convention.

## Complexity

- Time: `O(OH * OW * KH * KW)`
- Memory: `O(OH * OW)`

## Common mistakes

- Wrong output size formula
- Off-by-one indexing error
- Forgetting padding changes `H` and `W`
- Confusing mathematical convolution with cross-correlation

## Follow-ups

- Extend to multi-channel input: kernel shape `[C, KH, KW]`.
- Extend to batch and multiple output channels.
- How does im2col speed up convolution?

---

# Problem 8 - LayerNorm

## Interview question

Implement LayerNorm forward pass. Given `x: [..., D]`, normalize over the last dimension and apply learnable `gamma` and `beta`.

## What is being asked

This tests tensor operations, broadcasting, and numerical stability. It is also directly relevant to transformers.

## Clarifying questions

- Normalize over which dimension?
- What is the shape of `gamma` and `beta`?
- Should I implement backward too?
- What epsilon should I use?

## Code solution

```python
import numpy as np


def layer_norm(x, gamma=None, beta=None, eps=1e-5):
    """
    x: [..., D]
    gamma: [D] or None
    beta: [D] or None
    returns: same shape as x
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_hat = (x - mean) / np.sqrt(var + eps)

    if gamma is not None:
        x_hat = x_hat * gamma
    if beta is not None:
        x_hat = x_hat + beta
    return x_hat


# Tiny test
x = np.random.randn(2, 3, 4)
gamma = np.ones(4)
beta = np.zeros(4)
y = layer_norm(x, gamma, beta)
print(y.shape)
print(y.mean(axis=-1))
print(y.var(axis=-1))
```

## Explanation

LayerNorm normalizes each token/sample independently across its feature dimension:

```text
mean = mean(x, dim=-1)
var = var(x, dim=-1)
y = gamma * (x - mean) / sqrt(var + eps) + beta
```

For transformer input `[B, T, D]`, each `[D]` vector is normalized independently. This is different from BatchNorm, which normalizes using statistics across the batch.

Broadcasting works because `mean` and `var` have shape `[B, T, 1]`, while `gamma` and `beta` have shape `[D]`.

## Complexity

- Time: `O(number of elements)`
- Memory: same shape as input for output

## Common mistakes

- Normalizing across batch dimension instead of feature dimension
- Forgetting `keepdims=True`
- Forgetting epsilon
- Using BatchNorm behavior in a transformer setting

## Follow-ups

- Why do transformers use LayerNorm instead of BatchNorm?
- What is RMSNorm?
- What is pre-norm vs post-norm transformer architecture?

---

# Problem 9 - Adam Optimizer

## Interview question

Implement one Adam optimizer update step for a parameter vector `w` and gradient `grad`.

## What is being asked

You need to understand first moment, second moment, bias correction, and numerical stability.

## Clarifying questions

- Is this Adam or AdamW?
- Should weight decay be included?
- Is `t` one-indexed or zero-indexed?
- Are parameters NumPy arrays?

## Code solution

```python
import numpy as np


def adam_update(w, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    w: parameter array
    grad: gradient array, same shape as w
    m: first moment estimate
    v: second moment estimate
    t: timestep, should start at 1
    """
    m = beta1 * m + (1.0 - beta1) * grad
    v = beta2 * v + (1.0 - beta2) * (grad ** 2)

    # Bias correction
    m_hat = m / (1.0 - beta1 ** t)
    v_hat = v / (1.0 - beta2 ** t)

    w = w - lr * m_hat / (np.sqrt(v_hat) + eps)
    return w, m, v


def adamw_update(w, grad, m, v, t, lr=1e-3, weight_decay=1e-2,
                 beta1=0.9, beta2=0.999, eps=1e-8):
    # Decoupled weight decay
    w = w * (1.0 - lr * weight_decay)
    return adam_update(w, grad, m, v, t, lr, beta1, beta2, eps)
```

## Explanation

Adam keeps an exponential moving average of gradients:

```text
m_t = beta1 * m_{t-1} + (1-beta1) * grad
```

and squared gradients:

```text
v_t = beta2 * v_{t-1} + (1-beta2) * grad^2
```

At early timesteps, both `m` and `v` are biased toward zero, so Adam uses bias correction:

```text
m_hat = m_t / (1 - beta1^t)
v_hat = v_t / (1 - beta2^t)
```

The final update scales the gradient by the inverse RMS of recent gradients.

## Complexity

- Time: `O(number of parameters)`
- Memory: `2x` parameter size for `m` and `v`

## Common mistakes

- Starting `t` at zero, causing division by zero
- Forgetting bias correction
- Putting epsilon inside vs outside sqrt without knowing the convention
- Confusing Adam L2 regularization with AdamW decoupled weight decay

## Follow-ups

- Why does Adam often converge faster than SGD?
- Difference between Adam and AdamW?
- Why can Adam generalize worse than SGD sometimes?

---

# Problem 10 - Multi-Head Self Attention

## Interview question

Implement the forward pass of multi-head self attention in PyTorch. Do not use `nn.MultiheadAttention`.

## What is being asked

You need to implement Q/K/V projections, split heads, scaled dot-product attention, masking, softmax, weighted sum, merging heads, and output projection.

## Clarifying questions

- Is input batch-first `[B, T, D]`?
- Should I include causal or padding mask?
- Is `D` divisible by number of heads?
- Should I include dropout?
- Should projection matrices be learned parameters or passed in?

## Code solution

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.qkv = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x, attn_mask=None):
        """
        x: [B, T, D]
        attn_mask: optional mask broadcastable to [B, H, T, T]
                   mask should be True for positions to mask out.
        returns: [B, T, D]
        """
        B, T, D = x.shape
        H = self.num_heads
        Dh = self.head_dim

        qkv = self.qkv(x)                         # [B, T, 3D]
        qkv = qkv.view(B, T, 3, H, Dh)            # [B, T, 3, H, Dh]
        qkv = qkv.permute(2, 0, 3, 1, 4)          # [3, B, H, T, Dh]
        q, k, v = qkv[0], qkv[1], qkv[2]          # each [B, H, T, Dh]

        scores = q @ k.transpose(-2, -1)          # [B, H, T, T]
        scores = scores / math.sqrt(Dh)

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, float("-inf"))

        attn = F.softmax(scores, dim=-1)          # [B, H, T, T]
        out = attn @ v                            # [B, H, T, Dh]

        out = out.transpose(1, 2).contiguous()    # [B, T, H, Dh]
        out = out.view(B, T, D)                   # [B, T, D]
        return self.out_proj(out)                 # [B, T, D]


def causal_mask(T, device=None):
    # True means masked out.
    return torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1)
```

## Explanation

For each token, attention computes a weighted average of value vectors. Weights come from similarity between queries and keys:

```text
Attention(Q,K,V) = softmax(QK^T / sqrt(d_head)) V
```

For multi-head attention, we split `D` into `H` heads, each of dimension `D/H`. This lets different heads attend to different relationships.

Shape walkthrough:

```text
x:      [B, T, D]
qkv:    [B, T, 3D]
q,k,v:  [B, H, T, Dh]
scores: [B, H, T, T]
out:    [B, H, T, Dh]
merged: [B, T, D]
```

Scaling by `sqrt(Dh)` prevents dot products from becoming too large, which would make softmax overly sharp.

## Complexity

- Time: `O(B * H * T^2 * Dh) = O(B * T^2 * D)`
- Memory: `O(B * H * T^2)` for attention scores

## Common mistakes

- Wrong transpose before `q @ k.T`
- Forgetting `.contiguous()` before `.view()`
- Dividing by `sqrt(D)` instead of `sqrt(Dh)`
- Applying softmax over the wrong dimension
- Using a mask with wrong broadcast shape

## Follow-ups

- How do you make it causal? Add upper-triangular mask.
- How do you support padding? Add padding mask.
- What is cross-attention? Q comes from one sequence, K/V from another.
- Why is FlashAttention more memory efficient? It avoids materializing the full attention matrix in HBM.

---

# Problem 11 - Transformer Encoder Block

## Interview question

Implement a transformer encoder block with LayerNorm, multi-head self attention, residual connections, and an MLP.

## What is being asked

You need to assemble standard transformer components and understand pre-norm architecture.

## Clarifying questions

- Pre-norm or post-norm?
- Should I include dropout?
- What MLP expansion ratio?
- Which activation: ReLU or GELU?

## Code solution

```python
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(mlp_ratio * dim))

    def forward(self, x, attn_mask=None):
        """
        x: [B, T, D]
        """
        x = x + self.attn(self.ln1(x), attn_mask=attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x
```

## Explanation

This is a pre-norm transformer block:

```text
x = x + Attention(LayerNorm(x))
x = x + MLP(LayerNorm(x))
```

Pre-norm is often easier to train for deep transformers because gradients can flow more directly through the residual path.

The attention mixes information across tokens. The MLP transforms each token independently along the feature dimension.

## Complexity

For sequence length `T` and hidden size `D`:

- Attention time: `O(B T^2 D)`
- MLP time: `O(B T D * hidden_dim)`
- Attention memory: `O(B H T^2)`

## Common mistakes

- Forgetting residual connections
- Normalizing after instead of before without realizing the difference
- MLP output dimension not matching `D`
- Not passing masks through attention

## Follow-ups

- What changes for a decoder block? Add causal self-attention and cross-attention.
- Why use GELU?
- What is the difference between encoder-only, decoder-only, and encoder-decoder transformers?

---

# Problem 12 - Patch Embedding for ViT/DiT

## Interview question

Implement image patch embedding. Given images `x: [B, C, H, W]`, split each image into non-overlapping patches and project each patch into a token of dimension `D`.

## What is being asked

This tests tensor reshaping and the connection between images and transformer tokens.

## Clarifying questions

- Are `H` and `W` divisible by patch size?
- Should I use manual reshape or a convolution?
- Should output be `[B, N, D]`?
- Should I include positional embeddings?

## Code solution

```python
import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, embed_dim=768):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Linear(in_channels * patch_size * patch_size, embed_dim)

    def forward(self, x):
        """
        x: [B, C, H, W]
        returns: [B, N, D], where N = (H/P)*(W/P)
        """
        B, C, H, W = x.shape
        P = self.patch_size
        assert H % P == 0 and W % P == 0

        # [B, C, H/P, P, W/P, P]
        x = x.view(B, C, H // P, P, W // P, P)

        # [B, H/P, W/P, C, P, P]
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous()

        # [B, N, C*P*P]
        x = x.view(B, (H // P) * (W // P), C * P * P)

        # [B, N, D]
        return self.proj(x)


class ConvPatchEmbed(nn.Module):
    def __init__(self, in_channels=3, patch_size=16, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size)

    def forward(self, x):
        x = self.proj(x)              # [B, D, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        return x
```

## Explanation

A Vision Transformer treats an image as a sequence of patch tokens. If the patch size is `P`, each patch has `C*P*P` raw values. A linear projection maps this flattened patch to an embedding vector of size `D`.

The convolution version is equivalent to patch extraction plus linear projection, because a convolution with kernel size `P` and stride `P` applies the same linear projection to every non-overlapping patch.

## Complexity

- Number of tokens: `N = (H/P) * (W/P)`
- Projection time: `O(B * N * C * P^2 * D)`
- Memory: `O(B * N * D)`

## Common mistakes

- Wrong `permute` order
- Forgetting `.contiguous()` before `.view()`
- Not checking divisibility by patch size
- Returning `[B, D, H/P, W/P]` instead of `[B, N, D]`

## Follow-ups

- How do you unpatchify tokens back into an image?
- How do you add positional embeddings?
- What changes in DiT? Patchify noisy latent/image and condition transformer blocks on timestep/class.

---

# Problem 13 - DDPM Forward Process

## Interview question

Implement the DDPM forward diffusion process `q_sample`. Given clean data `x0`, timestep indices `t`, and a noise schedule `alpha_bar`, return noisy samples `x_t`.

## What is being asked

You need to implement the closed-form forward noising equation and handle batched timesteps with broadcasting.

## Clarifying questions

- What shape is `x0`? Image `[B,C,H,W]` or vector `[B,D]`?
- Is `t` a vector of shape `[B]`?
- Should noise be passed in or sampled inside?
- Are we using NumPy or PyTorch?

## Code solution

```python
import torch


def extract(a, t, x_shape):
    """
    a: [T]
    t: [B]
    x_shape: shape of x, e.g. [B,C,H,W]
    returns: [B,1,1,1] broadcastable to x_shape
    """
    out = a.to(t.device)[t]  # [B]
    return out.view(t.shape[0], *([1] * (len(x_shape) - 1)))


def q_sample(x0, t, alpha_bar, noise=None):
    """
    x0: [B, ...]
    t: [B]
    alpha_bar: [T]
    """
    if noise is None:
        noise = torch.randn_like(x0)

    sqrt_ab = torch.sqrt(extract(alpha_bar, t, x0.shape))
    sqrt_one_minus_ab = torch.sqrt(1.0 - extract(alpha_bar, t, x0.shape))

    xt = sqrt_ab * x0 + sqrt_one_minus_ab * noise
    return xt


# Tiny setup
T = 1000
betas = torch.linspace(1e-4, 0.02, T)
alphas = 1.0 - betas
alpha_bar = torch.cumprod(alphas, dim=0)
```

## Explanation

The DDPM forward process gradually adds Gaussian noise. The useful closed-form equation is:

```text
x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * eps
```

where `eps ~ N(0, I)`.

The tricky part is `t`: each batch element can have a different timestep. `extract` gathers the right scalar for each sample and reshapes it to broadcast over the remaining dimensions.

For image input `[B,C,H,W]`, `extract` returns `[B,1,1,1]`, which broadcasts over channels and spatial dimensions.

## Complexity

- Time: `O(number of elements in x0)`
- Memory: output same size as `x0`

## Common mistakes

- Using `alpha_t` instead of cumulative `alpha_bar_t`
- Not reshaping gathered timestep values for broadcasting
- Sampling one timestep for the whole batch when per-example timesteps are expected
- Forgetting that `noise` should have the same shape as `x0`

## Follow-ups

- What target does the model predict? Often epsilon, sometimes x0 or velocity.
- What is classifier-free guidance?
- How is flow matching different from DDPM?

---

# Problem 14 - Camera Projection

## Interview question

Given 3D points, camera intrinsics `K`, rotation `R`, and translation `t`, project the 3D points into image coordinates.

## What is being asked

This tests basic geometry, vectorization, and handling homogeneous/perspective division.

## Clarifying questions

- Are points in world coordinates or camera coordinates?
- Is `R,t` world-to-camera or camera-to-world?
- Should points behind the camera be filtered?
- Should output include depth?

## Code solution

```python
import numpy as np


def project_points(points_world, K, R, t, eps=1e-8):
    """
    points_world: [N, 3]
    K: [3, 3]
    R: [3, 3] world-to-camera rotation
    t: [3] world-to-camera translation

    returns:
        uv: [N, 2]
        depth: [N]
    """
    # World to camera
    points_cam = points_world @ R.T + t.reshape(1, 3)  # [N, 3]

    x = points_cam[:, 0]
    y = points_cam[:, 1]
    z = points_cam[:, 2]

    z_safe = np.maximum(z, eps)

    # Perspective projection
    xn = x / z_safe
    yn = y / z_safe

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u = fx * xn + cx
    v = fy * yn + cy

    uv = np.stack([u, v], axis=1)
    return uv, z


# Tiny test
points = np.array([[0, 0, 1], [1, 1, 2]], dtype=float)
K = np.array([[100, 0, 50], [0, 100, 50], [0, 0, 1]], dtype=float)
R = np.eye(3)
t = np.zeros(3)
print(project_points(points, K, R, t))
```

## Explanation

First transform world points into the camera coordinate system:

```text
X_cam = R X_world + t
```

Then do perspective division:

```text
x_norm = X_cam_x / X_cam_z
y_norm = X_cam_y / X_cam_z
```

Finally apply intrinsics:

```text
u = fx * x_norm + cx
v = fy * y_norm + cy
```

The code is fully vectorized over all points.

## Complexity

- Time: `O(N)`
- Memory: `O(N)`

## Common mistakes

- Using camera-to-world transform when world-to-camera is expected
- Forgetting perspective division by `z`
- Not handling points with `z <= 0`
- Mixing row-vector and column-vector conventions

## Follow-ups

- How do you backproject a depth map to 3D?
- How do you compute reprojection error?
- How would you handle distortion parameters?

---

# Problem 15 - Chamfer Distance

## Interview question

Implement Chamfer distance between two point clouds `A: [N, D]` and `B: [M, D]`.

## What is being asked

This tests pairwise distances, nearest-neighbor reductions, and geometry/world-modeling intuition.

## Clarifying questions

- Squared or non-squared distances?
- Symmetric or one-directional Chamfer?
- Should I average or sum?
- Are point clouds batched?

## Code solution

```python
import numpy as np


def pairwise_squared_distances(A, B):
    """
    A: [N, D]
    B: [M, D]
    returns: [N, M]
    """
    A2 = np.sum(A ** 2, axis=1, keepdims=True)      # [N, 1]
    B2 = np.sum(B ** 2, axis=1, keepdims=True).T    # [1, M]
    return np.maximum(A2 + B2 - 2 * A @ B.T, 0.0)


def chamfer_distance(A, B):
    """
    Symmetric mean squared Chamfer distance.
    """
    dists = pairwise_squared_distances(A, B)  # [N, M]
    a_to_b = np.min(dists, axis=1).mean()
    b_to_a = np.min(dists, axis=0).mean()
    return a_to_b + b_to_a


# Tiny test
A = np.array([[0., 0.], [1., 0.]])
B = np.array([[0., 0.], [2., 0.]])
print(chamfer_distance(A, B))
```

## Explanation

Chamfer distance measures how close two point sets are by nearest-neighbor distance in both directions:

```text
CD(A,B) = mean_a min_b ||a-b||^2 + mean_b min_a ||b-a||^2
```

The symmetric version is important because one-directional distance can miss coverage issues. For example, every point in `A` might be close to some point in `B`, but `B` might have extra regions not covered by `A`.

## Complexity

- Time: `O(NMD)`
- Memory: `O(NM)`

## Common mistakes

- Computing only one direction
- Using Euclidean distance instead of squared distance without realizing it
- Creating an enormous `[N,M]` matrix for large point clouds
- Not handling empty point clouds

## Follow-ups

- How would you reduce memory? Chunk the pairwise distance computation.
- How does Chamfer differ from Earth Mover's Distance?
- How would you implement batched Chamfer in PyTorch?

---

# One-Day Priority Plan

If you only have one day, practice in this order:

1. Vectorized KNN
2. Softmax + Cross Entropy
3. Logistic Regression
4. K-Means
5. PCA
6. Convolution
7. LayerNorm
8. Multi-Head Attention
9. Transformer Encoder
10. Patch Embedding
11. DDPM Forward Process
12. Camera Projection
13. Chamfer Distance
14. Adam
15. Linear Regression

# Last-Minute Interview Checklist

Before coding:

```text
- Clarify input and output shapes.
- Clarify dtype and library: NumPy or PyTorch.
- Ask about edge cases.
- State the simple mathematical formula.
```

While coding:

```text
- Keep shapes in comments.
- Prefer vectorized operations.
- Avoid unnecessary Python loops.
- Use stable numerical forms.
- Write a tiny test.
```

After coding:

```text
- Analyze time complexity.
- Analyze memory complexity.
- Mention failure modes.
- Suggest how to scale or optimize.
```
