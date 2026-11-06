import numpy as np

def _sigmoid(z: np.ndarray) -> np.ndarray:
    """
    Returns elementwise sigmoid values.
    """
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X: np.ndarray, y: np.ndarray, lr: float = 0.1, steps: int = 1000) -> tuple[np.ndarray, float]:
    """
    Returns the trained weights and bias as (w, b).
    """
    # Write code here
    omega, beta = np.zeros(len(X[0])), 0
    n = len(X)

    for step in range(steps):
        p = _sigmoid(X @ omega + beta)
        domega = (1/n) * X.T @ (p-y)
        dbeta = (1/n) * np.sum(p-y)

        omega+= -lr * domega
        beta += -lr * dbeta

    return omega, beta