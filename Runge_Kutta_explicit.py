import numpy as np

def R_K_method_general(f, a: np.float32, b: np.float32, y_0: np.float32, N: np.uint32, order: int, A: np.ndarray, C: np.ndarray, W: np.ndarray, decimals: int = 5):
    h = (b - a)/N
    x_n = np.array([a + n*h for n in range(N+1)], dtype=np.float32)
    y_n = np.zeros((N+1,), dtype=np.float32)
    y_n[0] = y_0
    for n in range(N):
        k = np.zeros((order,), dtype=np.float32)
        for i in range(order):
            k[i] = np.round(h * f(x_n[n] + C[i] * h, y_n[n] + A[i, :].T @ k), decimals)
        y_n[n + 1] = np.round(y_n[n] + W.T @ k, decimals)
    
    return x_n, y_n

def R_K2_Euler_Cauchy(f, a: np.float32, b: np.float32, y_0: np.float32, N: np.uint32, decimals: int):
    A = np.array([[0, 0], [1, 0]], dtype=np.float32)
    C = np.array([0, 1], dtype=np.float32)
    W = np.array([1/2, 1/2], dtype=np.float32)
    return R_K_method_general(f, a, b, y_0, N, 2, A, C, W, decimals)

def R_K2_Improved_Tangent(f, a: np.float32, b: np.float32, y_0: np.float32, N: np.uint32, decimals: int):
    A = np.array([[0, 0], [1/2, 0]], dtype=np.float32)
    C = np.array([0, 1/2], dtype=np.float32)
    W = np.array([0, 1], dtype=np.float32)
    return R_K_method_general(f, a, b, y_0, N, 2, A, C, W, decimals)

def R_K3_Nystrom(f, a: np.float32, b: np.float32, y_0: np.float32, N: np.uint32, decimals: int):
    A = np.array([[0, 0, 0],
                  [2/3, 0, 0],
                  [0, 2/3, 0]], dtype=np.float32)
    C = np.array([0, 2/3, 2/3], dtype=np.float32)
    W = np.array([2/8, 3/8, 3/8], dtype=np.float32)
    return R_K_method_general(f, a, b, y_0, N, 3, A, C, W, decimals)

def R_K4_Classical(f, a: np.float32, b: np.float32, y_0: np.float32, N: np.uint32, decimals: int):
    A = np.array([[0, 0, 0, 0],
                  [1/2, 0, 0, 0],
                  [0, 1/2, 0, 0],
                  [0, 0, 1, 0]], dtype=np.float32)
    C = np.array([0, 1/2, 1/2, 1], dtype=np.float32)
    W = np.array([1/6, 2/6, 2/6, 1/6], dtype=np.float32)
    return R_K_method_general(f, a, b, y_0, N, 4, A, C, W, decimals)
