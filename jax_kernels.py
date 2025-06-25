import numpy as np
import jax.numpy as jnp
from jax import jit, vmap
from jax import jacfwd, jacrev

class GradientKernel:
    def __init__(self, gradV, k):
        self.gradV = gradV
        self.k = k
        self.dkx = jit(jacrev(k, argnums=0))
        self.dky = jit(jacrev(k, argnums=1))
        self.d2k = jit(jacfwd(self.dky, argnums=0))

    def evaluate(self, x, y):
        dkx = self.dkx(x, y)
        dky = self.dky(x, y)
        d2k = jnp.trace(self.d2k(x, y))
        k = self.k(x, y)
        gradVx = self.gradV(x)
        gradVy = self.gradV(y)
        k0 = d2k + \
            jnp.dot(dkx, gradVy) + \
            jnp.dot(dky, gradVx) + \
            k * jnp.dot(gradVx, gradVy)
        return k0
    


class KernelGradientDiscrepancy:
    def __init__(self, k0):
        self.k0 = k0
        self.vfk0 = jit(vmap(k0.evaluate, 0, 0))

    def k0mat(self, x):
        n = x.shape[0]
        ir, ic = np.tril_indices(n, k=-1)
        k0_tril = self.vfk0(x[ir], x[ic])
        k0_diag = self.vfk0(x, x)
        return (k0_tril, k0_diag, ir, ic)

    def evaluate(self, x):
        k0_tril, k0_diag, _, _ = self.k0mat(x)
        k0_sum = np.sum(k0_tril) * 2 + np.sum(k0_diag)
        return np.sqrt(k0_sum) / x.shape[0]

    def cumeval(self, x):
        n = x.shape[0]
        kgd = np.empty(n)
        k0_tril, k0_diag, ir, _ = self.k0mat(x)
        k0_sum = 0.
        for i in range(n):
            k0_sum += np.sum(k0_tril[ir == i]) * 2 + k0_diag[i]
            kgd[i] = np.sqrt(k0_sum) / (i + 1)
        return kgd



#######GAUSSIAN KERNEL#######
def k_gauss(x, y, sigma):  # Gram matrix for Gaussian kernel
    X_norm = jnp.sum(x**2, axis=1)
    Y_norm = jnp.sum(y**2, axis=1)
    xy = jnp.dot(x, y.T)
    dist = X_norm[:, None] + Y_norm[None, :] - 2 * xy
    return jnp.exp(-dist / (2 * sigma**2))

def dk_gauss(x, y, sigma):  # Gram matrix of nabla_1 k(x,y), dim = (n,m,d)
    x_mat = jnp.tile(x, (len(y), 1, 1))
    y_mat = jnp.tile(y, (len(x), 1, 1))
    x_mat = jnp.transpose(x_mat, axes=(1, 0, 2))
    diff = y_mat - x_mat
    return diff / (sigma**2) * k_gauss(x, y, sigma)[:, :, None]

def ddk_gauss(x, y, sigma):  # Gram matrix of nabla_2 . nabla_1 k(x,y), dim = (n,m)
    d = len(x[0])
    X_norm = jnp.sum(x**2, axis=1)
    Y_norm = jnp.sum(y**2, axis=1)
    xy = jnp.dot(x, y.T)
    dist = X_norm[:, None] + Y_norm[None, :] - 2 * xy
    K = jnp.exp(-dist / (2 * sigma**2))
    return 1 / sigma**2 * K * (d - 1 / sigma**2 * dist)


######## INVERSE MULTIQUADRIC KERNEL ########






######## LAPLACE KERNEL ########
def k_laplace(x, y, sigma):  # Gram matrix for Laplace kernel
    X_norm = jnp.sum(x**2, axis=1)
    Y_norm = jnp.sum(y**2, axis=1)
    xy = jnp.dot(x, y.T)
    dist = X_norm[:, None] + Y_norm[None, :] - 2 * xy
    return jnp.exp(-jnp.sqrt(dist) / sigma)

def dk_laplace(x, y, sigma):  # Gram matrix of nabla_1 k(x,y), dim = (n,m,d)
    x_mat = jnp.tile(x, (len(y), 1, 1))
    y_mat = jnp.tile(y, (len(x), 1, 1))
    x_mat = jnp.transpose(x_mat, axes=(1, 0, 2))
    diff = y_mat - x_mat
    dist = jnp.linalg.norm(diff, axis=2)  # Compute the Euclidean distance
    grad = jnp.zeros_like(diff)  # Initialize gradient to zero
    non_zero_mask = dist > 0  # Mask for non-zero distances
    expanded_mask = non_zero_mask[:, :, None]
    grad = grad.at[expanded_mask].set(
        -diff[expanded_mask] / (sigma * dist[non_zero_mask][:, :, None]) * jnp.exp(-dist[non_zero_mask] / sigma)[:, :, None]
    )
    return grad

def ddk_laplace(x, y, sigma):  # Gram matrix of nabla_2 . nabla_1 k(x,y), dim = (n,m)
    x_mat = jnp.tile(x, (len(y), 1, 1))
    y_mat = jnp.tile(y, (len(x), 1, 1))
    x_mat = jnp.transpose(x_mat, axes=(1, 0, 2))
    diff = y_mat - x_mat
    dist = jnp.linalg.norm(diff, axis=2)  # Compute the Euclidean distance
    K = jnp.exp(-dist / sigma)  # Laplace kernel values
    d = x.shape[1]  # Dimensionality of the input data
    second_derivative = jnp.zeros((x.shape[0], y.shape[0]))  # Initialize second derivative matrix to zero
    non_zero_mask = dist > 0  # Mask for non-zero distances
    second_derivative = second_derivative.at[non_zero_mask].set(
        K[non_zero_mask] * ((d - 1) / (sigma**2 * dist[non_zero_mask]) - 1 / sigma**3)
    )
    return second_derivative