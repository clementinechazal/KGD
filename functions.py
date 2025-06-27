import numpy as np
import jax.numpy as jnp
from jax import jit, vmap
from jax import jacfwd, jacrev
from jax.scipy.stats import multivariate_normal


class GradientKernel:
    def __init__(self, s_PQ, k):
        self.s_PQ = s_PQ
        self.k = k
        self.dkx = jit(jacrev(k, argnums=0))
        self.dky = jit(jacrev(k, argnums=1))
        self.d2k = jit(jacfwd(self.dky, argnums=0))

    def evaluate(self, x, y):
        dkx = self.dkx(x, y)
        dky = self.dky(x, y)
        d2k = jnp.trace(self.d2k(x, y))
        k = self.k(x, y)
        s_PQx = self.s_PQ(x)
        s_PQy = self.s_PQ(y)
        k0 = d2k + \
            jnp.dot(dkx, s_PQy) + \
            jnp.dot(dky, s_PQx) + \
            k * jnp.dot(s_PQx, s_PQy)
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
    

class F_P:
    def __init__(self, q0, L):
        self.q0 = q0
        self.L = L
    def kl_divergence_kde(self,X, num_samples=1000):
        n, d = X.shape
    
        # KDE approximation
        def kde_density(data, points, bandwidth=1.0):
            diff = jnp.expand_dims(points, axis=1) - jnp.expand_dims(data, axis=0)  # (num_samples, n, d)
            dist = jnp.linalg.norm(diff, axis=2)  # (num_samples, n)
            weights = jnp.exp(-dist**2 / (2 * bandwidth**2))  # (num_samples, n)
            density = jnp.sum(weights, axis=1) / (n * (bandwidth * jnp.sqrt(2 * jnp.pi))**d)
            return jnp.log(density)

        # Resample points from KDE
        samples = X[np.random.choice(n, num_samples, replace=True)]  # Resampling from X
        log_qn = kde_density(X, samples)  # Log density of KDE at sampled points

        # Log density of standard Gaussian
        log_q0 = multivariate_normal.logpdf(samples, mean=jnp.zeros(d), cov=jnp.eye(d))

        # KL divergence estimate
        kl_estimate = jnp.mean(log_qn - log_q0)
        return kl_estimate

    def evaluate(self, X):
        KL = self.kl_divergence_kde(X)
        return self.L(X) + KL
        

    
# class NumericalExperiment:
#     def __init__(self,NumericalMethod,s_PQ, k, q0, L):
#         self.s_PQ = s_PQ
#         self.k = k
#         self.q0 = q0
#         self.L = L
#         self.k_PQ = GradientKernel(s_PQ, k)
#         self.KGD = KernelGradientDiscrepancy(self.gradient_kernel)
#         self.f_p = F_P(q0, L)
#         self.NumericalMethod = NumericalMethod

#         self.F_P = F_P(self.q0, self.L)
#         self.k_PQ = GradientKernel(s_PQ, k)
#         self.KGD = KernelGradientDiscrepancy(self.k_PQ)




#     def simulation(self,X):
        
#         if self.NumericalMethod == "MFLD":


