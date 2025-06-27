import numpy as np
import jax.numpy as jnp
from jax import jit, vmap
from jax import jacfwd, jacrev
from jax.scipy.stats import multivariate_normal
from fonctions import F_P, GradientKernel,KernelGradientDiscrepancy





class MeanFieldLangevinDynamics:
    def __init__(self,q0,s_q0,L,gradL,kn,T):
        self.q0 = q0
        self.s_q0 = s_q0
        self.L = L
        self.gradL = gradL
        self.n = n
        self.T = T
        self.s_PQ = lambda x : self.s_q0(x) - self.gradL(x)

        self.F_P = F_P(self.q0, self.L)
        self.k_PQ = GradientKernel(self.s_PQ, self.k)
        self.KGD = KernelGradientDiscrepancy(self.k_PQ)

        def selection_eta(self,Eta):

            n_sim = 5
            KGD_fin= np.zeros((n_sim,(len(Eta))))
            F_P_fin = np.zeros((n_sim,(len(Eta))))
            for sim in range(n_sim):
                for e in range(len(Eta)):
                    X = np.random.randn(n, d) + np.array([-5, 5])
                    for it in range(T):
                        Z = np.random.randn(n, d)
                        X = X - Eta[e] * gradV(X) + np.sqrt(2 * Eta[e]) * Z
                    KGD_fin[sim, e] = KGD.evaluate(X)
                    F_P_fin[sim, e] = F(L, X)
                        
            KGD_val_avg = np.mean(KGD_fin, axis=0)
            F_P_val_avg = np.mean(F_P_fin, axis=0)

            plt.figure()
            plt.plot(Eta, KGD_val_avg)
            plt.xlabel('Eta')
            plt.ylabel('Final KGD Value')
            plt.xscale('log')
            plt.title('Final KGD Values with Different Eta')
            plt.show()


            plt.plot(Eta, F_P_val_avg)
            plt.xlabel('Eta')
            plt.ylabel('Final F_P Value')
            plt.xscale('log')
            plt.title('Final F_P Values with Different Eta')
            plt.show()

        

        
