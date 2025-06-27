import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from jax import jit, vmap
from jax import jacfwd, jacrev
from jax.scipy.stats import multivariate_normal
from functions import F_P, GradientKernel,KernelGradientDiscrepancy





class MeanFieldLangevinDynamics:
    def __init__(self,q0,s_q0,L,gradL,k):
        self.q0 = q0
        self.s_q0 = s_q0
        self.L = L
        self.gradL = gradL
        self.s_PQ = lambda x : self.s_q0(x) - self.gradL(x)
        self.k = k

        self.F_P = F_P(self.q0, self.L)
        self.k_PQ = GradientKernel(self.s_PQ, self.k)
        self.KGD = KernelGradientDiscrepancy(self.k_PQ)

    # def simulation(self,Eta,T,X0):
    #     n, d = X0.shape
    #     n_sim = 5
    #     KGD_fin= np.zeros((T,(len(Eta))))
    #     F_P_fin = np.zeros((n_sim,(len(Eta))))
    #     for sim in range(n_sim):
    #         for e in range(len(Eta)):
    #             X = X0.copy()
    #             for it in range(T):
    #                 Z = np.random.randn(n, d)
    #                 s_PQ_X = self.s_PQ(X)
    #                 X = X + Eta[e] * s_PQ_X + np.sqrt(2 * Eta[e]) * Z
                
    #             KGD_fin[sim, e] = self.KGD.evaluate(X)
    #             F_P_fin[sim, e] = self.F_P.evaluate(X)
                    
    #     KGD_val_avg = np.mean(KGD_fin, axis=0)
    #     F_P_val_avg = np.mean(F_P_fin, axis=0)

    #     plt.figure()
    #     plt.plot(Eta, KGD_val_avg)
    #     plt.xlabel('Eta')
    #     plt.ylabel('Final KGD Value')
    #     plt.xscale('log')
    #     plt.title('Final KGD Values with Different Eta')
    #     plt.show()


    #     plt.plot(Eta, F_P_val_avg)
    #     plt.xlabel('Eta')
    #     plt.ylabel('Final F_P Value')
    #     plt.xscale('log')
    #     plt.title('Final F_P Values with Different Eta')
    #     plt.show()

    #     return KGD_val_avg, F_P_val_avg

    def run_particles(self,eta,T,X0):
        n,d = X0.shape
        KGD_values = np.zeros(T)
        F_P_values = np.zeros(T)
        all_particles = np.zeros((T, n, d))

        X = X0
        for it in range(T):
            Z = np.random.randn(n, d)
            X = X + eta * self.s_PQ(X) + np.sqrt(2*eta)* Z
            
            KGD_values[it] = self.KGD.evaluate(X)
            all_particles[it] = X
            F_P_values[it] = self.F_P.evaluate(X)
        return KGD_values, F_P_values, all_particles




    

        
