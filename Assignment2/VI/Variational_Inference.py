import numpy as np
import matplotlib.pyplot as plt
from math import gamma

MAXITER =  100
THRES = 0.001

def generateData(mean,var,N):

    Data = np.random.normal(mean,var,N)
    return Data

def estimatePosterior(mu,muN,LN,tau,aN,bN):
    exponent = - LN * (mu-muN)**2 / 2
    q = (np.sqrt(LN/(2 * np.pi)) * np.exp(exponent) ) * ((1.0 /gamma(aN)) * (bN**aN) * (tau**(aN-1)) * np.exp(- bN * tau))
    return q


def VI_algorithm(D,N,L_0,mu_0,a_0,b_0,b_N,L_N,mu,tau,num_iter = 0,save_fig = None):
    i = 0
    Post_esti = []
    a_N = a_0 + (N+1)/2.0
    xbar = np.sum(D) / N
    mu_N = (L_0 * mu_0 + N * xbar) /(L_0 + N) ## iteration
    num_iter = num_iter if num_iter != 0 else MAXITER
    while(i < num_iter):
    ## EXPECT VALUES
        E_tau = a_N / b_N
        E_mu = mu_N
        E_mu2 = mu_N ** 2 + 1 / L_N
        L_N = (L_0 + N) * E_tau
        b_N = b_0 + 0.5 * (np.sum(D**2) + L_0 * (mu_0**2)) - \
                 (np.sum(D) + mu_0 * L_0) * E_mu + \
                 0.5 * (L_0 + N) * E_mu2

        esti = estimatePosterior(mu,mu_N,L_N,tau,a_N,b_N) 
        Post_esti.append(esti)
        i=i+1
        

    return Post_esti

    
def truePosterior(D,N,mu,tau,mu0,L0,a0,b0):
    ## P(D|mu,tau)
    sumup = np.sum(D**2) - 2 * np.sum(D) * mu + N * mu**2 
    P_Likelihood = (tau / (2 * np.pi))**(N/2) * np.exp(-(tau/2) * sumup) ## P(mu | tau)
    sigma = 1/(L0 * tau)
    exponent =  - (mu0 - mu)**2 / (2 * sigma)
    P_prior = 1 / (np.sqrt(2 * np.pi * sigma)) * np.exp(exponent)
    ## P(tau)
    P_tau = (1 /gamma(a0)) * (b0**a0) * (tau**(a0-1)) * np.exp(-b0 * tau)

    return P_Likelihood *  P_prior * P_tau

def main():
    N = 10
    mean = 0
    var = 1
    Data = generateData(mean,var,N)
    ## some parameters
    mu_0 = 1e-16
    a_0 = 1e-16
    b_0 = 1e-16
    L_0 = 1e-16 ## lambda 0 bN =0.4
    L_N = 0.2
    b_N = 0.4
    iternum = 6
    ## some tau and some mu
    tau = np.linspace(0,4,100)
    mu = np.linspace(-2,2,100) 
    mu,tau = np.meshgrid(mu,tau)
    ## VI algorithm
    Pesti = VI_algorithm(Data,N,L_0,mu_0,a_0,b_0,b_N,L_N,mu,tau,iternum)
    trueP = truePosterior(Data,N,mu,tau,mu_0,L_0,a_0,b_0)

    ### plot the figure
    plt.figure()
    for i in range(iternum):
        plt.subplot(2,3,i+1)
        plt.contour(tau, mu,Pesti[i],colors="red")
        plt.contour(tau, mu,trueP, colors = "Blue")
        plt.xlabel('$mu$')
        plt.ylabel('tau')
        plt.title(f"iter = {i+1}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
