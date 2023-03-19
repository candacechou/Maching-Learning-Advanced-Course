import pkg_resources
# pkg_resources.require("matplotlib==2.1.1")

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
import sys
from scipy.stats import multivariate_normal, poisson

epsilon = sys.float_info.epsilon
def bivariate_normal(X, Y, sigmax=1.0, sigmay=1.0,
                 mux=0.0, muy=0.0, sigmaxy=0.0):
    """
    Bivariate Gaussian distribution for equal shape *X*, *Y*.
    See `bivariate normal
    <http://mathworld.wolfram.com/BivariateNormalDistribution.html>`_
    at mathworld.
    """
    Xmu = X-mux
    Ymu = Y-muy

    rho = sigmaxy/(sigmax*sigmay)
    z = Xmu**2/sigmax**2 + Ymu**2/sigmay**2 - 2*rho*Xmu*Ymu/(sigmax*sigmay)
    denom = 2*np.pi*sigmax*sigmay*np.sqrt(1-rho**2)
    return np.exp(-z/(2*(1-rho**2))) / denom

def generate_data(n_data, means, covariances, weights, rates):
    n_clusters, n_features = means.shape
    data = np.zeros((n_data, n_features))
    poission_data = np.zeros(n_data)
    colors = np.zeros(n_data, dtype='str')
    for i in range(n_data):
        # pick a cluster id and create data from this cluster
        k = np.random.choice(n_clusters, size=1, p=weights)[0]
        x = np.random.multivariate_normal(means[k], covariances[k])
        data[i] = x
        poission_data[i] = np.random.poisson(rates[k])
        if k == 0:
            colors[i] = 'red'
        elif k == 1:
            colors[i] = 'blue'
        elif k == 2:
            colors[i] = 'green'

    return data, poission_data, colors


# means, covs: means and covariances of Gaussians
# rates: rates of Poissons
# title: title of the plot defining which EM iteration
def plot_contours(X, S, means, covs, title, rates):
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], s=S)

    delta = 0.025
    k = means.shape[0]
    x = np.arange(-2.0, 7.0, delta)
    y = np.arange(-2.0, 7.0, delta)
    X, Y = np.meshgrid(x, y)
    col = ['green', 'red', 'indigo']
    for i in range(k):
        mean = means[i]
        cov = covs[i]
        sigmax = np.sqrt(cov[0][0])
        sigmay = np.sqrt(cov[1][1])
        sigmaxy = cov[0][1] / (sigmax * sigmay)
        Z = bivariate_normal(X, Y, sigmax, sigmay, mean[0], mean[1], sigmaxy)
        plt.contour(X, Y, Z, colors=col[i], linewidths=rates[i], alpha=0.1)

    plt.title(title)
    plt.tight_layout()


class EM:

    def __init__(self, n_components, n_iter, tol, seed):
        self.n_components = n_components
        self.n_iter = n_iter
        self.tol = tol
        self.seed = seed

    def fit(self, X, S):

        # data's dimensionality
        self.n_row, self.n_col = X.shape

        # initialize parameters
        np.random.seed(self.seed)
        chosen = np.random.choice(self.n_row, self.n_components, replace=False)
        self.means = X[chosen]
        self.weights = np.full(self.n_components, 1 / self.n_components)
        if self.n_components == 3:
            self.rates = (np.mean(S) * np.ones(self.n_components) / np.array([1, 2, 3])[np.newaxis]).flatten()
        elif self.n_components == 2:
            self.rates = (np.mean(S) * np.ones(self.n_components) / np.array([1, 2])[np.newaxis]).flatten()
        shape = self.n_components, self.n_col, self.n_col
        self.covs = np.full(shape, np.cov(X, rowvar=False))
        new_covs = []
        for c in self.covs:
            new_covs = np.append(new_covs, np.diag(np.diag(c))) # making the covariances diagonal (question assumption)
        self.covs = np.array(new_covs).reshape(self.n_components, 2, 2)

        log_likelihood = 0
        self.converged = False

        for i in range(self.n_iter):
            self._do_estep(X, S)
            self._do_mstep(X, S)
            log_likelihood_new = self._compute_log_likelihood(X, S)

            if (log_likelihood - log_likelihood_new) <= self.tol:
                self.converged = True
                print("Convergence achieved!")
                break

            log_likelihood = log_likelihood_new

        return self

    def _do_estep(self, X, S):
        """
        E-step
        """
        n_samples = self.n_row
        n_cluster = self.n_components
        self.r = np.zeros((n_samples,n_cluster))
        for n in range(n_samples):
            for k in range(n_cluster):
                self.r[n][k] = self.weights[k] * poisson.pmf(S[n], self.rates[k]) * \
                    multivariate_normal.pdf(X[n],mean = self.means[k],cov = self.covs[k])
                self.r[n][k] += sys.float_info.epsilon
        self.r = self.r / np.tile(np.sum(self.r , axis = 1).reshape((n_samples, 1)),n_cluster)
        ### update weight
        for k in range(n_cluster):
            self.weights[k] = np.sum(self.r[:,k])/n_samples

        return self

    def _do_mstep(self, X, S):
        """M-step, update parameters"""
        n_samples = self.n_row
        ### update mu
        for k in range(self.n_components):
            N = np.sum(self.r[:,k])
            self.means[k][0] = np.sum(self.r[:,k] * X[:,0]) / N
            self.means[k][1] = np.sum(self.r[:,k] * X[:,1]) / N
            self.rates[k] = np.sum(S * self.r[:,k]) / N
            dotx1 = (X[:,0] - self.means[k][0]).reshape(n_samples, 1)
            self.covs[k][0][0] = np.sum(dotx1**2 * self.r[:, k].reshape(n_samples, 1)) / N
            dotx2 = (X[:,1] - self.means[k][1]).reshape(n_samples, 1)
            self.covs[k][1][1] = np.sum(dotx2**2 * self.r[:, k].reshape(n_samples, 1)) / N

        return self

    def _compute_log_likelihood(self, X, S):
        """compute the log likelihood of the current parameter"""
        n_samples = self.n_row
        for n in range(n_samples):
            for k in range(self.n_components):
                self.r[n][k] = self.weights[k] * poisson.pmf(S[n], self.rates[k])\
                                                * multivariate_normal.pdf(X[n], self.means[k],
                                                    self.covs[k])
        log_likelihood = 0
        log_likelihood = np.sum(np.log(np.sum(self.r, axis=1)))

        return log_likelihood

###############################################################################################

# params for 3 clusters
means = np.array([
    [5, 0],
    [1, 1],
    [0, 5]
])

covariances = np.array([
    [[.5, 0.], [0, .5]],
    [[.92, 0], [0, .91]],
    [[.5, 0.], [0, .5]]
])

weights = [1 / 4, 1 / 2, 1 / 4]

# params for 2 clusters
means_2 = np.array([
    [5, 0],
    [1, 1]
])

covariances_2 = np.array([
    [[.5, 0.], [0, .5]],
    [[.92, 0], [0, .91]]
])

weights_2 = [1 / 4, 3 / 4]

np.random.seed(3)

rates = np.random.uniform(low=.2, high=20, size=3)
print("Poisson rates for 3 components:")
print(rates)

rates_2 = np.random.uniform(low=.2, high=20, size=2)
print("Poisson rates for 2 components:")
print(rates_2)

# generate data
X, S, colors = generate_data(100, means, covariances, weights, rates)
plt.scatter(X[:, 0], X[:, 1], s=S, c=colors) # the Poisson data is shown through size of the points: s
plt.show()

X_2, S_2, colors_2 = generate_data(100, means_2, covariances_2, weights_2, rates_2)
plt.scatter(X_2[:, 0], X_2[:, 1], s=S_2, c=colors_2) # the Poisson data is shown through size of the points: s
plt.show()

###############################################################################################
print("1")
em = EM(n_components=3, n_iter=1, tol=1e-4, seed=1)
em.fit(X, S)

# plot: call plot_contours and give it the params updated from EM with 3 components (after 1 iteration)
plot_contours(X, S, em.means, em.covs, "Result after 1 iteration for data set 1.", em.rates)
plt.show()

print("2")
em = EM(n_components=3, n_iter=50, tol=1e-4, seed=1)
em.fit(X, S)

# plot: call plot_contours and give it the params updated from EM with 3 components (after 50 iterations)
plot_contours(X, S, em.means, em.covs, "Result after 50 iterations for data set 1.", em.rates)
plt.show()

print("3")
em_2 = EM(n_components=2, n_iter=1, tol=1e-4, seed=1)
em_2.fit(X_2, S_2)

# plot: call plot_contours and give it the params updated from EM with 2 components (after 1 iteration)
plot_contours(X_2, S_2, em_2.means, em_2.covs, "Result after 1 iteration for data set 2.", em_2.rates)
plt.show()

print("4")
em_2 = EM(n_components=2, n_iter=50, tol=1e-4, seed=1)
em_2.fit(X_2, S_2)

# plot: call plot_contours and give it the params updated from EM with 2 components (after 50 iterations)
plot_contours(X_2, S_2, em_2.means, em_2.covs, "Result after 50 iteration for data set 2.", em_2.rates)
plt.show()