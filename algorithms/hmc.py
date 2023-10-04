import matplotlib.pyplot as plt
%config InlineBackend.figure_format = 'retina'
import seaborn as sns
sns.set_theme('paper', style='darkgrid')
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.metrics import mean_absolute_error
from algorithms.exploratory_analysis import ExploratoryAnalysis
from scipy import stats
import algorithms.hmc_Lab as hmc
from algorithms.mle import MLE


class Gaussian:
    """2D Gaussian example"""

    def __init__(self):
        self.covar = np.array([[1.0875, 1],[1, 1.0875]])
        self.R = 5000
        self.L = 20
        self.eps = 0.9
        self.burn = int(self.R/10) 

        # Plotting parameters
        self.f = 5  # The "frequency" argument for the energy, used here to demonstrate how to use "args"

        # Other plotting parameters
        self.fsz = (10, 8)
        self.gsz = 100
        self.lim = 3


    def energy_func(self, x, covar):
        """Energy function. Returns Neglogpdf."""

        return np.negative(stats.multivariate_normal.logpdf(x, cov=covar))
        
    def energy_grad(self, x, covar):
        """Gradient function."""

        return np.linalg.inv(covar) @ x


    # Setup the mesh grid
    def plot_posterior(self):
        """Plot 2D Gaussian Posterior"""

        gx = np.linspace(-self.lim, self.lim, self.gsz)
        self.GX, self.GY = np.meshgrid(gx, gx)

        GX = self.GX
        GY = self.GY

        G = np.hstack((GX.reshape((GX.size, 1)), GY.reshape((GX.size, 1))))

        # Plot the figure
        plt.figure(figsize = self.fsz)
        self.P = np.asarray([np.exp(-self.energy_func(g, self.covar)) for g in G])
        plt.contour(GX, GY, self.P.reshape((self.gsz, self.gsz)), cmap='Reds', linewidths=3, zorder=1);

        try:
            if not isinstance(self.covar ,np.ndarray):
                print('covar  should be an array')        
            if self.covar.shape != (2,2):
                print('covar  is arranged in wrong shape')
        except Exception as err:
            print('Error output:', err)


    def chain(self):
        """Hamiltonian chain"""

        # Initialise the state for the first time step
        x0 = np.random.normal(size=2)

        # Call the function from the pre-defined hmc module
        hmc.gradient_check(x0, self.energy_func, self.energy_grad, self.covar)


        np.random.seed(seed=1)  # Fix the random seed for better reproducibility

        # HMC sampling
        self.S, *_ = hmc.sample(x0, self.energy_func, self.energy_grad, self.R, self.L,
                        self.eps, burn = self.burn, checkgrad=True, args = [self.covar])
        
        # Plot posterior
        fig, ax = plt.subplots(figsize=(10,10))

        plt.plot(self.S[:, 0], self.S[:, 1], '.', ms=6, color='CadetBlue', alpha=0.25, zorder=0)
        plt.contour(self.GX, self.GY, self.P.reshape((self.gsz, self.gsz)), cmap='Reds', linewidths=3, zorder=1);

        # Plot the Marginal distribution
        (sns.jointplot(x=self.S[:, 0], y=self.S[:, 1], marker='.', height=10, space=0)
        .plot_joint(sns.kdeplot, n_levels=20, cmap="RdBu_r")).set_axis_labels('x', 'y');
    

    def plot_means(self):
        """Plot Mean estimates"""

        S_mean = np.cumsum(self.S,axis=0)/(np.array(range(len(self.S)))+1).reshape(-1,1)
        fig, ax = plt.subplots(figsize=(10,6))
        plt.plot(S_mean[:,0],label='Estimate for $\mu_1$')
        plt.plot(S_mean[:,1],label='Estimate for $\mu_2$')

        plt.plot([0,5000],[0,0],'b--')
        plt.legend(loc='upper right')
        plt.title('Convergence of means')
        plt.tight_layout()


class Hamiltonian:
    """HMC to Linear Regression Model"""

    def __init__(self):
        self.R = 20000
        self.burn = int(self.R/10) 
        self.L = 100 
        self.eps = 0.0025
        self.hps = np.array([0.1, 0.1] + list(mu_n))

    def calc_errors(self, X_train, y_train, X_test, y_test, mu_n):
        """RMSE and MAE errors."""

        rmse_train = ExploratoryAnalysis.error_rms(y_train, np.dot(X_train, mu_n))
        rmse_test = ExploratoryAnalysis.error_rms(y_test, np.dot(X_test, mu_n))
        mae_train = mean_absolute_error(y_train, np.dot(X_train, mu_n))
        mae_test = mean_absolute_error(y_test, np.dot(X_test, mu_n))
        return rmse_train, rmse_test, mae_train, mae_test

    def energy_func_lr(self, hps, x, y):
        """Energy function."""

        alpha = hps[0]
        beta = hps[1]
        w = np.array(hps[2:])
        
        M = x.shape[1]
        N = x.shape[0]
        
        a = (N / 2 * np.log(beta)) - ((beta/2)*np.sum((y - x @ w) ** 2))
        b = (M / 2 * np.log(alpha)) - ((alpha / 2) * np.sum(w ** 2))

        neglgp = - (a + b)
        return neglgp

    def energy_grad_lr(self, hps, x, y):
        """Gradient function returns arr of partial derivatives of the energy function."""

        alpha = hps[0]
        beta = hps[1]
        w = np.array(hps[2:])
        
        M = x.shape[1]
        N = x.shape[0]
        
        grad_alpha = - M/(2*alpha) + np.sum(w ** 2)/2
        grad_beta = -N/(2*beta) +  0.5*np.sum((y - x @ w) ** 2)
        grad_w = (alpha*w) - beta*((y - x @ w)@x)

        g = np.array([grad_alpha, grad_beta] + list(grad_w))

        return g

    def chain(self, X_train, Y_train):
        """Hamiltonian chain"""


        np.random.seed(seed=1)  

        self.S, *_ = hmc.sample(self.hps, self.energy_func_lr, 
                           self.energy_grad_lr, self.R, self.L, 
                           self.eps, burn = self.burn, checkgrad=True, 
                           args=[X_train, Y_train])
        
        S = self.S

        # Plot Marginal distributions
        (sns.jointplot(x=S[:, 0], y=S[:, 1], marker='.', height=10, space=0)
        .plot_joint(sns.kdeplot, n_levels=20, cmap="RdBu_r")).set_axis_labels('x', 'y');

        # Plot Estimated means
        self.S_mean = np.cumsum(S, axis=0) / (np.array(range(len(S))) + 1).reshape(-1,1)
        S_mean = self.S_mean
        
        fix, ax = plt.subplots(ncols=3, figsize=(20,6))
        ax[0].plot(S_mean[:,0])
        ax[0].set_xlabel('# Samples')
        ax[0].set_title(r'$\alpha$')

        ax[1].plot(S_mean[:,1])
        ax[1].set_xlabel('# Samples')
        ax[1].set_title(r'$\beta$')

        ax[2].plot(S_mean[:,-1])
        ax[2].set_xlabel('# Samples')
        ax[2].set_title('Bias')

        self.alpha = S_mean[-1,0]
        print(f"Expected Alpha: {self.alpha}")
        print(f"Expected Beta: {S_mean[-1,1]}")
        print(f'Expected Bias = {S_mean[-1,-1]}')

    def plot_posterior(self, X_train, y_train):
        """Plot posterior"""

        alphas = np.exp(np.linspace(-5, 0, 100))
        betas = np.exp(np.linspace(-5, 0, 100))

        log_prob_y = np.array([
            MLE.compute_log_marginal(X_train, y_train, alph, beta) 
            for alph in alphas 
            for beta in betas
            ]).reshape(100,100)

        X, Y = np.log(np.meshgrid(alphas, betas))

        # Posterior
        S_mean = self.S_mean

        fig, ax, = plt.subplots(figsize=(10,10))
        plt.contourf(X, Y, log_prob_y.T, levels=20);
        plt.scatter(np.log(self.alpha), np.log(S_mean[-1,1]), c='r')
        plt.annotate(r'log($\alpha)=$' + f'{np.log(self.alpha):.2f}\n'+ 
                     r'log($\beta)=$' + f'{np.log(S_mean[-1,1]):.2f}\n', 
                     (np.log(self.alpha)-0.4, np.log(S_mean[-1,1])))
        
        plt.xlabel(r'log($\alpha$)')
        plt.ylabel(r'log($\beta$)');

    def get_predictions(self, X_train, y_train, X_test, y_test):
        """Get predictions for Hamiltonian Markov Chains"""

        S_mean = np.cumsum(S, axis=0) / (np.array(range(len(self.S))) + 1).reshape(-1,1)
        best_w = S_mean[-1,2:]
        rmse_train, rmse_test, mae_train, mae_test = self.calc_errors(best_w)

        print(f"RMSE Train (hmc): {rmse_train}")
        print(f"RMSE Test (hmc): {rmse_test}")
        print(f"MAE Train (hmc): {mae_train}")
        print(f"MAE Test (hmc): {mae_test}")

        y_pred_train = np.dot(X_train, best_w)
        y_pred_test = np.dot(X_test, best_w)

        cov = (np.identity(best_w.shape[0]) + np.matmul(best_w, best_w.T)) 
        std_pred_train =  np.sqrt(np.diag(X_train @ cov @ X_train.T)) / 10
        std_pred_test = np.sqrt(np.diag(X_test @ cov @ X_test.T)) / 10

        ExploratoryAnalysis.plot_pred(y_train, y_pred_train, 
                                      mae_train, std_pred_train, 
                                      'Hamiltonian Monte Carlo', 'Training Set')

        ExploratoryAnalysis.plot_pred(y_test, y_pred_test, 
                                      mae_test, std_pred_test, 
                                      'Hamiltonian Monte Carlo', 'Test Set')
