import matplotlib.pyplot as plt
%config InlineBackend.figure_format = 'retina'
import seaborn as sns
sns.set_theme('paper', style='darkgrid')
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.metrics import mean_absolute_error
from exploratory_analysis import ExploratoryAnalysis

class MLE:
    """Type-II Maximum Likelihood Estimation"""

    def __init__(self):
        pass

    def compute_log_marginal(self, PHI, y, alph, s2):
        '''Compute the log of the marginal likelihood for BLR model.'''

        def woodbury_inverse(I, U, V):
            '''Woodbury Identity'''
            return np.identity(U.shape[0]) - U @ np.linalg.inv(I + V @ U) @ V
        # beta = 1/s2
        s2 = 1 / s2
        I = np.identity(PHI.shape[1])
        U = 1 / (alph * s2) * PHI
        V = PHI.T

        const = np.log(2 * np.pi) * (len(y) / 2)
        p1 = -(np.log(np.diagonal(np.linalg.cholesky((s2 * np.identity(PHI.shape[1]) + (PHI.T @ (alph ** -1 * PHI)))))).sum() \
            + np.log(np.sqrt(s2)) * (PHI.shape[0] - PHI.shape[1]))
        p2 = - 0.5 * y.T @ ((1 / s2) * woodbury_inverse(I, U, V)) @ y
        lgp = const + p1 + p2
        return lgp

    def plot_posterior(self, X_train, y_train):
        """Plot posterior"""

        alphas = np.exp(np.linspace(-5, 0, 100))
        betas = np.exp(np.linspace(-5, 0, 100))

        log_prob_y = np.array([
            self.compute_log_marginal(X_train, y_train, alph, beta) 
            for alph in alphas 
            for beta in betas
            ]).reshape(100,100)

        idx = divmod(np.argmax(log_prob_y), 100)
        optimal_alpha = alphas[idx[0]]
        optimal_beta = betas[idx[1]]
        optimal_log = log_prob_y[idx[0]][idx[1]]

        X, Y = np.log(np.meshgrid(alphas, betas))
        fig, ax, = plt.subplots(figsize=(10,10))
        plt.contourf(X, Y, log_prob_y.T, levels=20);
        plt.scatter(np.log(optimal_alpha), np.log(optimal_beta), c='r')
        plt.annotate(r'log($\alpha)=$' + f'{np.log(optimal_alpha):.2f}\n'+ r'log($\beta)=$' + f'{np.log(optimal_beta):.2f}\n', (np.log(optimal_alpha), np.log(optimal_beta)))
        plt.xlabel(r'log($\alpha$)')
        plt.ylabel(r'log($\beta$)');

        print(f"Most probable alpha: {optimal_alpha}, log(alpha): {np.log(optimal_alpha)}")
        print(f"Most probable beta: {optimal_beta}, log(beta): {np.log(optimal_beta)}")
        print(f"Best log probability: {optimal_log}")

        self.optimal_alpha = optimal_alpha
        self.optimal_beta = optimal_beta

    def compute_posterior(self, X, y, alph, beta):
        """Compute the posterior."""

        Mu = np.linalg.inv(X.T @ X + beta * alph * np.identity(X.shape[1])) @ X.T @ y
        SIGMA = beta * np.linalg.inv(X.T @ X + beta*alph*np.identity(X.shape[1]))
        return Mu, SIGMA

    def predict(self, x_train, y_train, alph, beta, x_test):
        """Predict from the posterior."""

        w, mu = self.compute_posterior(x_train, y_train, alph, beta)
        y_pred = np.matmul(x_test, w)
        return y_pred, mu

    def get_predictions(self, X_train, y_train, X_test, y_test):
        """Get predictions for Type-II Maximum Likelihood"""

        # Train predictions
        y_pred_train, mu_train = self.predict(X_train, y_train, self.optimal_alpha, self.optimal_beta, X_train)
        # Test predictions
        y_pred_test, mu_test = self.predict(X_test, y_test, self.optimal_alpha, self.optimal_beta, X_test)

        # Calc the errors in train and test sets
        std_pred_train = np.sqrt(np.diag(X_train @ mu_train @ X_train.T))
        std_pred_test = np.sqrt(np.diag(X_test @ mu_test @ X_test.T))
        ExploratoryAnalysis.plot_pred(y_train, y_pred_train, mean_absolute_error(y_train, y_pred_train), std_pred_train, 'Bayesian Linear Regression', 'Training Set')
        ExploratoryAnalysis.plot_pred(y_test, y_pred_test, mean_absolute_error(y_test, y_pred_test), std_pred_test, 'Bayesian Linear Regression', 'Test Set')
        print(f"RMSE Train (blr): {ExploratoryAnalysis.error_rms(y_train, y_pred_train)}")
        print(f"RMSE Test (blr): {ExploratoryAnalysis.error_rms(y_test, y_pred_test)}")
        print(f"MAE Train (blr): {mean_absolute_error(y_train, y_pred_train)}")
        print(f"MAE Test (blr): {mean_absolute_error(y_test, y_pred_test)}")

    