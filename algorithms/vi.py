import matplotlib.pyplot as plt
%config InlineBackend.figure_format = 'retina'
import seaborn as sns
sns.set_theme('paper', style='darkgrid')
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.metrics import mean_absolute_error
from algorithms.exploratory_analysis import ExploratoryAnalysis
from algorithms.mle import MLE

class VariationalInference:
    """Variational Inference algorithm"""
    
    def __init__(self):
        pass

    def VI(self, X_train, y_train):
        """Variational Inference algorithm"""

        a0, b0, c0, d0 = [1e-4] * 4
        alpha0 = a0 / b0
        beta0 = c0 / d0
        mu_n, sig_n = MLE.compute_posterior(X_train, y_train, alpha0, beta0)
        an = a0 + 9/2
        cn = c0 + 384/2
        bn,dn = b0,d0
        
        for i in range(len(X_train)):
            bn = b0 + 1/2 * (mu_n.T @ mu_n + np.trace(sig_n))
            dn = d0 + 1/2 * ((y_train - X_train @ mu_n).T @ (y_train - X_train @ mu_n))
            alphaN, betaN = an/bn, cn/dn
            mu_n, sig_n = MLE.compute_posterior(X_train, y_train, alphaN, betaN)
        
        return an, bn, cn, dn, mu_n, sig_n

    def get_predictions(self, X_train, y_train, X_test, y_test):
        """Get predictions for Variational Inference"""
        
        _, _, _, _, mu_n, sig_n = self.VI(X_train, y_train)

        self.mu_n = mu_n
        self.sig_n = sig_n

        self.rmse_train = ExploratoryAnalysis.error_rms(y_train, (X_train @ mu_n))
        self.rmse_test = ExploratoryAnalysis.error_rms(y_test, (X_test @ mu_n))
        self.mae_train = mean_absolute_error(y_train, (X_train @ mu_n))
        self.mae_test = mean_absolute_error(y_test, (X_test @ mu_n))
        # Get predictions
        y_pred_train = np.dot(X_train, mu_n)
        y_pred_test = np.dot(X_test, mu_n)
        # Standard deviation in predications
        std_pred_train = np.sqrt(np.diag(X_train @ sig_n @ X_train.T))
        std_pred_test = np.sqrt(np.diag(X_test @ sig_n @ X_test.T))
        ExploratoryAnalysis.plot_pred(y_train, y_pred_train, self.mae_train, std_pred_train, 'Variational Inference', 'Training Set')
        ExploratoryAnalysis.plot_pred(y_test, y_pred_test, self.mae_test, std_pred_test, 'Variational Inference', 'Test Set')

    def plot_posterior(self, X_train, y_train, optimal_alpha, optimal_beta):
        """Plot posterior"""

        alphas = np.exp(np.linspace(-5, 0, 100))
        betas = np.exp(np.linspace(-5, 0, 100))

        log_prob_y = np.array([
            MLE.compute_log_marginal(X_train, y_train, alph, beta) 
            for alph in alphas 
            for beta in betas
            ]).reshape(100,100)

        X, Y = np.log(np.meshgrid(alphas, betas))
        fig, ax, = plt.subplots(figsize=(10,10))
        plt.contourf(X, Y, log_prob_y.T, levels=20);
        plt.scatter(np.log(optimal_alpha), np.log(optimal_beta), c='r')
        plt.annotate(r'log($\alpha)=$' + f'{np.log(optimal_alpha):.2f}\n'+ r'log($\beta)=$' + f'{np.log(optimal_beta):.2f}\n', (np.log(optimal_alpha), np.log(optimal_beta)))
        plt.xlabel(r'log($\alpha$)')
        plt.ylabel(r'log($\beta$)');

        print(f"Most probable alpha: {optimal_alpha}, log(alpha): {np.log(optimal_alpha)}")
        print(f"Most probable beta: {optimal_beta}, log(beta): {np.log(optimal_beta)}")

    def expectations(self, X_train, y_train):
        """Print out the expectation of alpha and beta"""

        an, bn, cn, dn, _, _ = self.VI(X_train, y_train)
        exp_alpha = an / bn
        exp_beta = cn / dn

        self.exp_alpha = exp_alpha
        self.exp_beta = exp_beta

        self.plot_posterior(X_train, y_train, self.exp_alpha, self.exp_beta)

        print(f"Expected alpha: {exp_alpha}")
        print(f"Expected beta: {exp_beta}")
        print(f"Expected log(alpha): {np.log(exp_alpha)}")
        print(f"Expected log(beta): {np.log(exp_beta)}")
        print(f"Best log probability: {MLE.compute_log_marginal(X_train, y_train, exp_alpha, exp_beta)}")


        print(f"Train RMSE (vi): {self.rmse_train}")
        print(f"Test RMSE (vi): {self.rmse_test}")
        print(f"Train MAE (vi): {self.mae_train}")
        print(f"Test MAE (vi): {self.mae_test}")

    def checkup(self, mu_n, sig_n):
        """Check whether your variables are correctly defined"""

        try:
            if not isinstance(mu_n, np.ndarray):
                print('mu_n should be an array')        
            if mu_n.shape != (9, ):
                print('mu_n is arranged in wrong shape')
        except Exception as err:
            print('Error output:', err)
            
        try:
            if not isinstance(sig_n,np.ndarray):
                print('sig_n should be an array')        
            if sig_n.shape != (9,9):
                print('sig_n is arranged in wrong shape')
        except Exception as err:
            print('Error output:', err)
            
        try:
            self.exp_alpha
            self.exp_beta
        except Exception as err:
            print('Error output:', err)