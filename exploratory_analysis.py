import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%config InlineBackend.figure_format = 'retina'
import seaborn as sns
sns.set_theme('paper', style='darkgrid')
import warnings
warnings.filterwarnings("ignore")

from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from matplotlib.offsetbox import AnchoredText
from scipy.stats import pearsonr
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor


class ExploratoryAnalysis:
    def __init__(self):
        self.path_train = "ee-train.csv"
        self.path_test = "ee-test.csv"

    def error_rms(self, t, y):
        """Compute RMS error for a prediction vector"""
        err = np.sqrt(np.mean((y - t) ** 2))
        return err
    
    def plot_pred(self, y_test, y_pred, err, std_pred, title, dataset):
        """Plots the predictions and test, with the confidence interval."""
        fig, ax = plt.subplots(figsize=(20,5))
        plt.scatter(x=range(len(y_test)), y=y_test, c='k', alpha=0.7, label=dataset)
        plt.plot(y_pred, c='r', alpha=0.7, label='Predictions')
        plt.fill_between(x=range(len(y_pred)), y1=y_pred - 1.975 * std_pred, 
            y2=y_pred + 1.975 * std_pred, alpha=0.2, color='b', label='95% Confidence Interval')
        plt.title(title)
        
        at = AnchoredText(f'MAE: {err}', loc='lower right')
        ax.add_artist(at)
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y$')
        plt.legend(loc='upper left')

    def corrfunc(self, x, y, **kws):
        '''Gets the pearson corr for each graph.'''
        (r, p) = pearsonr(x, y)
        ax = plt.gca()
        ax.set_title(f"r = {r:.2f}, p = {p:.2f}")

    def exploratory_analysis(self):
        """Apply exploratory analysis"""
        self.train = pd.read_csv(self.path_train)
        train = self.train

        # Plot the correlation
        g = sns.pairplot(pd.DataFrame(train),
            height=4,
            y_vars=['Heating Load'],
            x_vars=[
                'Relative Compactness', 'Surface Area', 'Wall Area',
                'Roof Area', 'Overall Height', 'Orientation', 'Glazing Area',
                'Glazing Area Distribution'
            ])
        g.map(self.corrfunc)

    def heatmap(self):
        """Plot heatmap"""
        train = self.train
        corr = train.drop('const', axis=1).corr()
        mask = np.zeros_like(corr)
        mask[np.triu_indices_from(mask)] = True
        with sns.axes_style("white"):
            f, ax = plt.subplots(figsize=(10, 10))
            ax = sns.heatmap(corr, mask=mask, vmin=-1, vmax=1, square=True, annot=True, cbar_kws={"shrink": 0.7})

    def target_plot(self, y_train, y_test):
        """Plot the train and test set target variable"""

        fig, ax = plt.subplots(figsize=(20,5))
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_title('The Target Variable "Heating Load"')
        sns_c = sns.color_palette(palette='deep')
        sns.lineplot(x=range(384), y=y_train.flatten(), color=sns_c[0], label='Training Set', ax=ax)
        fig, ax = plt.subplots(figsize=(20,5))
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_title('The Target Variable "Heating Load"')
        sns.lineplot(x=range(384), y=y_test.flatten(), color=sns_c[1], label='Test Set', ax=ax);

    def ols(self,X_train, y_train, X_test, y_test):
        """Perform Ordinary Least Squares Estimation"""

        est = sm.OLS(y_train, X_train).fit()
        y_pred_train = est.predict(X_train)
        cov_train = est.cov_params()
        y_pred_test = est.predict(X_test)
        cov_test = est.cov_params()

        std_pred_train = np.sqrt(np.diag(X_train @ cov_train @ X_train.T))
        std_pred_test = np.sqrt(np.diag(X_test @ cov_train @ X_test.T))
        self.plot_pred(y_train, y_pred_train, mean_absolute_error(y_train, y_pred_train), std_pred_train, 'Ordinary Least Squares', 'Training Set')
        self.plot_pred(y_test, y_pred_test, mean_absolute_error(y_test, y_pred_test), std_pred_test, 'Ordinary Least Squares', 'Test Set')
        est.summary()

        print(f"RMSE Train (ols): {self.error_rms(y_train, y_pred_train)}")
        print(f"RMSE Test (ols): {self.error_rms(y_test, y_pred_test)}")
        print(f"MAE Train (ols): {mean_absolute_error(y_train, y_pred_train)}")
        print(f"MAE Test (ols): {mean_absolute_error(y_test, y_pred_test)}")

    def rf(self, X_train, y_train, X_test, y_test):
        """Random Forest Regressor"""
        clf = RandomForestRegressor()
        clf.fit(X_train, y_train)
        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_test)

        print(f"RMSE Train (rf): {self.error_rms(y_train, y_pred_train)}")
        print(f"RMSE Test (rf): {self.error_rms(y_test, y_pred_test)}")
        print(f"MAE Train (rf): {mean_absolute_error(y_train, y_pred_train)}")
        print(f"MAE Test (rf): {mean_absolute_error(y_test, y_pred_test)}")
        self.plot_pred(y_train, y_pred_train, mean_absolute_error(y_train, y_pred_train), 0.1, 'Random Forest', 'Training Set')
        self.plot_pred(y_test, y_pred_test, mean_absolute_error(y_test, y_pred_test), 0.1, 'Random Forest', 'Test Set')

    def checkup(self, X_train, y_train):
        """Check whether fixed variables are correctly defined"""
        try:
            if not isinstance(X_train,np.ndarray):
                print('X_train should be an array')        
            if X_train.shape != (384,9):
                print('X_train is arranged in wrong shape')
        except Exception as err:
            print('Error output:', err)
            
        try:
            if not isinstance(y_train,np.ndarray):
                print('y_train should be an array')        
            if y_train.shape != (384,):
                print('y_train is arranged in wrong shape')
        except Exception as err:
            print('Error output:', err)