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

from exploratory_analysis import ExploratoryAnalysis
from mle import MLE
from vi import VariationalInference
from hmc import Hamiltonian
from gp import GaussianProcess, BNN

def import_data(path_train, path_test):
    train = pd.read_csv(path_train)
    test = pd.read_csv(path_test)

    # Apply normalization
    ss = StandardScaler()
    train_ss = pd.DataFrame(ss.fit_transform(train), columns=train.columns)
    test_ss = pd.DataFrame(ss.transform(test), columns=train.columns)
    # Reset the const col
    train_ss['const'] = 1
    test_ss['const'] = 1

    X_train = train_ss.drop('Heating Load', axis=1).values
    y_train = train['Heating Load'].values
    X_test = test_ss.drop('Heating Load', axis=1).values
    y_test = test['Heating Load'].values

    ExploratoryAnalysis.checkup(X_train, y_train)

    return X_train, y_train, X_test, y_test

def predictions():
    """Get predictions for all algorithms"""

    # Import and Scale the Data
    path_train = "data//ee-train.csv"
    path_test = "data//ee-test.csv"
    X_train, y_train, X_test, y_test = import_data(path_train, path_test)

    # Ordinary Least Squares estimation
    ExploratoryAnalysis.ols(X_train, y_train, X_test, y_test)

    # Random Forest regressor
    ExploratoryAnalysis.rf(X_train, y_train, X_test, y_test)

    # Maximum-Likelihood Estimation
    MLE.get_predictions(X_train, y_train, X_test, y_test)

    # Variational Inference
    VariationalInference.get_predictions(X_train, y_train, X_test, y_test)

    # Hamiltonian Markov Chains
    Hamiltonian.get_predictions(X_train, y_train, X_test, y_test)

    # Gaussian Processes
    GaussianProcess.kernels(X_train, y_train, X_test, y_test)

    # Bayesian Neural Network
    bnn = BNN.network(X_train)
    BNN.inference(X_train, y_train, X_test, bnn)

def plot_posteriors():
    """Plot posterior distribution for all algorithms"""

    # Import and Scale the Data
    path_train = "data//ee-train.csv"
    path_test = "data//ee-test.csv"
    X_train, y_train, X_test, y_test = import_data(path_train, path_test)

    # Type-II Maximum Likelihood
    MLE.plot_posterior(X_train, y_train)

    # Variational Inference
    VariationalInference.plot_posterior(X_train, y_train)

    # Hamiltonian Markov Chains
    Hamiltonian.plot_posterior(X_train, y_train)


