import matplotlib.pyplot as plt
%config InlineBackend.figure_format = 'retina'
import seaborn as sns
sns.set_theme('paper', style='darkgrid')
import warnings
warnings.filterwarnings("ignore")
from matplotlib.offsetbox import AnchoredText

import numpy as np
from sklearn.metrics import mean_absolute_error
from exploratory_analysis import ExploratoryAnalysis
from scipy import stats
import hmc_Lab as hmc
from mle import MLE

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import tensorflow as tf
import tensorflow_probability as tfp

class GaussianProcess:
    """Gaussian Processes"""
    def __init__(self):
        pass

    def get_predictions(self, X_train, y_train, X_test, y_test, kernel = None):
        """Predictions from Gaussian Process algorithm"""

        # Default Kernel
        gpr = GaussianProcessRegressor(kernel = kernel)
        gpr.fit(X_train, y_train)
        acc = gpr.score(X_test, y_test)
        log_prob = gpr.log_marginal_likelihood()
        y_pred_test, std_pred_test = gpr.predict(X_test, return_std=True)
        y_pred_train, std_pred_train = gpr.predict(X_train, return_std=True)

        rmse_test = ExploratoryAnalysis.error_rms(y_test, y_pred_test)
        rmse_train = ExploratoryAnalysis.error_rms(y_train, y_pred_train)
        mae_test = mean_absolute_error(y_test, y_pred_test)
        mae_train = mean_absolute_error(y_train, y_pred_train)

        print(f"R Score: {acc}")
        print(f"RMSE Train (gp): {rmse_train}")
        print(f"RMSE Test (gp): {rmse_test}")
        print(f"MAE Train (gp): {mae_train}")
        print(f"MAE Test (gp): {mae_test}")
        print(f"Log Marginal Liklihood: {log_prob:.2f}")

        ExploratoryAnalysis.plot_pred(y_train, y_pred_train, 
                                      mae_train, std_pred_train, 
                                      f'Gaussian Process ({kernel})', 'Training Set')
        
        ExploratoryAnalysis.plot_pred(y_test, y_pred_test, 
                                      mae_test, std_pred_test, 
                                    f'Gaussian Process ({kernel} Kernel)', 'Test Set')
        
    def kernels(self, X_train, y_train, X_test, y_test):
       """Applying the algorithm with different kernels"""

       # Default kernel
       self.get_predictions(self, X_train, y_train, X_test, y_test, kernel = None)

       # Using RBF Kernel
       kernel = 1.0 * RBF(length_scale=0.5, length_scale_bounds=(1e-3, 1e2))
       self.get_predictions(self, X_train, y_train, X_test, y_test, kernel = kernel)

       # Using RBF + Noise Kernel
       kernel = 1.0 * RBF(length_scale=0.5, length_scale_bounds=(1e-3, 1e2)) + WhiteKernel(
    noise_level=1e-4, noise_level_bounds=(1e-22, 1e2))
       
       self.get_predictions(self, X_train, y_train, X_test, y_test, kernel = kernel)
       
        


# GP Bayesian Neural Network
class RBFKernelFn(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super(RBFKernelFn, self).__init__(**kwargs)
    dtype = kwargs.get('dtype', None)

    self._amplitude = self.add_weight(
            initializer=tf.constant_initializer(0),
            dtype=dtype,
            name='amplitude')

    self._length_scale = self.add_weight(
            initializer=tf.constant_initializer(0),
            dtype=dtype,
            name='length_scale')

  def call(self, x):
    # Never called -- this is just a layer so it can hold variables
    # in a way Keras understands.
    return x

  @property
  def kernel(self):
    return tfp.math.psd_kernels.ExponentiatedQuadratic(
      amplitude=tf.nn.softplus(.1 * self._amplitude),
      length_scale=tf.nn.softplus(50. * self._length_scale)
    )

class BNN:
   """Bayesian Neural Network"""
   def __init__(self):
        self.num_inducing_points = 50
        self.batch_size = 32

   def network(self, X_train):
        """Network architecture"""

        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=[9], dtype=X_train.dtype),
            tf.keras.layers.Dense(9, kernel_initializer='ones', use_bias=False),
            tfp.layers.VariationalGaussianProcess(
                num_inducing_points=self.num_inducing_points,
                kernel_provider=RBFKernelFn(dtype=X_train.dtype),
                event_shape=[1],
                inducing_index_points_initializer=tf.constant_initializer(X_train[np.random.choice(X_train.shape[0], 
                                                                                                    size=self.num_inducing_points, 
                                                                                                    replace=False),:]),
                unconstrained_observation_noise_variance_initializer=(
                    tf.constant_initializer(
                        np.log(np.expm1(1.)).astype(X_train.dtype))),
            ),
        ])

        return model

   def inference(self, X_train, y_train, X_test, model):
        """Do the inference"""

        loss = lambda y, rv_y: rv_y.variational_loss(
            y, kl_weight=np.array(self.batch_size, X_train.dtype) / X_train.shape[0])

        model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001, amsgrad=True), loss=loss)

        model.fit(X_train, y_train, batch_size=self.batch_size, epochs=100, verbose=True, validation_split=0.1)

        yhat_train = model(X_train)
        yhat_test = model(X_test)


        fig, ax = plt.subplots(figsize=(20, 5)) 
        num_samples = 100
        samples = []
        for i in range(num_samples):
            sample_ = yhat_train.sample().numpy()
            plt.plot(range(384),
                sample_[..., 0].T,
                'b',
                linewidth=0.9,
                alpha=0.05,
                zorder=1,
                label='Samples' if i == 0 else None)
            samples.append(sample_)

        plt.xlabel('x')
        plt.ylabel('y')
        plt.scatter(x=range(384), y=y_train, c='k', label='Training Set', zorder=2, alpha=0.5)
        plt.plot(np.array(samples).mean(axis=0), label='Avg Samples', color='r', zorder=3)

        mae_err = mean_absolute_error(y_train, np.array(samples).mean(axis=0))
        rmse_err = ExploratoryAnalysis.error_rms(y_train, np.array(samples).mean(axis=0))
        print(f'MAE Train: {mae_err}')
        print(f'RMSE Train: {rmse_err}')

        at = AnchoredText(f'MAE: {mae_err}', loc='lower right')
        ax.add_artist(at)
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y$')
        plt.legend(loc='upper left')
        plt.title(f'Gaussian Process Bayesian Neural Network with {num_samples} samples');