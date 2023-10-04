### Overview

This work involves the evaluation of Bayesian modelling methods on a real multivariate regression task. The guiding objectives are to derive a good predictor for data derived from an “energy efficiency” data set, and to estimate which of the input variables are relevant for prediction. In particular, the exercise focuses on approximating (and averaging over) posterior distributions using the Hamiltonian Monte Carlo (HMC), Variational Inference (VI), and Gaussian Processes (GP) methods.

### Data

You will be analysing the “Energy efficiency” data set, originally from the University of Oxford, and now made available at the UCI Machine Learning Repository. This multivariate data set contains 768 examples and comprises 1 constant bias and 8 input variables, where the first is the constant bias and others presenting some basic architectural parameters for buildings (e.g. “Roof Area” and “Glazing Area”) with the intention of predicting a tenth target variable, the required “Heating Load”. This can be considered a real-value variable, suitable for standard regression modelling.