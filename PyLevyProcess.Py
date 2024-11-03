"""
Разработанный Адриелу Ванг от ДанСтат Консульти́рования
"""

import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal, Distribution
import plotly.graph_objects as go
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.integrate import simps
from scipy.stats import levy_stable
from scipy.optimize import minimize

# Ensure reproducibility
torch.manual_seed(42)
np.random.seed(42)

class HMC_Sampler:
    def __init__(self, log_prob_fn, step_size=0.01, num_steps=10, num_samples=1000, burn_in=200):
        """
        Initialize the HMC sampler.

        Args:
            log_prob_fn (callable): Function to compute the log probability.
            step_size (float): Step size for the leapfrog integrator.
            num_steps (int): Number of leapfrog steps per iteration.
            num_samples (int): Number of samples to generate.
            burn_in (int): Number of initial samples to discard.
        """
        self.log_prob_fn = log_prob_fn
        self.step_size = step_size
        self.num_steps = num_steps
        self.num_samples = num_samples
        self.burn_in = burn_in

    def sample(self, initial_position):
        """
        Generate samples using HMC.

        Args:
            initial_position (torch.Tensor): Starting point for the sampler.

        Returns:
            samples (torch.Tensor): Generated samples after burn-in.
        """
        position = initial_position.clone().detach().requires_grad_(True)
        samples = []

        for i in range(self.num_samples + self.burn_in):
            # Sample random momentum
            momentum = torch.randn_like(position)

            # Save the current state
            current_position = position.clone().detach()
            current_momentum = momentum.clone().detach()

            # Compute current potential and kinetic energy
            current_U = -self.log_prob_fn(position)
            current_K = torch.sum(momentum ** 2) / 2

            # Leapfrog integration
            # Half step for momentum
            momentum = momentum - 0.5 * self.step_size * torch.autograd.grad(current_U, position, retain_graph=True)[0]

            # Alternate full steps
            for _ in range(self.num_steps):
                # Full step for position
                position = position + self.step_size * momentum
                position = position.clone().detach().requires_grad_(True)

                # Compute potential energy at new position
                U = -self.log_prob_fn(position)

                if _ != self.num_steps - 1:
                    # Full step for momentum, except at the end of trajectory
                    momentum = momentum - self.step_size * torch.autograd.grad(U, position, retain_graph=True)[0]

            # Half step for momentum at the end
            U = -self.log_prob_fn(position)
            momentum = momentum - 0.5 * self.step_size * torch.autograd.grad(U, position, retain_graph=True)[0]

            # Negate momentum for symmetry
            momentum = -momentum

            # Compute new potential and kinetic energy
            new_U = -self.log_prob_fn(position)
            new_K = torch.sum(momentum ** 2) / 2

            # Metropolis acceptance step
            acceptance_prob = torch.exp((current_U + current_K) - (new_U + new_K))
            if torch.rand(1) < acceptance_prob:
                # Accept the move
                pass
            else:
                # Reject the move, revert to current position
                position = current_position.clone().detach().requires_grad_(True)

            if i >= self.burn_in:
                samples.append(position.clone().detach())

        return torch.stack(samples)


def mape(y_true, y_pred):
    """
    Calculate Mean Absolute Percentage Error.

    Args:
        y_true (array-like): True values.
        y_pred (array-like): Predicted values.

    Returns:
        float: MAPE value.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero
    non_zero = y_true != 0
    return np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100


class StochasticPriceModel:
    def __init__(self, liquid_data, illiquid_data=pd.Series([], dtype=float)):
        self.liquid = liquid_data
        self.liquid_returns = torch.log(torch.tensor(self.liquid.values, dtype=torch.float)) - torch.log(torch.tensor(self.liquid.shift(1).values, dtype=torch.float))
        self.liquid_returns = self.liquid_returns[1:]  # Remove NaN

        if not illiquid_data.empty:
            self.illiquid = illiquid_data
            self.illiquid_returns = torch.log(torch.tensor(self.illiquid.values, dtype=torch.float)) - torch.log(torch.tensor(self.illiquid.shift(1).values, dtype=torch.float))
            self.illiquid_returns = self.illiquid_returns[1:]  # Remove NaN

        self.lower_confidence = None
        self.median_confidence = None
        self.upper_confidence = None
        self.selectedBest = None  # Now holds stable parameters
        self.backtestMode = False
        self.backtestMAPE = 0
        self.test = None
        self.train = None
        self.assetCorrelation = None
        self.lastAssetType = None

    def fit_ecf_stable(self, train_data, num_thetas=100, thetas=None):
        """
        Fit the Lévy stable distribution parameters using the Empirical Characteristic Function (ECF) method.

        Args:
            train_data (numpy.ndarray): Training data.
            num_thetas (int): Number of theta points for ECF.
            thetas (numpy.ndarray, optional): Specific theta points. If None, uniformly spaced.

        Returns:
            dict: Estimated stable parameters {'alpha': ..., 'beta': ..., 'gamma': ..., 'delta': ...}
        """
        # Define theta points
        if thetas is None:
            thetas = np.linspace(0.1, 10, num_thetas)

        # Empirical Characteristic Function
        ecf = np.mean(np.exp(1j * np.outer(thetas, train_data)))

        # Define the theoretical characteristic function of stable distribution
        def theoretical_cf(params, t):
            alpha, beta, gamma, delta = params
            if alpha == 1:
                term = -gamma * np.abs(t) * (1 + 1j * beta * (2 / np.pi) * np.sign(t) * np.log(np.abs(t)))
            else:
                term = -gamma**alpha * (np.abs(t)**alpha) * (1 - 1j * beta * np.sign(t) * np.tan(np.pi * alpha / 2))
            return np.exp(term + 1j * delta * t)

        # Define loss function: integrated squared difference between ECF and theoretical CF
        def loss_fn(params):
            alpha, beta, gamma, delta = params
            # Constraints
            if not (0 < alpha <= 2):
                return np.inf
            if not (-1 <= beta <= 1):
                return np.inf
            if gamma <= 0:
                return np.inf
            # Compute theoretical CF
            theor_cf = theoretical_cf(params, thetas)
            # Compute integrated squared difference
            diff = ecf - theor_cf
            loss = np.sum(np.abs(diff)**2)
            return loss

        # Optimize parameters
        initial_guess = [1.5, 0.0, 1.0, 0.0]  # [alpha, beta, gamma, delta]
        bounds = [(0.1, 2), (-1, 1), (1e-6, None), (None, None)]

        result = minimize(loss_fn, initial_guess, method='L-BFGS-B', bounds=bounds)

        if result.success:
            estimated_params = result.x
            print(f"ECF Estimated Parameters: alpha={estimated_params[0]}, beta={estimated_params[1]}, gamma={estimated_params[2]}, delta={estimated_params[3]}")
            return {
                'alpha': estimated_params[0],
                'beta': estimated_params[1],
                'gamma': estimated_params[2],
                'delta': estimated_params[3]
            }
        else:
            raise RuntimeError("ECF parameter estimation failed.")

    def log_posterior(self, params, thetas, ecf_data):
        """
        Compute the log posterior for the parameters given the data using the ECF method.

        Args:
            params (torch.Tensor): Parameters (alpha, beta, gamma, delta).
            thetas (torch.Tensor): Theta points.
            ecf_data (torch.Tensor): Empirical characteristic function values.

        Returns:
            torch.Tensor: Log posterior value.
        """
        alpha, beta, gamma, delta = params

        # Parameter constraints handled via penalties
        violation = ((alpha <= 0) | (alpha > 2) | (beta < -1) | (beta > 1) | (gamma <= 0)).float()
        penalty = violation * 1e10  # High penalty for violations

        # Theoretical characteristic function
        # To handle differentiable operations, we use torch.where
        pi = torch.tensor(np.pi, dtype=torch.float)

        # Avoid division by zero or log(0)
        eps = 1e-8
        thetas = torch.clamp(thetas, min=eps)

        # Compute terms based on alpha
        if alpha == 1.0:
            # Handle alpha = 1 separately
            term_real = -gamma * torch.abs(thetas)
            term_imag = -gamma * torch.abs(thetas) * (beta * (2 / pi)) * torch.sign(thetas) * torch.log(thetas)
        else:
            term_real = -gamma**alpha * torch.abs(thetas)**alpha * (1)
            term_imag = -gamma**alpha * torch.abs(thetas)**alpha * (beta * torch.sign(thetas) * torch.tan(pi * alpha / 2))

        # Theoretical CF
        phi_real = torch.exp(term_real) * torch.cos(term_imag)
        phi_imag = torch.exp(term_real) * torch.sin(term_imag) + delta * thetas

        # Empirical CF
        ecf_real = torch.real(ecf_data)
        ecf_imag = torch.imag(ecf_data)

        # Compute loss as squared differences
        loss_real = (ecf_real - phi_real)**2
        loss_imag = (ecf_imag - phi_imag)**2

        # Total loss
        loss = torch.sum(loss_real + loss_imag) + torch.sum(penalty)

        # Log posterior is negative loss
        return -loss

    def sample_parameters_hmc(self, thetas, ecf_data, num_samples=1000, step_size=0.01, num_steps=10, burn_in=200):
        """
        Sample parameters using HMC based on ECF.

        Args:
            thetas (torch.Tensor): Theta points.
            ecf_data (torch.Tensor): Empirical characteristic function values.
            num_samples (int): Number of samples to generate.
            step_size (float): Step size for HMC.
            num_steps (int): Number of leapfrog steps.
            burn_in (int): Number of initial samples to discard.

        Returns:
            torch.Tensor: Sampled parameters.
        """
        def log_prob_fn(params):
            return self.log_posterior(params, thetas, ecf_data)

        sampler = HMC_Sampler(log_prob_fn, step_size=step_size, num_steps=num_steps, num_samples=num_samples, burn_in=burn_in)
        # Initialize parameters close to ECF estimates or use reasonable defaults
        initial_position = torch.tensor([1.5, 0.0, 1.0, 0.0], dtype=torch.float, requires_grad=True)
        samples = sampler.sample(initial_position)
        return samples

    def compute_ecf(self, data, thetas):
        """
        Compute the empirical characteristic function for the data.

        Args:
            data (numpy.ndarray): Training data.
            thetas (numpy.ndarray): Theta points.

        Returns:
            numpy.ndarray: ECF values.
        """
        ecf = np.mean(np.exp(1j * np.outer(thetas, data)))
        return ecf

    def simulate_levy_process(self, alpha_samples, beta_samples, gamma_samples, delta_samples, So, T, N):
        """
        Simulate price paths using sampled Lévy stable parameters.

        Args:
            alpha_samples (torch.Tensor): Sampled alpha parameters.
            beta_samples (torch.Tensor): Sampled beta parameters.
            gamma_samples (torch.Tensor): Sampled gamma parameters.
            delta_samples (torch.Tensor): Sampled delta parameters.
            So (float): Initial price.
            T (int): Time horizon.
            N (int): Number of simulation paths.

        Returns:
            pd.DataFrame: Simulated price paths.
        """
        prices = np.zeros((T, N))
        prices[0, :] = So

        for t in range(1, T):
            # Randomly select parameters for each path
            indices = torch.randint(0, len(alpha_samples), (N,))
            alphas = alpha_samples[indices].numpy()
            betas = beta_samples[indices].numpy()
            gammas = gamma_samples[indices].numpy()
            deltas = delta_samples[indices].numpy()

            # Generate returns from Lévy stable distribution
            returns = levy_stable.rvs(alpha=alphas, beta=betas, loc=deltas, scale=gammas, size=N)

            # Update prices
            prices[t, :] = prices[t - 1, :] * np.exp(returns)

        return pd.DataFrame(prices)

    def illiquidModel(self, backtesting=False, horizon=10, timeout=120, N=1000, train_size=0.8):
        self.lastAssetType = "illiquid"
        if not backtesting:
            self.backtestMode = False
            training = self.illiquid_returns
            liqtrain = self.liquid_returns
            horizon = horizon
        else:
            self.backtestMode = True
            train, test = train_test_split(self.illiquid, train_size=train_size)
            training, testing = train_test_split(self.illiquid_returns, train_size=train_size)
            liqtrain, liqtest = train_test_split(self.liquid_returns, train_size=0.8)
            self.test = test
            self.train = train
            horizon = len(test)

        # Fit stable distribution using ECF
        training_np = training.numpy()
        ecf_estimates = self.fit_ecf_stable(training_np)

        # Store selectedBest as stable parameters
        self.selectedBest = ecf_estimates

        # Compute Pearson correlation using NumPy
        liqtrain_np = liqtrain.numpy()
        training_np = training.numpy()
        diagnosticcorrelation = np.corrcoef(liqtrain_np, training_np)[0, 1]
        print(f"Pearson correlation between liquid and illiquid asset is {diagnosticcorrelation}")
        self.assetCorrelation = diagnosticcorrelation

        # Compute empirical characteristic function
        thetas = np.linspace(0.1, 10, 100)
        ecf_values = self.compute_ecf(training_np, thetas)
        thetas_tensor = torch.tensor(thetas, dtype=torch.float)
        ecf_tensor = torch.tensor(ecf_values, dtype=torch.complex64)

        # Convert to PyTorch tensors
        thetas_tensor = thetas_tensor.requires_grad_(False)
        ecf_tensor = ecf_tensor.requires_grad_(False)

        # Sample parameters using HMC
        mu_sigma_samples = self.sample_parameters_hmc(thetas_tensor, ecf_tensor, num_samples=5000, step_size=0.005, num_steps=20, burn_in=1000)
        alpha_samples = mu_sigma_samples[:, 0]
        beta_samples = mu_sigma_samples[:, 1]
        gamma_samples = mu_sigma_samples[:, 2]
        delta_samples = mu_sigma_samples[:, 3]

        # Simulate price paths using sampled parameters
        prices = self.simulate_levy_process(alpha_samples, beta_samples, gamma_samples, delta_samples, So=self.illiquid.iloc[-1], T=horizon, N=N)

        # Calculate confidence intervals
        self.lower_confidence = prices.quantile(0.05, axis=1)
        self.median_confidence = prices.quantile(0.5, axis=1)
        self.upper_confidence = prices.quantile(0.95, axis=1)

    def liquidModel(self, backtesting=False, horizon=10, timeout=120, N=1000, train_size=0.8):
        self.lastAssetType = "liquid"
        if not backtesting:
            self.backtestMode = False
            training = self.liquid_returns
            horizon = horizon
        else:
            self.backtestMode = True
            train, test = train_test_split(self.liquid, train_size=train_size)
            training, testing = train_test_split(self.liquid_returns, train_size=train_size)
            self.test = test
            self.train = train
            horizon = len(test)

        # Fit stable distribution using ECF
        training_np = training.numpy()
        ecf_estimates = self.fit_ecf_stable(training_np)

        # Store selectedBest as stable parameters
        self.selectedBest = ecf_estimates

        # Compute empirical characteristic function
        thetas = np.linspace(0.1, 10, 100)
        ecf_values = self.compute_ecf(training_np, thetas)
        thetas_tensor = torch.tensor(thetas, dtype=torch.float)
        ecf_tensor = torch.tensor(ecf_values, dtype=torch.complex64)

        # Convert to PyTorch tensors
        thetas_tensor = thetas_tensor.requires_grad_(False)
        ecf_tensor = ecf_tensor.requires_grad_(False)

        # Sample parameters using HMC (assuming no correlation for liquid assets)
        mu_sigma_samples = self.sample_parameters_hmc(thetas_tensor, ecf_tensor, num_samples=5000, step_size=0.005, num_steps=20, burn_in=1000)
        alpha_samples = mu_sigma_samples[:, 0]
        beta_samples = mu_sigma_samples[:, 1]
        gamma_samples = mu_sigma_samples[:, 2]
        delta_samples = mu_sigma_samples[:, 3]

        # Simulate price paths using sampled parameters
        prices = self.simulate_levy_process(alpha_samples, beta_samples, gamma_samples, delta_samples, So=self.liquid.iloc[-1], T=horizon, N=N)

        # Calculate confidence intervals
        self.lower_confidence = prices.quantile(0.05, axis=1)
        self.median_confidence = prices.quantile(0.5, axis=1)
        self.upper_confidence = prices.quantile(0.95, axis=1)

    def plotEstimates(self, assetType="liquid"):
        if self.backtestMode:
            axis = np.arange(len(self.train) + len(self.test))
            plt.plot(axis[:len(self.train)], self.train, c="blue", label="Training Data")
            plt.plot(axis[len(self.train):], self.test, c="green", label="Test Data")
            plt.plot(axis[len(self.train):], self.median_confidence, c="red", label="Median Prediction")
            plt.fill_between(axis[len(self.train):], self.lower_confidence, self.upper_confidence, color="red", alpha=0.2, label="5th-95th Percentile")
            plt.legend()
            plt.show()

            self.backtestMAPE = mape(self.median_confidence, self.test)
            print(f"Mean Absolute Percentage Error: {self.backtestMAPE}")

        else:
            if assetType == "liquid" or self.lastAssetType == "liquid":
                train = self.liquid
            elif assetType == "illiquid" or self.lastAssetType == "illiquid":
                train = self.illiquid

            axis = np.arange(len(train) + len(self.median_confidence))
            plt.plot(axis[:len(train)], train, c="blue", label="Training Data")
            plt.plot(axis[len(train):], self.median_confidence, c="red", label="Median Prediction")
            plt.fill_between(axis[len(train):], self.lower_confidence, self.upper_confidence, color="red", alpha=0.2, label="5th-95th Percentile")
            plt.legend()
            plt.show()

    def generateEstimateFigure(self, assetType="liquid"):
        fig = go.Figure()

        if self.backtestMode:
            train = self.train
            test = self.test
        else:
            if assetType == "liquid" or self.lastAssetType == "liquid":
                train = self.liquid
            elif assetType == "illiquid" or self.lastAssetType == "illiquid":
                train = self.illiquid

        median_confidence = self.median_confidence
        lower_confidence = self.lower_confidence
        upper_confidence = self.upper_confidence

        # Plot the training data
        fig.add_trace(go.Scatter(x=list(range(len(train))), y=train, mode='lines', name='Training Data', line=dict(color='blue')))

        if self.backtestMode:
            # Plot the test data
            fig.add_trace(go.Scatter(x=list(range(len(train), len(train) + len(test))), y=test, mode='lines', name='Test Data', line=dict(color='green')))
            self.backtestMAPE = mape(self.median_confidence, self.test)
            print(f"Mean Absolute Percentage Error: {self.backtestMAPE}")

        # Plot the median confidence
        fig.add_trace(go.Scatter(x=list(range(len(train), len(train) + len(median_confidence))), y=median_confidence, mode='lines', name='Median Prediction', line=dict(color='red')))

        # Plot the confidence intervals
        fig.add_trace(go.Scatter(
            x=list(range(len(train), len(train) + len(lower_confidence))),
            y=lower_confidence,
            mode='lines',
            name='Lower Confidence (5th percentile)',
            line=dict(color='red', width=0),
            showlegend=False
        ))
        fig.add_trace(go.Scatter(
            x=list(range(len(train), len(train) + len(upper_confidence))),
            y=upper_confidence,
            mode='lines',
            name='Upper Confidence (95th percentile)',
            line=dict(color='red', width=0),
            showlegend=False,
            fill='tonexty'
        ))

        # Update layout
        fig.update_layout(title='Price Estimates with Confidence Intervals', xaxis_title='Time', yaxis_title='Price')

        return fig
