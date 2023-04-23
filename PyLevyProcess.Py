import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy
from fitter import Fitter
import plotly.graph_objects as go

class PyLevyProcess:
    def __init__(self, liquid_data, illiquid_data = pd.Series([], dtype = float)):
        self.liquid = liquid_data
        self.liquid_returns = np.log(self.liquid) - np.log(self.liquid.shift(1))
        if not illiquid_data.empty:
            self.illiquid = illiquid_data
            self.illiquid_returns = np.log(self.illiquid) - np.log(self.illiquid.shift(1))
        self.lower_confidence = None
        self.median_confidence = None
        self.upper_confidence = None
        self.selectedBest = None
        self.backtestMode = False
        self.backtestMAPE = 0
        self.test = None
        self.train = None
        self.assetCorrelation = None
        self.lastAssetType = None

    def fit_distribution(self, train_data, timeout, distributions = None):
        f = Fitter(train_data, timeout=timeout, distributions=distributions)
        f.fit()
        f.summary()
        selected_best = f.get_best(method="bic")
        self.selectedBest = selected_best
        #return selected_best

    def simulate_gbm(self, So, mu, sigma, T, N, distChosen, distParams, diagnosticcorrelation, illiquid = False):
        distribution = getattr(scipy.stats, distChosen)

        def GBMsimulatorUniVar(So, mu, sigma, T, N):
            S = np.zeros([T + 1, int(N)])
            S[0, :] = So
            for t in range(1, int(T) + 1):
                newprices = []
                for i in range(0, int(N)):
                    drift = mu - 0.5 * sigma**2
                    # I forgot whether I used Ito's Lemma to derive this drift.
                    # When making the function, I did use Ito's Lemma with the Ang et al model, but I forgot whether I accidentally overwrote it.
                    Z1 = distribution.rvs(*distParams)
                    if illiquid == True:
                        Z2 = distribution.rvs(*distParams)
                    else:
                        Z2 = 0

                    diffusion = sigma*Z1*diagnosticcorrelation + sigma*np.sqrt(1 - diagnosticcorrelation**2) * Z2
                    S[t][i] = S[t - 1][i]*np.exp(drift + diffusion)
            return S[1:]

        prices = GBMsimulatorUniVar(So, mu, sigma, T, N)
        return prices

    def illiquidModel(self, backtesting = False, horizon = 10, timeout = 120, N = 1000, train_size = 0.8):
        self.lastAssetType = "illiquid"
        if backtesting == False:
            self.backtestMode = False
            training = self.illiquid_returns.dropna()
            liqtrain = self.liquid_returns
            horizon = horizon
        else:
            self.backtestMode = True
            train, test = train_test_split(self.illiquid, train_size=train_size)
            training, testing = train_test_split(self.illiquid_returns.dropna(), train_size=train_size)
            liqtrain, liqtest = train_test_split(self.liquid_returns, train_size=0.8)
            self.test = test
            self.train = train
            horizon = len(test)

        if self.selectedBest !=None:
            pass
        else:
            self.fit_distribution(training, timeout = timeout)

        distChosen = list(self.selectedBest.keys())[0]
        distParams = tuple(self.selectedBest[distChosen].values())

        # Model used taken from Ang, A., Papanikolaou, D., & Westerfield, M. M. (2014).
        diagnosticcorrelation = pd.DataFrame([liqtrain, training]).transpose().dropna().corr(method="pearson").iloc[0].iloc[1]
        print("Pearson correlation between liquid and illiquid asset is {}".format(diagnosticcorrelation))
        self.assetCorrelation = diagnosticcorrelation

        prices = self.simulate_gbm(So=self.illiquid.iloc[-1], mu=training.mean(), sigma=training.std(), T=horizon, N=N, distChosen=distChosen, distParams=distParams, diagnosticcorrelation=diagnosticcorrelation)
        prices = pd.DataFrame(prices)
        #self.prices = pd.DataFrame(prices).mean(axis=1)
        self.lower_confidence = prices.quantile(0.05, axis=1)
        self.median_confidence = prices.quantile(0.5, axis=1)
        self.upper_confidence = prices.quantile(0.95, axis=1)

    def liquidModel(self, backtesting = False, horizon = 10, timeout = 120, N = 1000, train_size = 0.8):
        self.lastAssetType = "liquid"
        if backtesting == False:
            self.backtestMode = False
            training = self.liquid_returns
            horizon = horizon
        else:
            self.backtestMode = True
            train, test = train_test_split(self.liquid, train_size=train_size)
            training, testing = train_test_split(self.liquid_returns.dropna(), train_size=train_size)
            self.test = test
            self.train = train
            horizon = len(test)

        if self.selectedBest !=None:
            pass
        else:
            self.fit_distribution(training, timeout = timeout)

        distChosen = list(self.selectedBest.keys())[0]
        distParams = tuple(self.selectedBest[distChosen].values())

        prices = self.simulate_gbm(So=self.liquid.iloc[-1], mu=training.mean(), sigma=training.std(), T=horizon, N=N, distChosen=distChosen, distParams=distParams, diagnosticcorrelation=1)
        prices = pd.DataFrame(prices)
        #self.prices = pd.DataFrame(prices).mean(axis=1)
        self.lower_confidence = prices.quantile(0.05, axis=1)
        self.median_confidence = prices.quantile(0.5, axis=1)
        self.upper_confidence = prices.quantile(0.95, axis=1)

    def plotEstimates(self, assetType = "liquid"):
        if self.backtestMode == True:
            axis = np.arange(len(self.train) + len(self.test))
            plt.plot(axis[:len(self.train)], self.train, c="blue", label="Training Data")
            plt.plot(axis[len(self.train):], self.test, c="green", label="Test Data")
            plt.plot(axis[len(self.train):], np.array(self.median_confidence), c="red", label="Median Prediction")
            plt.fill_between(axis[len(self.train):], self.lower_confidence, self.upper_confidence, color="red", alpha=0.2, label="5th-95th Percentile")
            plt.legend()
            plt.show()

            self.backtestMAPE = mape(self.median_confidence, self.test)
            print("Mean Absolute Percentage Error: {}".format(self.backtestMAPE))

        else:
            if assetType == "liquid" or self.lastAssetType == "liquid":
                train = self.liquid
            elif assetType == "illiquid" or self.lastAssetType == "illiquid":
                train = self.illiquid

            axis = np.arange(len(train) + len(self.median_confidence))
            plt.plot(axis[:len(train)], train, c="blue", label="Training Data")
            plt.plot(axis[len(train):], np.array(self.median_confidence), c="red", label="Median Prediction")
            plt.fill_between(axis[len(train):], self.lower_confidence, self.upper_confidence, color="red", alpha=0.2, label="5th-95th Percentile")
            plt.legend()
            plt.show()

    def generateEstimateFigure(self, assetType = "liquid"):
        fig = go.Figure()

        if self.backtestMode == True:
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

        if self.backtestMode == True:
            # Plot the test data
            fig.add_trace(go.Scatter(x=list(range(len(train), len(train) + len(test))), y=test, mode='lines', name='Test Data', line=dict(color='green')))
            self.backtestMAPE = mape(self.median_confidence, self.test)
            print("Mean Absolute Percentage Error: {}".format(self.backtestMAPE))

        # Plot the median confidence
        fig.add_trace(go.Scatter(x=list(range(len(train), len(train) + len(median_confidence))), y=median_confidence, mode='lines', name='Median Prediction', line=dict(color='red')))

        # Plot the confidence intervals
        fig.add_trace(go.Scatter(x=list(range(len(train), len(train) + len(lower_confidence))), y=lower_confidence, mode='lines', name='Lower Confidence (5th percentile)', line=dict(color='red', width=0), showlegend=False))
        fig.add_trace(go.Scatter(x=list(range(len(train), len(train) + len(upper_confidence))), y=upper_confidence, mode='lines', name='Upper Confidence (95th percentile)', line=dict(color='red', width=0), showlegend=False, fill='tonexty'))

        # Update layout
        fig.update_layout(title='Price Estimates with Confidence Intervals', xaxis_title='Time', yaxis_title='Price')

        return fig
