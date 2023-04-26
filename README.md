# PyLevyProcess
Standard treatments of financial returns in Economic literature model them with a normal distribution, leading to Geometric Brownian Motion as a standard model. However, Mandelbrot and others have found that financial returns are better modelled as a generalized Levy process than with Normally distributed processes.

A **Lévy process** $X(t)$ is a stochastic process that has the following properties:
- $X(0) = 0$ almost surely.
- The process has independent increments, i.e., for any $0 \leq s < t$, the random variables $X(t) - X(s)$ are independent.
- The process has stationary increments, i.e., the distribution of $X(t) - X(s)$ depends only on $t - s$.
- The process has continuous paths, i.e., the function $t \mapsto X(t)$ is continuous almost surely.
 
 Lévy processes are a general class of stochastic processes that include Brownian motion (Wiener process), Poisson processes, and many other stochastic processes used in finance, physics, and other fields. `PyLevyProcess` automates work for modelling returns as Levy processes. Note that you will have to do a serial correlation test to verify whether increments are independent. You may use a function like, for example, `statsmodels.stats.stattools.durbin_watson`.

## To-do

- Replace direct Monte-Carlo method with Hamiltonian Monte-Carlo to ensure independent and stationary simulated increments.

# Examples
## Preliminaries
Import the class.
```
from PyLevyProcess import *
```
Use these functions along with the class itself, or use appropriate functions from other packages.
```
def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))
    
def train_test_split(data, train_size):
    split_index = int(len(data) * train_size)
    train = data[:split_index]
    test = data[split_index:]
    return train, test
```
It is also recommended to set a random seed for reproducibility.
Use this function to fetch data:
```
def get_data(symbols, timeframe, start_date, end_date=None):
    exchange = ccxt.cryptocom()
    #since = exchange.parse8601(start_date)

    if end_date != None:
        end_date_unix = exchange.parse8601(end_date)

    closing_prices = pd.DataFrame()

    for symbol in symbols:
        since = exchange.parse8601(start_date)
        data = []
        while True:
            candles = exchange.fetch_ohlcv(symbol, timeframe, since)
            if not candles:
                break
            data.extend(candles)
            since = candles[-1][0] + 1

            if end_date != None and since >= end_date_unix:
                break

        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        closing_prices[symbol] = df['close']

    return closing_prices
```
For data already fetched by the above function, use this function to update it.
```
def update_data(symbol, timeframe, csv_file):
    # Read the existing CSV file into a DataFrame
    existing_data = pd.read_csv(csv_file, index_col='timestamp', parse_dates=True)

    # Get the last timestamp in the DataFrame
    last_timestamp = existing_data.index[-1]

    # Convert the last timestamp to ISO 8601 format
    start_date = last_timestamp.strftime('%Y-%m-%dT%H:%M:%SZ')

    # Fetch new data from the exchange starting after the last timestamp
    new_data = get_data(symbol, timeframe, start_date)

    # Append the new data to the existing DataFrame
    updated_data = existing_data.append(new_data)

    # Save the updated DataFrame to the CSV file
    updated_data.to_csv(csv_file)
```
We use cryptocurrency as an example, but the Levy Process simulator can be used for any asset class.
```
# Liquid and Illiquid Asset:
symbols = ['ETH/USD', 'BIFI/USD']
timeframe = '1d'
start_date = '2022-01-01T00:00:00Z'
end_date = '2023-04-20T00:00:00Z'

# Initial fetch data
closing_prices = get_data(symbols, timeframe, start_date, end_date)
closing_prices.to_csv('IlliquidAssetData.csv')

# Update already fetched data
update_data(symbols, timeframe, "IlliquidAssetData.csv")
```
## Liquid Asset price
We use Ethereum as an example. It is assumed that closing prices are stored in a `pandas` DataFrame called `liquid` with no empty rows.

Create a class instance, then call the `liquidModel()` method to begin.
```
eth_model = StochasticPriceModel(liquid_data = liquid)
eth_model.liquidModel(backtesting = True, timeout = 120, train_size = 0.8)
```
Before running the model, you may call the `fit_distribution()` method to set the returns distribution yourself. Change arguments as needed.

`distributions` is a list of scipy distributions.
```
eth_model.fit_distribution(train_data, timeout, distributions = None)
```
After running the model, you may perform appropriate visualizations.
```
eth_model.plotEstimates()
eth_model.generateEstimateFigure()
```
If backtesting proves successful, you may call the `liquidModel()` method again to estimate projections.
```
eth_model.illiquidModel(backtesting = False, horizon = 30)
eth_model.plotEstimates()
eth_model.generateEstimateFigure()
```
You may also get needed properties for future use.
```
eth_model.backtestMAPE
eth_model.lower_confidence
eth_model.median_confidence
eth_model.upper_confidence
```
## Illiquid Asset Price
Here, we use Beefy Finance as an example. You will need a liquid asset when modeling illiquid asset prices. Here, we use Ethereum as our liquid asset. It is assumed that closing prices are stored in a `pandas` DataFrame called `liquid` with no empty rows. Once again, you can call the `fit_distribution()` method before running the model.

Create a class instance, then call the `illiquidModel()` method to begin.
```
illiquid_model = StochasticPriceModel(liquid_data = liquid, illiquid_data = illiquid)
illiquid_model.illiquidModel(backtesting = True, timeout = 120, train_size = 0.8)
```
Use the same visualization functions as above, and get the needed properties. You may also get an additional property with illiquid assets.
```
illiquid_model.assetCorrelation
```
# References
* Ang, A., Papanikolaou, D., & Westerfield, M. M. (2014). Portfolio choice with illiquid assets. Management Science, 60(11), 2737-2761.
* Nolan, J. P. (2003). Modeling financial data with stable distributions. In Handbook of heavy tailed distributions in finance (pp. 105-130). North-Holland.
