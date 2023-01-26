import numpy as np
import pandas as pd

from helpers import plot_indicators


class Indicator:
    def __init__(self, to_test:str, seed:int, rolling_window:float):
        
        assert 0. <= rolling_window <= 1., 'Rolling window must lie in [0, 1].'

        # Initialize result arrays for different models
        self.name = None
        self.to_test = to_test
        self.seed = seed
        self.rolling_window = rolling_window

        self.timeseries = None
        self.indicator = None
        self.results = None

    def open_data(self):
        
        # Open csv containing simulated data
        df = pd.read_csv(f'simulated_timeseries/simdata_{self.to_test}_{self.seed}.csv')

        # Generate data structures for timeseries, indicator and results
        self.timeseries = df[df.columns[1:]]
        # self.indicator = np.zeros((df.shape[0], df.shape[1] - 1))
        self.results = np.zeros((df.shape[0], 2))
        self.results[:, 0] = df[df.columns[0]].to_numpy()

    def save_results(self):
        pass

    def compute_indicator(self):
        pass

    def compute_ktau(self):
        pass

    def compute_results(self):
        pass


class Variance(Indicator):
    def __init__(self, to_test:str, seed:int, rolling_window:float):
        super().__init__(to_test, seed, rolling_window)
        self.name = 'var'

        self.open_data()

    def compute_indicator(self):
        self.indicator = self.timeseries.rolling(window=int(self.rolling_window * self.timeseries.shape[1]), axis=1).var()


if __name__ == "__main__":

    to_test = 'Ricker'
    seed = 1234
    rolling_window = 0.25

    indicators = [
                    Variance(to_test, seed, rolling_window)
                 ]

    for indicator in indicators:
        indicator.compute_indicator()
    plot_indicators(indicators)
