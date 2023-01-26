import numpy as np
import pandas as pd
import ewstools
from copy import copy

from helpers import plot_indicators


class Indicators:
    def __init__(self, to_test:str, seed:int, rolling_window:float, bifurcation_threshold:float,
                 pred_interval_max:float=0.8, pred_interval_size:float=0.2):
        
        assert 0. <= rolling_window <= 1., 'Rolling window must lie in [0, 1].'

        # Initialize result arrays for different models
        self.name = None
        self.to_test = to_test
        self.seed = seed
        self.rolling_window = rolling_window
        self.bifurcation_threshold = bifurcation_threshold
        self.pred_interval_max = pred_interval_max
        self.pred_interval_size = pred_interval_size

        self.timeseries = None
        self.results = None

        self.open_data()

    def open_data(self):
        
        # Open csv containing simulated data
        df = pd.read_csv(f'simulated_timeseries/simdata_{self.to_test}_{self.seed}.csv')

        self.timeseries = [ewstools.TimeSeries(df.iloc[i][df.columns[1:]]) 
                           for i in range(df.shape[0])]

        # Find when timeseries bifurcate
        for is_bufurcated, ts in zip(df['0'], self.timeseries):

            # If last bifurcation was found, break
            if not is_bufurcated:
                break

            # Set transition time to first time threshold is passed
            ts.transition = (ts.state.state <= self.bifurcation_threshold).idxmax()


        self.results = np.zeros((df.shape[0], 2))
        self.results[:, 0] = df[df.columns[0]].to_numpy()

    def compute_indicators(self):
        for ts in self.timeseries:
            ts.compute_var(rolling_window=self.rolling_window)
            ts.compute_cv(rolling_window=rolling_window)
            ts.compute_auto(rolling_window=rolling_window)
            ts.compute_skew(rolling_window=rolling_window)

    def compute_ktau(self):
        for ts in self.timeseries:
            ts.compute_ktau()

    def compute_results(self):

        results = []

        pred_interval = np.array([self.pred_interval_max - self.pred_interval_size, 
                                  self.pred_interval_max])
        
        for ts in self.timeseries:
            
            if ts.transition != None:
                # Compute if bifurcation is predicted correctly
                t_start = int(len(ts.state.state) * self.rolling_window)
                t_pred_start = str(int(t_start + (int(ts.transition) - t_start) * pred_interval[0]))
                t_pred_end = str(int(t_start + (int(ts.transition) - t_start) * pred_interval[1]))

                ts.compute_ktau(tmin=t_pred_start, tmax=t_pred_end)
                res_dict = copy(ts.ktau)
                res_dict['true'] = 1

            else:
                # Compute if nonbifurcation does not yield signal
                ts.compute_ktau()
                res_dict = copy(ts.ktau)
                res_dict['true'] = 0

            res_dict['pred_interval_max'] = pred_interval[1]
            results.append(res_dict)

        self.results = pd.DataFrame(results)
        self.save_results()

    def save_results(self):
        self.results.to_csv(f'prediction_values/pred_{self.to_test}_{self.seed}.csv')
        

if __name__ == "__main__":

    all_sim_names = [
                        'Ricker', 
                        # 'May'
                    ]
    seed = 1234
    rolling_window = 0.25

    bifurcation_thresholds = {
        'Ricker': 2.5,
        'May': 0.25,
        'Rosen-Mac': 0.1
    }

    indicators = [Indicators(sim_name, seed, rolling_window, bifurcation_thresholds[sim_name]) 
                  for sim_name in all_sim_names]

    for indicator in indicators:
        indicator.compute_indicators()
        indicator.compute_results()
    # plot_indicators(indicators, bifurcation_thresholds)
    
