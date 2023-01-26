import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from ewstools.models import simulate_may, simulate_ricker, simulate_rosen_mac

from helpers import plot_timeseries


class Simulator:
    def __init__(self, n_per_model:int, frac_bifurcate:float, tmax:int, seed:int, 
                 standard_params:bool):

        assert 0 <= frac_bifurcate <= 1., "bifurcation fraction must lie in [0, 1]"

        self.name = ''                                          # Name of model
        self.n_per_model = n_per_model                          # Number of simulations per model
        self.frac_bifurcate = frac_bifurcate                    # Fraction of total sim with bifurcations
        self.n_bifurcate = int(frac_bifurcate * n_per_model)    # Total number of sim with bifurcations
        self.tmax = tmax                                        # Max simulation length
        self.seed = seed                                        # Seed used by numpy random generator
        self.standard_params = standard_params                  # Use only standard parameters given by model, or stochastic range of forcer

        self.timeseries = None                                  # Array that holds the time series

        self.f_min = None                                       # Minimum value of forcer variable range
        self.f_bif_min = None                                   # Minimum value where forcer variable causes bifurcation
        self.f_max = None                                       # Maximum value of forcer variable range
        self.f_patch = None                                     # Patch between range of forcer used for bifurcation and nonbifurcation 
        self.f_range_bif = None                                 # Array that holds forcer range for bifurcation
        self.f_range_nobif = None                               # Array that holds forcer range for nonbifurcation


    def generate_timeseries(self):

        # Set indicators of bifurcation
        self.timeseries[:self.n_bifurcate, 0] = 1

        for i, j in zip(range(self.n_bifurcate), range(self.n_bifurcate, self.n_per_model)):

            # If nonstandard params used, generate new
            if not self.standard_params:
                self.set_nonstandard_params()

            # Generate time series
            self.timeseries[i, 1:] = self.gen_bifurcation()
            self.timeseries[j, 1:] = self.gen_nonbifurcation()

    def set_nonstandard_params(self):
        self.f_range_bif = [self.f_min, self.f_bif_min + (self.f_max - self.f_bif_min) * np.random.rand()]
        self.f_range_nobif = [self.f_min, self.f_min + (self.f_bif_min - self.f_min - self.f_patch) * np.random.rand()]

    def gen_bifurcation(self):
        pass

    def gen_nonbifurcation(self):
        pass

    def save_simdata(self):
        pd.DataFrame(self.timeseries).to_csv(
            f'simulated_timeseries/simdata_{self.name}_{self.seed}.csv',
            index=False
        )


class May(Simulator):
    def __init__(self, n_per_model:int, frac_bifurcate:float, 
                 tmax:int, dt:int, seed:int, standard_params:bool=False):
        super().__init__(n_per_model, frac_bifurcate, tmax, seed, standard_params)
        
        self.name = 'May'
        self.dt = dt
        self.timeseries = np.zeros((n_per_model, 1 + int(tmax / dt)))
        self.f_min = 0.15
        self.f_bif_min = 0.27
        self.f_max = 0.4
        self.f_patch = 0.01

        # Set up standard forcing parameter (h) ranges as used by default in ewstools package
        self.f_range_bif = [self.f_min, self.f_bif_min]
        self.f_range_nobif = [self.f_min, self.f_min]

    def gen_bifurcation(self):
        return simulate_may(tmax=self.tmax, dt=self.dt, h=self.f_range_bif).x

    def gen_nonbifurcation(self):
        return simulate_may(tmax=self.tmax, dt=self.dt, h=self.f_range_nobif).x


class Ricker(Simulator):
    def __init__(self, n_per_model:int, frac_bifurcate:float, 
                 tmax:int, seed:int, standard_params:bool=False):
        super().__init__(n_per_model, frac_bifurcate, tmax, seed, standard_params)
        
        self.name = 'Ricker'
        self.timeseries = np.zeros((n_per_model, 1 + tmax))
        self.f_min = 0.
        self.f_bif_min = 2.7
        self.f_max = 4.
        self.f_patch = 0.3

        # Set up standard forcing parameter (F) ranges as used by default in ewstools package
        self.f_range_bif = [self.f_min, self.f_bif_min]
        self.f_range_nobif = [self.f_min, self.f_min]

    def gen_bifurcation(self):
        return simulate_ricker(tmax=self.tmax, F=self.f_range_bif)

    def gen_nonbifurcation(self):
        return simulate_ricker(tmax=self.tmax, F=self.f_range_nobif)


class RosenMac(Simulator):
    def __init__(self, n_per_model:int, frac_bifurcate:float, 
                 tmax:int, dt:int, seed:int, standard_params:bool=False):
        super().__init__(n_per_model, frac_bifurcate, tmax, seed, standard_params)
        
        self.name = 'Rosen-Mac'
        self.dt = dt
        self.timeseries = np.zeros((n_per_model, 1 + int(tmax / dt)))
        self.f_min = 12.
        self.f_bif_min = 16.
        self.f_max = 18.
        self.f_patch = 0.5

        # Set up standard forcing parameter (a) ranges as used by default in ewstools package
        self.f_range_bif = [self.f_min, self.f_bif_min]
        self.f_range_nobif = [self.f_min, self.f_min]

    def gen_bifurcation(self):
        return simulate_rosen_mac(tmax=self.tmax, dt=self.dt, a=self.f_range_bif).x

    def gen_nonbifurcation(self):
        return simulate_rosen_mac(tmax=self.tmax, dt=self.dt, a=self.f_range_nobif).x

    
if __name__ == "__main__":

    # Set parameters used in simulation
    n_per_model = 100
    frac_bifurcate = 0.5
    tmax = 500
    dt = 0.1
    seed = 1234
    standard_params = False

    # Generate simulator objects
    simulators = [
                    # May(n_per_model, frac_bifurcate, tmax, dt, seed, standard_params=standard_params),
                    Ricker(n_per_model, frac_bifurcate, tmax, seed, standard_params=standard_params),
                    # RosenMac(n_per_model, frac_bifurcate, tmax, dt, seed, standard_params=standard_params)
                 ]
    
    # Generate simulation data and save time series
    for simulator in simulators:
        simulator.generate_timeseries()
        simulator.save_simdata()

    # Plot generated time series
    plot_timeseries(simulators)