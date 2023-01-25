import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from ewstools.models import simulate_may, simulate_ricker, simulate_rosen_mac

from helpers import plot_timeseries


class Simulator:
    def __init__(self, n_per_model:int, frac_bifurcate:float, tmax:int, seed:int):

        assert 0<= frac_bifurcate <= 1., "bifurcation fracton must lie in [0, 1]"

        self.name = ''
        self.n_per_model = n_per_model
        self.frac_bifurcate = frac_bifurcate
        self.n_bifurcate = int(frac_bifurcate * n_per_model)
        self.tmax = tmax
        self.seed = seed

        self.timeseries = None

    def generate_timeseries(self):

        # Set indicators of bifurcation
        self.timeseries[:self.n_bifurcate, 0] = 1

        for i in range(self.n_bifurcate):
            self.timeseries[i, 1:] = self.gen_bifurcation()

        for i in range(self.n_bifurcate, self.n_per_model):
            self.timeseries[i, 1:] = self.gen_nonbifurcation()

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
                 tmax:int, dt:int, seed:int):
        super().__init__(n_per_model, frac_bifurcate, tmax, seed)
        
        self.name = 'May'
        self.dt = dt
        self.timeseries = np.zeros((n_per_model, 1 + int(tmax / dt)))

    def gen_bifurcation(self):
        return simulate_may(tmax=self.tmax, dt=self.dt).x

    def gen_nonbifurcation(self):
        return simulate_may(tmax=self.tmax, dt=self.dt).x


class Ricker(Simulator):
    def __init__(self, n_per_model:int, frac_bifurcate:float, 
                 tmax:int, seed:int):
        super().__init__(n_per_model, frac_bifurcate, tmax, seed)
        
        self.name = 'Ricker'
        self.timeseries = np.zeros((n_per_model, 1 + tmax))

    def gen_bifurcation(self):
        return simulate_ricker(tmax=self.tmax)

    def gen_nonbifurcation(self):
        return simulate_ricker(tmax=self.tmax)


class RosenMac(Simulator):
    def __init__(self, n_per_model:int, frac_bifurcate:float, 
                 tmax:int, dt:int, seed:int):
        super().__init__(n_per_model, frac_bifurcate, tmax, seed)
        
        self.name = 'Rosen-Mac'
        self.dt = dt
        self.timeseries = np.zeros((n_per_model, 1 + int(tmax / dt)))

    def gen_bifurcation(self):
        return simulate_rosen_mac(tmax=self.tmax, dt=self.dt).x

    def gen_nonbifurcation(self):
        return simulate_rosen_mac(tmax=self.tmax, dt=self.dt).x

    
if __name__ == "__main__":

    n_per_model = 2
    frac_bifurcate = 0.5
    tmax = 500
    dt = 0.01
    seed = 1234

    # Generate simulation results
    # sim = Simulator(n_per_model, frac_bifurcate, tmax, dt)
    # sim.gen_all_models()
    # sim.plot_timeseries()

    simulators = [
                    May(n_per_model, frac_bifurcate, tmax, dt, seed),
                    Ricker(n_per_model, frac_bifurcate, tmax, seed),
                    RosenMac(n_per_model, frac_bifurcate, tmax, dt, seed)
                 ]
    for simulator in simulators:
        simulator.generate_timeseries()
        simulator.save_simdata()
    plot_timeseries(simulators)