import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from ewstools.models import simulate_may, simulate_ricker, simulate_rosen_mac


class Simulator:
    def __init__(self, n_per_model:int, tmax:int, dt:float):

        self.n_per_model = n_per_model
        self.tmax = tmax
        self.dt = dt

        # Simulation data
        self.sims_may = np.zeros((n_per_model, 1 + int(tmax / dt)))
        self.sims_ricker = np.zeros((n_per_model, 1 + tmax))
        self.sims_rosen_mac = np.zeros((n_per_model, 1 + int(tmax / dt)))

    def gen_all_models(self):
        self.gen_may()
        self.gen_ricker()
        self.gen_rosen_mac()

    def gen_may(self):
        for i in range(self.n_per_model):
            self.sims_may[i, 1:] = simulate_may(tmax=self.tmax, dt=self.dt).x
    
    def gen_rosen_mac(self):
        for i in range(n_per_model):
            self.sims_rosen_mac[i, 1:] = simulate_rosen_mac(tmax=self.tmax, dt=self.dt).x

    def gen_ricker(self):
        for i in range(self.n_per_model):
            self.sims_ricker[i, 1:] = simulate_ricker(tmax=self.tmax)

    def plot_timeseries(self):
        fig, [ax1, ax2, ax3] = plt.subplots(3, 1, figsize=(10, 10))

        ax1.plot(self.sims_may[:, 1:].T)
        ax1.set_title('May')

        ax2.plot(self.sims_ricker[:, 1:].T)
        ax2.set_title('Ricker')

        ax3.plot(self.sims_rosen_mac[:, 1:].T)
        ax3.set_title('Rosen Mac')

        plt.show()

if __name__ == "__main__":

    n_per_model = 10
    tmax = 500
    dt = 0.01

    # Generate simulation results
    sim = Simulator(n_per_model, tmax, dt)
    sim.gen_all_models()
    sim.plot_timeseries()