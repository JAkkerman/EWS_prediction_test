import matplotlib.pyplot as plt
import numpy as np

def plot_timeseries(sim_types:list):

    n_types = len(sim_types)
    fig, ax = plt.subplots(n_types, 1, figsize=(7, 2 * n_types))

    if type(ax) != np.ndarray:
        ax = [ax]

    for i, sim_type in enumerate(sim_types):

        # Set colors for bifurcations and non-bifurcations
        ax[i].set_prop_cycle(color=np.concatenate((
                                    ['red' for _ in range(sim_type.n_bifurcate)], 
                                    ['blue' for _ in range(sim_type.n_per_model - sim_type.n_bifurcate)]))
                            )
        
        # Plot time series
        ax[i].plot(sim_type.timeseries[:, 1:].T, alpha=0.5)
        ax[i].set_xlabel('time')
        ax[i].set_ylabel('$y_t$')
        ax[i].set_title(sim_type.name)

    plt.tight_layout()
    plt.show()


def plot_indicators(indicator_types:list):

    n_types = len(indicator_types)

    fig, ax = plt.subplots(n_types + 1, 1, figsize=(7, 2 * (n_types + 1)))

    if type(ax) != np.ndarray:
        ax = [ax]

    # First plot the simulated data
    # ax[0].set_prop_cycle(color=np.concatenate((
    #                                 ['red' for _ in range(indicator_type.n_bifurcate)], 
    #                                 ['blue' for _ in range(indicator_type.n_per_model - indicator_type.n_bifurcate)]))
    #                         )
    ax[0].plot(indicator_types[0].timeseries.to_numpy().T, alpha=0.5)
    ax[0].set_title(indicator_types[0].to_test)


    for i, indicator_type in enumerate(indicator_types):        
        # Plot indicator time series
        ax[i+1].plot(indicator_type.indicator.to_numpy().T)
        ax[i+1].set_xlabel('time')
        ax[i+1].set_ylabel('$I_t$')
        ax[i+1].set_title(indicator_type.name)

    plt.tight_layout()
    plt.show()