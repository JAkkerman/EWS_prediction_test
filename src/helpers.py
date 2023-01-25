import matplotlib.pyplot as plt
import numpy as np

def plot_timeseries(simtypes:list):

    n_types = len(simtypes)
    fig, ax = plt.subplots(n_types, 1, figsize=(7, 2 * n_types))

    if type(ax) != np.ndarray:
        ax = [ax]

    for i, simtype in enumerate(simtypes):

        # Set colors for bifurcations and non-bifurcations
        ax[i].set_prop_cycle(color=np.concatenate((
                                    ['red' for _ in range(simtype.n_bifurcate)], 
                                    ['blue' for _ in range(simtype.n_per_model - simtype.n_bifurcate)]))
                            )
        
        # Plot time series
        ax[i].plot(simtype.timeseries[:, 1:].T, alpha=0.5)
        ax[i].set_xlabel('time')
        ax[i].set_ylabel('$y_t$')
        ax[i].set_title(simtype.name)

    plt.tight_layout()
    plt.show()