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


def plot_indicators(sim_types:list, bifurcation_thresholds:dict, 
                    plot_ktau:bool=False):

    for sim_type in sim_types:

        n_indicators = len(sim_type.timeseries[0].ews.columns)
        fig, ax = plt.subplots(n_indicators + 1, 1, figsize=(7, 2 * (n_indicators + 1)))
        
        ax[0].axhline(bifurcation_thresholds[sim_type.to_test], color='red', alpha=0.5, linestyle='dotted')

        for i, ts in enumerate(sim_type.timeseries):

            warmup = int(len(ts.state.state) * sim_type.rolling_window)
            T = np.arange(len(ts.state.state))

            # First plot the simulated data
            ax[0].plot(T[warmup:], ts.state.state.to_numpy()[warmup:], alpha=0.5)
            ax[0].set_title(sim_type.to_test)

            # Plot indicator time series
            for j, indicator_type in enumerate(ts.ews.columns):
                if not plot_ktau:
                    ax[j+1].plot(ts.ews[indicator_type].to_numpy())
                ax[j+1].set_xlabel('time')
                ax[j+1].set_ylabel('$I_t$')
                ax[j+1].set_title(f'{indicator_type}')
                if ts.transition != None:
                    ax[j+1].axvline(int(ts.transition), alpha=0.5, linestyle='dashed', color='black')

        plt.tight_layout()
        plt.show()