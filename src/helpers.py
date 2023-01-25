import matplotlib.pyplot as plt

def plot_timeseries(simtypes:list):

    n_types = len(simtypes)
    fig, ax = plt.subplots(n_types, 1, figsize=(7, 2 * n_types))

    for i, simtype in enumerate(simtypes):
        ax[i].plot(simtype.timeseries[:, 1:].T)
        ax[i].set_xlabel('time')
        ax[i].set_ylabel('$y_t$')
        ax[i].set_title(simtype.name)

    plt.tight_layout()
    plt.show()