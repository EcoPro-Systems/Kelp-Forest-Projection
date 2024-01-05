import matplotlib.pyplot as plt
import numpy as np
import pickle
from itertools import cycle

kelp_files = [
    #'Data/kelp_metrics_27_37_kelp_timeseries.pkl',
    'Data/kelp_metrics_32_37_kelp_timeseries.pkl',
    'Data/kelp_metrics_27_32_kelp_timeseries.pkl',
]

sst_files = [
    #'Data/kelp_metrics_27_37_sst_timeseries.pkl',
    'Data/kelp_metrics_32_37_sst_timeseries.pkl',
    'Data/kelp_metrics_27_32_sst_timeseries.pkl',
]

if __name__ == "__main__":
    # create figure and overplot data
    fig, ax = plt.subplots(2,1, figsize=(10,10))

#        '2019-08-15T00:00:00.000000000', '2019-11-15T00:00:00.000000000',
#        '2020-02-15T00:00:00.000000000', '2020-05-15T00:00:00.000000000',
#        '2020-08-15T00:00:00.000000000', '2020-11-15T00:00:00.000000000'],
#       dtype='datetime64[ns]')

    colors = cycle(['salmon', 'limegreen', 'purple'])

    # plot kelp time series
    for file in kelp_files:
        ncolor = next(colors)
        with open(file, 'rb') as f:
            data = pickle.load(f)
        parts = file.split('_')
        #ax[0].errorbar(data['time'], data['mean'], yerr=data['std'], fmt='-', markersize=4, capsize=2, label=f"{parts[2]}-{parts[3]}N", color=ncolor, alpha=0.9)
        ax[0].plot(data['time'], data['mean'], color=ncolor, alpha=0.9, label=f"{parts[2]}-{parts[3]}N")
        # plot upper limit
        ax[0].fill_between(data['time'], data['mean']+data['std'], data['mean']-data['std'], color=ncolor, alpha=0.2)
        ax[0].set_ylabel(r'Kelp Area [$m^2$]', fontsize=14)
        ax[0].set_title('Kelp Time Series', fontsize=16)
        ax[0].grid(True, ls='--')
    
    colors = cycle(['salmon', 'limegreen', 'purple'])

    # plot sst time series
    for file in sst_files:
        ncolor = next(colors)
        with open(file, 'rb') as f:
            data = pickle.load(f)
        parts = file.split('_')
        ax[1].errorbar(data['time'], data['mean']-273.15, yerr=data['std'], fmt='-', markersize=4, capsize=2, label=f"{parts[2]}-{parts[3]}N", color=ncolor, alpha=0.9)
        ax[1].set_ylabel(r'Temperature [$^\circ$C]', fontsize=14)
        ax[1].set_title('SST Time Series', fontsize=16)
        ax[1].grid(True, ls='--')
    
    # set xticks
    ax[1].set_xlabel('Time [year]', fontsize=14)
    ax[0].set_ylim([0,825])
    # set legend
    ax[0].legend(loc='best')
    ax[1].legend(loc='best')

    # save figure
    plt.tight_layout()
    plt.savefig('Data/kelp_sst_timeseries.png')
    plt.show()
    plt.close()