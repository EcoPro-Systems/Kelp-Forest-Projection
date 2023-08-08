import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.regression.linear_model import OLS

def lag_correlation(kelp_metrics):
    # find lat limits
    lower = np.min(kelp_metrics['dlat'])
    upper = np.max(kelp_metrics['dlat'])

    # find unique times, compute mean + stdev of bins, and plot
    unique_times = np.unique(kelp_metrics['dtime'])
    mean_temp = []
    mean_kelp = []
    std_temp = []
    mean_temp_lag = []
    std_temp_lag = []
    mean_temp_lag2 = []
    std_temp_lag2 = []

    # compute mean and std of temperature and kelp area
    for t in unique_times:
        mask = kelp_metrics['dtime'] == t
        mean_temp.append(np.mean(kelp_metrics['dtemp'][mask]))
        mean_kelp.append(np.mean(kelp_metrics['dkelp'][mask]))
        std_temp.append(np.std(kelp_metrics['dtemp_temp'][mask]))
        mean_temp_lag.append(np.mean(kelp_metrics['dtemp_temp_lag'][mask]))
        std_temp_lag.append(np.std(kelp_metrics['dtemp_temp_lag'][mask]))
        mean_temp_lag2.append(np.mean(kelp_metrics['dtemp_temp_lag2'][mask]))
        std_temp_lag2.append(np.std(kelp_metrics['dtemp_temp_lag2'][mask]))

    # convert to numpy arrays
    mean_temp = np.array(mean_temp)
    mean_kelp = np.array(mean_kelp)
    std_temp = np.array(std_temp)
    mean_temp_lag = np.array(mean_temp_lag)
    std_temp_lag = np.array(std_temp_lag)
    mean_temp_lag2 = np.array(mean_temp_lag2)
    std_temp_lag2 = np.array(std_temp_lag2)


    fig, ax = plt.subplots(figsize=(7,6))
    ax.plot(mean_temp-273.15, mean_kelp,  'o', color='black', markersize=4)
    
    # measure trend line
    A = np.vstack([mean_temp-273.15, np.ones(len(mean_temp))]).T
    res = OLS(mean_kelp, A).fit()
    m,b = res.params[0], res.params[1]
    x = np.linspace(np.min(mean_temp-273.15), np.max(mean_temp-273.15), 100)
    corrcoeff = np.corrcoef(mean_temp-273.15, mean_kelp)[0,1]
    ax.plot(x, m*x+b, 'k-',alpha=0.75,lw=2,label=f'Temperature ($r={corrcoeff:.2f}$)')

    # temp lagged by one quarter
    ax.plot(mean_temp_lag[1:]-273.15, mean_kelp[1:],  'o', color='red', markersize=4)
    
    # measure trend line
    A = np.vstack([mean_temp_lag[1:]-273.15, np.ones(len(mean_temp_lag[1:]))]).T
    res = OLS(mean_kelp[1:], A).fit()
    m,b = res.params[0], res.params[1]
    corrcoeff = np.corrcoef(mean_temp_lag[1:]-273.15, mean_kelp[1:])[0,1]
    ax.plot(x, m*x+b, 'r-',alpha=0.75,lw=2,label=r'Temperature Lagged by One Quarter ($r=%.2f$)'%corrcoeff)

    # temp lagged by two quarters
    ax.plot(mean_temp_lag2[2:]-273.15, mean_kelp[2:],  'o', color='blue', markersize=4)
    
    # measure trend line
    A = np.vstack([mean_temp_lag2[2:]-273.15, np.ones(len(mean_temp_lag2[2:]))]).T
    res = OLS(mean_kelp[2:], A).fit()
    m,b = res.params[0], res.params[1]
    corrcoeff = np.corrcoef(mean_temp_lag2[2:]-273.15, mean_kelp[2:])[0,1]
    ax.plot(x, m*x+b, 'b-',alpha=0.75,lw=2,label=r'Temperature Lagged by Two Quarters ($r=%.2f$)'%corrcoeff)

    ax.set_ylabel(r'Change in Kelp Area $[m^2]$', fontsize=14)
    ax.set_xlabel('Sea Surface Temperature [C]', fontsize=14)
    ax.set_title(f"Lag Correlation between ({lower:.1f} - {upper:.1f}N)", fontsize=16)
    ax.grid(True,ls='--')
    ax.legend(loc='best')
    plt.tight_layout()
    return fig, ax


if __name__ == "__main__":
    # argparse for input filepath
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file_path', type=str, 
                        help='path to input metrics file', 
                        default="Data/kelp_metrics_31_36.pkl")
    args = parser.parse_args()

    # load data from disk
    with open(args.file_path, 'rb') as f:
        data = pickle.load(f)

    # plot time series
    fig, ax = lag_correlation(data)
    plt.savefig(args.file_path.replace('.pkl', '_lag_correlation.png'))
    plt.show()