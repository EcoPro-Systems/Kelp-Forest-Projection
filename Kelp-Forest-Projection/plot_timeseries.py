import os
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

def sst_time_series(kelp_metrics, file_name):
    # create time vs temp plot
    fig, ax = plt.subplots(figsize=(10,5))

    lower = np.min(kelp_metrics['lat'])
    upper = np.max(kelp_metrics['lat'])

    # set up plot axes
    ax.set_ylabel('Sea Surface Temperature [C]', fontsize=14)
    ax.set_xlabel('Time [year]', fontsize=14)
    ax.set_title(f"Quarterly Averages of Temperature between ({lower:.1f} - {upper:.1f}N)", fontsize=16)

    # find unique times, compute mean + stdev of bins, and plot
    unique_times = np.unique(kelp_metrics['time'])
    mean_temp = []
    std_temp = []

    for t in unique_times:
        mask = kelp_metrics['time'] == t
        mean_temp.append(np.mean(kelp_metrics['temp'][mask]))
        std_temp.append(np.std(kelp_metrics['temp'][mask]))

    # convert to numpy arrays
    mean_temp = np.array(mean_temp)
    std_temp = np.array(std_temp)

    ax.errorbar(unique_times, mean_temp-273.15, yerr=std_temp, fmt='o', color='red', markersize=4, capsize=2)
    ax.plot(unique_times, mean_temp-273.15, 'k-',alpha=0.75,lw=2)
    ax.grid(True,ls='--')
    ax.set_xlim([np.min(unique_times), np.max(unique_times)])
    ax.set_ylim([np.min(mean_temp-273.15)-2, 
                 np.max(mean_temp-273.15)+max(std_temp)])
    plt.tight_layout()

    # save unique times, mean, and std as pickle file
    sst_time_series = {'time': unique_times, 'mean': mean_temp, 'std': std_temp}
    with open(file_name.replace('.png', '.pkl'), 'wb') as f:
        pickle.dump(sst_time_series, f)
    print(f"Saved pickle file to {file_name.replace('.png', '.pkl')}")
    
    # save figure
    plt.savefig(file_name)
    plt.close()
    print(f"Saved figure to {file_name}")

def kelp_time_series(kelp_metrics, file_name):
    # create time vs temp plot
    fig, ax = plt.subplots(figsize=(10,5))

    lower = np.min(kelp_metrics['lat'])
    upper = np.max(kelp_metrics['lat'])

    # set up plot axes
    ax.set_ylabel(r'Kelp Area [$m^2$]', fontsize=14)
    ax.set_xlabel('Time [year]', fontsize=14)
    ax.set_title(f"Quarterly Averages of Kelp between ({lower:.1f} - {upper:.1f}N)", fontsize=16)

    # find unique times, compute mean + stdev of bins, and plot
    unique_times = np.unique(kelp_metrics['time'])
    mean_kelp = []
    std_kelp = []

    for t in unique_times:
        mask = kelp_metrics['time'] == t
        mean_kelp.append(np.mean(kelp_metrics['kelp'][mask]))
        std_kelp.append(np.std(kelp_metrics['kelp'][mask]))

    # convert to numpy arrays
    mean_kelp = np.array(mean_kelp)
    std_kelp = np.array(std_kelp)

    ax.errorbar(unique_times, mean_kelp, yerr=std_kelp, fmt='o', color='red', markersize=4, capsize=2)
    ax.plot(unique_times, mean_kelp, 'k-',alpha=0.75,lw=2)
    ax.grid(True,ls='--')
    ax.set_xlim([np.min(unique_times), np.max(unique_times)])
    ax.set_ylim([0, np.max(mean_kelp)+max(std_kelp)])
    plt.tight_layout()

    # save unique times, mean, and std as pickle file
    kelp_time_series = {'time': unique_times, 'mean': mean_kelp, 'std': std_kelp}
    with open(file_name.replace('.png', '.pkl'), 'wb') as f:
        pickle.dump(kelp_time_series, f)
    print(f"Saved pickle file to {file_name.replace('.png', '.pkl')}")

    # save figure
    plt.savefig(file_name)
    plt.close()
    print(f"Saved figure to {file_name}")

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
    sst_time_series(data, args.file_path.replace('.pkl', '_sst_timeseries.png'))
    kelp_time_series(data, args.file_path.replace('.pkl', '_kelp_timeseries.png'))