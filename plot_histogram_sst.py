import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle

# create histogram of kelp area change for each season
def histogram_kelp(kelp_metrics):
    # seasonal derivatives
    season_names = {
        1:'Winter (Jan-Mar)',  # | Q1      | winter        | January – March         | 02-15T00:00:00.00 |
        2:'Spring (Apr-Jun)',  # | Q2      | spring        | April – June            | 05-15T00:00:00.00 |
        3:'Summer (Jul-Sep)',  # | Q3      | summer        | July – September        | 08-15T00:00:00.00 |
        4:'Fall (Oct-Dec)'     # | Q4      | fall          | October – December      | 11-15T00:00:00.00 |
    }

    # find lat limits
    lower = np.min(kelp_metrics['lat'])
    upper = np.max(kelp_metrics['lat'])

    # find seasons for numpy.datetime64 type
    seasons = np.array([season_names[(t.astype('datetime64[M]').astype(int)%12)//3+1] for t in kelp_metrics['time']])

    fig, ax = plt.subplots(2,2, figsize=(8,8))
    fig.suptitle(f"Sea Surface Temperature by Season between ({lower:.1f} - {upper:.1f}N)")
    ax = ax.flatten()

    # create dict for saving
    sst_histogram = {}

    for i, season in enumerate(season_names.values()):
        mask = seasons == season
        mean = np.mean(kelp_metrics['temp'][mask])-273.15
        std = np.std(kelp_metrics['temp'][mask])
        ax[i].hist(kelp_metrics['temp'][mask]-273.15, bins=np.linspace(10,25,50), label=f"Mean: {mean:.1f}\nStd: {std:.1f}")
        ax[i].set_title(season)
        ax[i].set_xlabel(r'Temperature ($^\circ$C)')
        ax[i].set_yticks([])
        ax[i].legend(loc='best')
        ax[i].grid(True, ls='--', alpha=0.5)

        # save data to dict
        sst_histogram[season] = kelp_metrics['temp'][mask]-273.15
    
    plt.tight_layout()
    return fig, ax, sst_histogram

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
    fig, ax, hdata = histogram_kelp(data)
    plt.savefig(args.file_path.replace('.pkl', '_histogram_sst.png'))
    plt.close()

    # save data to pickle file
    with open(args.file_path.replace('.pkl', '_histogram_sst.pkl'), 'wb') as f:
        pickle.dump(hdata, f)