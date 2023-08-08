import matplotlib.pyplot as plt
import numpy as np
import argparse
import pickle

# create histogram of kelp area change for each season
def histogram_kelp(kelp_metrics):
    # seasonal derivatives
    season_names = {
        1:'Winter -> Spring', # 3
        2:'Spring -> Summer', # 6
        3:'Summer -> Fall',   # 9
        4:'Fall -> Winter'    # 12
    }

    # find lat limits
    lower = np.min(kelp_metrics['lat'])
    upper = np.max(kelp_metrics['lat'])

    # find seasons for numpy.datetime64 type
    seasons = np.array([season_names[(t.astype('datetime64[M]').astype(int)%12)//3+1] for t in kelp_metrics['dtime']])

    fig, ax = plt.subplots(2,2, figsize=(8,8))
    fig.suptitle(f"Change in Kelp Area by Season between ({lower:.1f} - {upper:.1f}N)")
    ax = ax.flatten()

    for i, season in enumerate(season_names.values()):
        mask = seasons == season
        mean = np.mean(kelp_metrics['dkelp'][mask])
        std = np.std(kelp_metrics['dkelp'][mask])
        ax[i].hist(kelp_metrics['dkelp'][mask], bins=np.linspace(-1000,1000,50), label=f"Mean: {mean:.1f}\nStd: {std:.1f}")
        ax[i].set_title(season)
        ax[i].set_xlabel(r'Change in Kelp Area ($m^2$)')
        ax[i].set_yticks([])
        ax[i].legend(loc='best')
        ax[i].grid(True, ls='--', alpha=0.5)
    
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
    fig, ax = histogram_kelp(data)
    plt.savefig(args.file_path.replace('.pkl', '_histogram.png'))
    plt.show()