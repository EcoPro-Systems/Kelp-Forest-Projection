import pickle
import argparse
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_regression

if __name__ == "__main__":
    # argparse for input filepath
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file_path', type=str, 
                        help='path to input metrics file', 
                        default="Data/kelp_metrics_27_30.pkl")

    parser.add_argument('-fs', '--file_path_sim', type=str, 
                        help='path to input metrics file', 
                        default="Data/kelp_metrics_sim_27_30.pkl")

    args = parser.parse_args()

    # load data from disk
    with open(args.file_path, 'rb') as f:
        data = pickle.load(f)

    # convert datetime64[ns] to days since min date 
    time = data['time'].astype('datetime64[D]')
    time = time - np.min(time)
    time = time.astype(int)
    time_dt = data['time'] # datetime format

    # inputs: time, periodic_time, lon, lat, temp -> kelp
    y = data['kelp']

    # calculate daylight duration as input feature
    print(len(time))

    # construct features
    features = [
        time, # days, 0-365*20
        data['sunlight'],
        data['temp_lag']-273.15,
        data['temp_lag2']-273.15,
        np.ones(len(time)) # w_1 * x_1 + w_2 * x_2 + ... + w_n * x_n + b
    ]

    feature_names = [
        'time',
        'sunlight',
        'temp_lag',
        'temp_lag2',
        'bias'
    ]

    X = np.array(features).T

    # remove nans
    nanmask = np.isnan(data['temp_lag']) | np.isnan(data['temp_lag2'])
    X = X[~nanmask]
    y = y[~nanmask]
    time = time[~nanmask]
    time_dt = time_dt[~nanmask]


    # load simulation data from disk
    with open(args.file_path_sim, 'rb') as f:
        data_sim = pickle.load(f)

    # convert datetime64[ns] to days since min date 
    time_sim = data_sim['time'].astype('datetime64[D]')
    time_sim = time_sim - np.min(time)
    time_sim = time_sim.astype(int)
    time_sim_dt = data_sim['time'] # datetime format

    # calculate daylight duration as input feature
    print(len(time_sim))

    # construct features
    features = [
        time_sim, # days, 0-365*20
        data_sim['sunlight'],
        data_sim['temp_lag'],
        data_sim['temp_lag2'],
        np.ones(len(time_sim)) # w_1 * x_1 + w_2 * x_2 + ... + w_n * x_n + b
    ]

    feature_names = [
        'time',
        'sunlight',
        'temp_lag',
        'temp_lag2',
        'bias'
    ]

    X_test = np.array(features).T

    # remove nans
    nanmask_test = np.isnan(data_sim['temp_lag']) | np.isnan(data_sim['temp_lag2'])
    X_test = X_test[~nanmask_test]
    time_sim = time_sim[~nanmask_test]
    time_sim_dt = time_sim_dt[~nanmask_test]

    # # compute correlation coefficient
    # for i in range(X.shape[1]-1):
    #     print(f'Correlation Coefficient {feature_names[i]}: {np.corrcoef(X[:,i], y)[0,1]:.3f}')

    # # compute mutual information
    # mi = mutual_info_regression(X[:,:-1], y)
    # for i in range(X.shape[1]-1):
    #     print(f'Mutual Information {feature_names[i]}: {mi[i]:.3f}')

    # sort data in time
    si = np.argsort(time)
    X = X[si]
    y = y[si]
    time = time[si]
    time_dt = time_dt[si]

    # fit a linear model to the data with OLS
    res = sm.OLS(y, X).fit()
    coeffs = res.params
    y_ols_train = np.dot(X, coeffs)
    y_ols_test = np.dot(X_test, coeffs)

    # compute the average absolute error
    abs_err_ols_train = np.abs(y - y_ols_train).mean()
    print(f"Avg. Absolute Error (OLS) Train: {abs_err_ols_train:.3f} m^2")

    # print average output
    print(f"Avg. Output (OLS) Train: {y_ols_train.mean():.3f} m^2")
    print(f"Avg. Output (OLS) Test: {y_ols_test.mean():.3f} m^2")

    # get unique times and bin data
    utime = np.unique(time)
    utime_dt = np.unique(time_dt)
    bmean = np.zeros(len(utime)) # binned data
    bstd = np.zeros(len(utime))

    # loop over each quarter and compute the mean and std
    for i, t in enumerate(utime):
        mask = time == t
        bmean[i] = np.mean(y[mask])
        bstd[i] = np.std(y[mask])

    utime_train = np.unique(time)
    utime_train_dt = np.unique(time_dt)
    mean_ols_train = np.zeros(len(utime_train)) # binned data OLS
    std_ols_train = np.zeros(len(utime_train))
    mean_sst_train = np.zeros(len(utime_train)) # binned data SST
    std_sst_train = np.zeros(len(utime_train))

    # loop over each quarter and compute the mean and std
    for i, t in enumerate(utime_train):
        mask = time == t
        mean_ols_train[i] = np.mean(y_ols_train[mask])
        std_ols_train[i] = np.std(y_ols_train[mask])
        mean_sst_train[i] = np.mean(data['temp'][~nanmask][si][mask])
        std_sst_train[i] = np.std(data['temp'][~nanmask][si][mask])

    utime_test = np.unique(time_sim)
    utime_test_dt = np.unique(time_sim_dt)
    mean_ols_test = np.zeros(len(utime_test)) # binned data OLS
    std_ols_test = np.zeros(len(utime_test))
    mean_sst_test = np.zeros(len(utime_test)) # binned data SST
    std_sst_test = np.zeros(len(utime_test))

    # loop over each quarter and compute the mean and std
    for i, t in enumerate(utime_test):
        mask = time_sim == t
        mean_ols_test[i] = np.mean(y_ols_test[mask])
        std_ols_test[i] = np.std(y_ols_test[mask])
        mean_sst_test[i] = np.mean(data_sim['temp'][~nanmask_test][mask])
        std_sst_test[i] = np.std(data_sim['temp'][~nanmask_test][mask])


    # plot the data
    fig, ax = plt.subplots(3, 1, figsize=(11, 10))
    # lat lon limits to title
    # extract lat/lon from file_path
    lat = float(args.file_path.split('_')[2])
    lon = float(args.file_path.split('_')[3].split('.')[0])
    fig.suptitle(f"Kelp Projections at {lat:.0f}--{lon:.0f} N")
    ax[0].errorbar(utime_dt, bmean, yerr=bstd, fmt='o', color='black', label='Kelp Watch Data')
    ax[0].errorbar(utime_train_dt, mean_ols_train, yerr=std_ols_train, fmt='.', ls='-', color='red', label=f'OLS Model (avg. err: {abs_err_ols_train:.1f} m^2)')
    #ax[0].errorbar(utime_train_dt, mean_ols_train, yerr=std_ols_train, fmt='.', ls='-',alpha=0.33, color='blue')
    ax[0].legend(loc='upper left')
    #ax[0].set_xlabel("Time")
    ax[0].set_ylabel("Kelp Area [m^2]")
    ax[0].grid(True,ls='--',alpha=0.5)
    ax[0].set_ylim([0,666])

    ax[1].plot(utime_train_dt, mean_ols_train,  ls='-', color='k', label='OLS Model')
    ax[1].plot(utime_test_dt, mean_ols_test, ls='-', color='red', label='Predicted')
    ax[1].set_ylim([0,666])
    ax[1].grid(True,ls='--',alpha=0.5)
    ax[1].set_xlabel("Time")
    ax[1].set_ylabel("Kelp Area [m^2]")
    ax[1].legend(loc='upper left')

    # plot temperature time series for each location
    ax[2].plot(utime_train_dt, mean_sst_train-273.15, 'k-', label='JPL MUR')
    ax[2].plot(utime_test_dt, mean_sst_test, 'r-', label='ESM4_ssp126')
    ax[2].set_xlabel('Time')
    ax[2].set_ylabel('Sea Surface Temperature [C]')
    ax[2].legend()
    ax[2].grid(True,ls='--',alpha=0.5)

    plt.tight_layout()
    plt.savefig(args.file_path.replace('.pkl', '_regressors.png'))
    plt.show()
