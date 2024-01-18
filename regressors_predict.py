import pickle
import argparse
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from scipy.spatial import cKDTree
import xarray as xr
from tqdm import tqdm

if __name__ == "__main__":
    # argparse for input filepath
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file_path', type=str, 
                        help='path to input metrics file', 
                        default="Data/kelp_metrics_27_37.pkl")
    parser.add_argument('-fs', '--file_path_sim', type=str, 
                        help='path to input metrics file', 
                        default="Data/kelp_metrics_sim_27_37_ssp585_BGL.pkl")
    #model type
    parser.add_argument('-m', '--model', type=str, 
                        help='model type (OLS or MLP)',
                        default="OLS")
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

    # construct features
    features = [
        time_sim, # days
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
    print(len(time_sim))

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

    if args.model.lower() == 'ols':
        # fit a linear model to the data with OLS
        res = sm.OLS(y, X).fit()
        coeffs = res.params
        y_ols_train = np.dot(X, coeffs)
        y_ols_test = np.dot(X_test, coeffs)
    elif args.model.lower() == 'mlp':
        X = X[:,:-1] # remove bias
        X_test = X_test[:,:-1]
        Xp = (X - X.mean(axis=0)) / X.std(axis=0)
        Xp_test = (X_test - X_test.mean(axis=0)) / X_test.std(axis=0)
        # fit a linear model to the data with OLS
        # 500 - 22930.63165755

        mlp = MLPRegressor(hidden_layer_sizes=(8,), max_iter=100, alpha=0.001, solver='adam', verbose=True)
        mlp.fit(Xp, y)
        y_ols_train = mlp.predict(Xp)
        y_ols_test = mlp.predict(Xp_test)

    # compute the average absolute error
    abs_err_ols_train = np.abs(y - y_ols_train).mean()
    print(f"Avg. Absolute Error Train: {abs_err_ols_train:.3f} m^2")

    # print average output
    print(f"Avg. Output Train: {y_ols_train.mean():.3f} m^2")
    print(f"Avg. Output Test: {y_ols_test.mean():.3f} m^2")

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

    # compute slope of time vs temperature
    slope, intercept = np.polyfit(utime_train, mean_ols_train, 1)
    print(f"Slope: {slope:.5f} C / year")

    # compute for test data
    slope_test, intercept_test = np.polyfit(utime_test, mean_ols_test, 1)
    print(f"Slope: {slope_test:.5f} C / year")


    # plot the data
    fig, ax = plt.subplots(3, 1, figsize=(11, 10))
    # lat lon limits to title
    # extract lat/lon from file_path
    lat = float(args.file_path_sim.split('_')[3])
    lon = float(args.file_path_sim.split('_')[4].split('.')[0])
    # extract type of model from fs
    climate_model = args.file_path_sim.split('_')[-2]
    scaling = args.file_path_sim.split('_')[-1].split('.')[0]
    fig.suptitle(f"Kelp Projections at {lat:.0f}-{lon:.0f} N using {climate_model.upper()} {scaling.upper()}", fontsize=16)
    ax[0].errorbar(utime_dt, bmean, yerr=bstd, fmt='o', color='black', label='Kelp Watch Data',alpha=0.90)
    ax[0].plot(utime_test_dt, mean_ols_test, ls='-', color='red', label='Projections')
    ax[0].errorbar(utime_train_dt, mean_ols_train, yerr=std_ols_train, fmt='.', ls='-', color='limegreen', label=rf'{args.model.upper()} Model (avg. err: {abs_err_ols_train:.1f} m$^2$)')

    #ax[0].errorbar(utime_train_dt, mean_ols_train, yerr=std_ols_train, fmt='.', ls='-',alpha=0.33, color='blue')
    ax[0].legend(loc='upper left')
    #ax[0].set_xlabel("Time")
    ax[0].set_ylabel(r"Kelp Area [m$^2$]")
    ax[0].grid(True,ls='--',alpha=0.5)
    ax[0].set_ylim([0,666])
    ax[0].set_xlim([np.min(utime_dt), np.max(utime_dt)])

    ax[1].plot(utime_test_dt, mean_ols_test, ls='-', color='red', label='Projections',alpha=0.9)
    ax[1].plot(utime_train_dt, mean_ols_train,  ls='-', color='limegreen', label=f'{args.model.upper()} Model')
    ax[1].set_ylim([0,666])
    ax[1].grid(True,ls='--',alpha=0.5)
    ax[1].set_xlabel("Time")
    ax[1].set_ylabel(r"Kelp Area [m$^2$]")
    ax[1].legend(loc='upper left')
    ax[1].set_xlim([np.min(utime_test_dt), np.max(utime_test_dt)])

    # plot temperature time series for each location
    ax[2].plot(utime_test_dt, mean_sst_test, 'c-', label=f'{climate_model.upper()} {scaling.upper()}')
    ax[2].plot(utime_train_dt, mean_sst_train-273.15, 'k-', label=f'JPL MUR')
    ax[2].set_xlabel('Time')
    ax[2].set_ylabel('Sea Surface Temperature [C]')
    ax[2].legend()
    ax[2].grid(True,ls='--',alpha=0.5)
    ax[2].set_xlim([np.min(utime_test_dt), np.max(utime_test_dt)])

    plt.tight_layout()
    file_name = args.file_path_sim.replace('.pkl', '_regressors.png')
    file_name = file_name.replace('metrics', args.model)
    plt.savefig(file_name)
    print(f"Saved {file_name}")
    plt.close()


    # create nearest neighbor algorithm to map time, lat, lon to kelp
    tree = cKDTree(np.array([time_sim, data_sim['lat'][~nanmask_test], data_sim['lon'][~nanmask_test]]).T)
    def predict_kelp(time, lat, lon):
        dist, idx = tree.query(np.array([time, lat, lon]).T)
        X_test = np.array([time, data_sim['sunlight'][idx], data_sim['temp_lag'][idx], data_sim['temp_lag2'][idx], 1])
        return np.dot(X_test, coeffs)    

    # days since min date
    times = np.unique(data_sim['time']).astype('datetime64[D]')
    times = times - np.min(time)
    times = times.astype(int)

    lat = data_sim['lat']
    lon = data_sim['lon']
    # find unique lat/lon pairs
    latlon = np.array([lat, lon]).T
    latlon = np.unique(latlon, axis=0)

    kelp = np.zeros((len(times), len(latlon)))
    for i, t in tqdm(enumerate(times)):
        for j, ll in enumerate(latlon):
            kelp[i,j] = predict_kelp(t, ll[0], ll[1])

    # create xarray dataset
    # use xarray to save the data in the format
    # Coordinates:
    # * time        (time) datetime64[ns] 1984-02-15 1984-05-15 ... 2022-11-15
    # Dimensions without coordinates: station
    # Data variables: (12/13)
    #     latitude    (station) float64 ...
    #     longitude   (station) float64 ...
    #     year        (time) int32 1984 1984 1984 1984 1985 ... 2022 2022 2022 2022
    #     quarter     (time) int16 1 2 3 4 1 2 3 4 1 2 3 4 ... 1 2 3 4 1 2 3 4 1 2 3 4
    #     biomass     (time, station) float64 ...


    ds = xr.Dataset(
        data_vars = {
            'latitude': (['station'], latlon[:,0]),
            'longitude': (['station'], latlon[:,1]),
            'year': (['time'], times // 365 + 1984),
            'quarter': (['time'], times % 365 // 91 + 1),
            'biomass': (['time', 'station'], kelp)
        },
        coords = {
            'time': times
        }
    )
    ds.to_netcdf(args.file_path_sim.replace('.pkl', '_regressors.nc'))

    # email peter how to get:
    # how to get four ecopro projections before mid-feb