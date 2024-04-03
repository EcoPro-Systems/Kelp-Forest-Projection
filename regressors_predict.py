import joblib
import argparse
import numpy as np
import xarray as xr
from tqdm import tqdm
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d
from sklearn.neural_network import MLPRegressor
from create_interpolated_sst_sim import NearestNeighbor

if __name__ == "__main__":
    # argparse for input filepath
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file_path', type=str, 
                        help='path to input metrics file', 
                        default="Data/kelp_metrics_27_37.pkl")
    parser.add_argument('-fs', '--file_path_sim', type=str, 
                        help='path to input metrics file', 
                        default="Data/kelp_metrics_sim_27_37_GFDL-ESM4_ssp585_BGL.pkl")
    #model type
    parser.add_argument('-m', '--model', type=str, 
                        help='model type (OLS or MLP)',
                        default="OLS")
    # sunlight bool default false
    parser.add_argument('-s', '--sunlight', action='store_true',
                        help='use sunlight as input feature')
    
    args = parser.parse_args()

    # extract climate model, scenario, and scaling from file_path_sim
    parts = args.file_path_sim.split('_')
    climate_scenario = parts[-2] # ssp585
    climate_model = parts[-3] # GFDL-ESM4
    scaling = parts[-1].split('.')[0]

    # load data from disk
    with open(args.file_path, 'rb') as f:
        data = joblib.load(f)

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

    if args.sunlight:
        features = [
            #time, # days, 0-365*20
            data['sunlight'], # SUNLIGHT
            data['temp'] - 273.15,
            data['temp_lag']-273.15,
            data['temp_lag2']-273.15,
            np.ones(len(time)) # w_1 * x_1 + w_2 * x_2 + ... + w_n * x_n + b
        ]

        feature_names = [
            #'time',
            'sunlight', # SUNLIGHT
            'temp',
            'temp_lag',
            'temp_lag2',
            'bias'
        ]
    else:
        features = [
            #time, # days, 0-365*20
            data['temp'] - 273.15,
            data['temp_lag']-273.15,
            data['temp_lag2']-273.15,
            np.ones(len(time)) # w_1 * x_1 + w_2 * x_2 + ... + w_n * x_n + b
        ]

        feature_names = [
            #'time',
            'temp',
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
        data_sim = joblib.load(f)

    # convert datetime64[ns] to days since min date 
    time_sim = data_sim['time'].astype('datetime64[D]')
    time_sim = time_sim - np.min(time)
    time_sim = time_sim.astype(int)
    time_sim_dt = data_sim['time'] # datetime format

    if args.sunlight:

        # construct features
        features = [
            #time_sim, # days
            data_sim['sunlight'], # SUNLIGHT
            data_sim['temp'],
            data_sim['temp_lag'],
            data_sim['temp_lag2'],
            np.ones(len(time_sim)) # w_1 * x_1 + w_2 * x_2 + ... + w_n * x_n + b
        ]

        feature_names = [
            #'time',
            'sunlight', # SUNLIGHT
            'temp',
            'temp_lag',
            'temp_lag2',
            'bias'
        ]
    else:
        features = [
            #time_sim, # days
            data_sim['temp'],
            data_sim['temp_lag'],
            data_sim['temp_lag2'],
            np.ones(len(time_sim)) # w_1 * x_1 + w_2 * x_2 + ... + w_n * x_n + b
        ]

        feature_names = [
            #'time',
            'temp',
            'temp_lag',
            'temp_lag2',
            'bias'
        ]

    X_test = np.array(features).T

    # remove nans
    nanmask_test = np.isnan(data_sim['temp_lag']) | np.isnan(data_sim['temp_lag2']) | np.isnan(data_sim['temp'])
    X_test = X_test[~nanmask_test]
    time_sim = time_sim[~nanmask_test]
    time_sim_dt = time_sim_dt[~nanmask_test]
    print(f"Samples: {len(X)} Train, {len(X_test)} Test")

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
        #res = sm.OLS(y, X).fit()
        # add some regularization
        res = sm.OLS(y, X).fit_regularized(alpha=0.001, L1_wt=0.1)
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

        mlp = MLPRegressor(hidden_layer_sizes=(4,2), max_iter=150, alpha=0.001, solver='adam', verbose=True)
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

    # slope between lag_temperature and kelp
    #slope_kelp, intercept_kelp = np.polyfit(np.roll(mean_sst_train-273.15, 1)[1:], mean_ols_train[1:], 1)
    # mur trend
    slope_kelp, intercept_kelp = np.polyfit(np.roll(mean_sst_train-273.15, 1)[1:], bmean[1:], 1)

    print(f"Slope: {slope_kelp:.5f} m^2 / C")

    # mean_sst_test
    kelp_projection = mean_sst_test * slope_kelp  + intercept_kelp
    kelp_train = (mean_sst_train-273.15) * slope_kelp + intercept_kelp

    # MSE kelp + ols
    mse_kelp = np.mean(np.abs(bmean - mean_ols_train))
    mse_projection = np.mean(np.abs(bmean - kelp_projection[:72]))
    mse_test = np.mean(np.abs(bmean - mean_ols_test[:72]))
    mse_train = np.mean(np.abs(bmean - kelp_train))

    # plot the data
    fig, ax = plt.subplots(3, 1, figsize=(11, 10))
    # lat lon limits to title
    # extract lat/lon from file_path
    lat = float(args.file_path_sim.split('_')[3])
    lon = float(args.file_path_sim.split('_')[4].split('.')[0])
    # extract type of model from fs
    fig.suptitle(f"Kelp Projections at {lat:.0f}-{lon:.0f} N using {climate_model} {climate_scenario.upper()} {scaling.upper()}", fontsize=16)
    ax[0].errorbar(utime_dt, bmean, yerr=bstd, fmt='o', color='black', label='Kelp Watch Data',alpha=0.90)

    #ax[0].plot(utime_test_dt, mean_ols_test, ls='-', color='red', label=f'Projections (avg. err: {mse_test:.1f} m$^2$)',alpha=0.9)
    #ax[0].errorbar(utime_train_dt, mean_ols_train, yerr=std_ols_train, fmt='.', ls='-', color='limegreen', label=rf'{args.model.upper()} Model (avg. err: {abs_err_ols_train:.1f} m$^2$)')
    ax[0].plot(utime_test_dt[1:], kelp_projection[:-1], ls='-', color='limegreen', label=f'Climate Model (avg. err: {mse_projection:.1f} m$^2$)')
    ax[0].plot(utime_train_dt[1:], kelp_train[:-1],  ls='-', color='red', label=f'MUR Model (avg. err: {mse_train:.1f} m$^2$)')
    ax[0].legend(loc='upper left')
    ax[0].set_ylabel(r"Average Kelp Area per Station [m$^2$]")
    ax[0].grid(True,ls='--',alpha=0.5)
    ax[0].set_ylim([0,666])
    ax[0].set_xlim([np.min(utime_dt), np.max(utime_dt)])

    #ax[1].plot(utime_dt, bmean, ls='-', color='black', label='Kelp Watch Data',alpha=0.90)
    #ax[1].plot(utime_test_dt, mean_ols_test, ls='-', color='red', label='Projections',alpha=0.9)
    #ax[1].plot(utime_train_dt, mean_ols_train,  ls='-', color='limegreen', label=f'{args.model.upper()} Model')
    ax[1].plot(utime_train_dt[1:], kelp_train[:-1], ls='-', color='red', label=f'MUR Model')
    ax[1].plot(utime_test_dt[1:], kelp_projection[:-1], ls='-', color='limegreen', label=f'Climate Model')
    ax[1].set_ylim([0,666])
    ax[1].grid(True,ls='--',alpha=0.5)
    ax[1].set_xlabel("Time")
    ax[1].set_ylabel(r"Average Kelp Area per Station [m$^2$]")
    ax[1].legend(loc='upper left')
    ax[1].set_xlim([np.min(utime_test_dt), np.max(utime_test_dt)])

    # plot temperature time series for each location
    ax[2].plot(utime_test_dt, mean_sst_test, 'c-', label=f'{climate_model} {climate_scenario.upper()} {scaling.upper()}')
    ax[2].plot(utime_train_dt, mean_sst_train-273.15, 'k-', label=f'JPL MUR Data')
    ax[2].set_xlabel('Time')
    ax[2].set_ylabel('Sea Surface Temperature [C]')
    ax[2].legend()
    ax[2].grid(True,ls='--',alpha=0.5)
    ax[2].set_xlim([np.min(utime_test_dt), np.max(utime_test_dt)])
    ax[2].set_ylim([13,22.5])
    plt.tight_layout()
    file_name = args.file_path_sim.replace('.pkl', '_regressors.png')
    file_name = file_name.replace('metrics', args.model)
    if args.sunlight:
        file_name = file_name.replace('regressors', 'sunlight_regressors')
    plt.savefig(file_name)
    print(f"Saved {file_name}")
    plt.close()

    # TODO create a fill between region for projection uncertainties
    # TODO add slopes to the plot

    # create nearest neighbor algorithm to map time, lat, lon to kelp
    sim_lat = data_sim['lat'][~nanmask_test]
    sim_lon = data_sim['lon'][~nanmask_test]
    sim_sunlight = data_sim['sunlight'][~nanmask_test]
    sim_temp = data_sim['temp'][~nanmask_test]
    sim_temp_lag = data_sim['temp_lag'][~nanmask_test]
    sim_temp_lag2 = data_sim['temp_lag2'][~nanmask_test]

    # days since min date
    times_dt = np.unique(data_sim['time']).astype('datetime64[D]')
    times = times_dt - np.min(times_dt)
    times = times.astype(int)

    lat = data_sim['lat']
    lon = data_sim['lon']
    # find unique lat/lon pairs
    latlon = np.array([lat, lon]).T
    latlon = np.unique(latlon, axis=0)

    temp = np.zeros((len(times), len(latlon)))

    try:
        # faster to interpolate from netcdf file then to use KDTree
        sim_data = xr.open_dataset(f"Data/tos_Omon_{climate_model}_{climate_scenario}_r1i1p1f1_gr_2002-2100.downscaled_{scaling}.unique.nc", decode_times=False)
        sim_times =  np.datetime64('1900-01-16T12:00:00') + np.array(sim_data.time.values, dtype='timedelta64[h]')
        sim_sst = NearestNeighbor(sim_data.lat.values, sim_data.lon.values, sim_data.sst.values)
        sim_data.close()

        sim_times_day = sim_times.astype('datetime64[D]')
        sim_times_day = sim_times_day - np.min(times_dt)
        sim_times_day = sim_times_day.astype(int)
    
        # interpolate temperatures
        for j, ll in tqdm(enumerate(latlon)):
            sim_temps = sim_sst(ll[0], ll[1])

            # second order interpolation
            f_temp = interp1d(sim_times_day, sim_temps, kind='quadratic', fill_value='extrapolate')
            temp[:,j] = f_temp(times)

    except Exception as ex:
        print(ex)
        print("Using KDTree to interpolate temperature")
        # nearest neighbor interpolation
        tree_temp = cKDTree(np.array([time_sim, sim_lat, sim_lon]).T)
        def predict_temp(time, lat, lon):
            dist, idx = tree_temp.query(np.array([time, lat, lon]).T)
            return sim_temp[idx]

        for i, t in tqdm(enumerate(times)):
            for j, ll in enumerate(latlon):
                temp[i,j] = predict_temp(t, ll[0], ll[1])

    # use temperature to predict kelp
    print(len(times), len(latlon))
    kelp = np.zeros((len(times), len(latlon)))
    # loop over locations
    for j, ll in tqdm(enumerate(latlon)):
        # create lagged temperature
        temp_lag = np.roll(temp[:,j], 1)
        # average data from different years but same quarter
        temp_lag[0] = (temp[0,j] + temp[4,j] )/ 2
        # predict kelp
        kelp[:, j] = slope_kelp * temp[:,j] + intercept_kelp

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
    #     temp        (time, station) float64 ...

    ds = xr.Dataset(
        data_vars = {
            'latitude': (['station'], latlon[:,0]),
            'longitude': (['station'], latlon[:,1]),
            'year': (['time'], times_dt.astype('datetime64[Y]').astype(int)),
            'quarter': (['time'], times_dt.astype('datetime64[M]').astype(int) % 12 // 3 + 1),
            'biomass': (['time', 'station'], kelp),
            'temp': (['time', 'station'], temp),
            'slope': (['feature'], [slope_kelp]),
            'intercept': (['feature'], [intercept_kelp]),
        },
        coords = {
            'time': times_dt
        }
    )

    ds.to_netcdf(args.file_path_sim.replace('.pkl', '_regressors.nc'))