import pickle
import argparse
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # argparse for input filepath
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file_path', type=str, 
                        help='path to input metrics file', 
                        default="Data/kelp_metrics_25_37.pkl")
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
    # construction features
    features = [
        time, # days, 0-365*20
        np.sin(2*np.pi*time/365), # -1 - 1
        np.cos(2*np.pi*time/365), # -1 - 1
        data['lat'], # 25-45
        data['lon'], # -130 - -115
        data['temp'], 
        data['temp_lag'],
        np.ones(len(time)) # w_1 * x_1 + w_2 * x_2 + ... + w_n * x_n + b
    ]

    X = np.array(features).T

    # remove nans
    nanmask = np.isnan(data['temp_lag'])
    X = X[~nanmask]
    y = y[~nanmask]
    time = time[~nanmask]
    time_dt = time_dt[~nanmask]

    # sort data in time
    si = np.argsort(time)
    X = X[si]
    y = y[si]
    time = time[si]
    time_dt = time_dt[si]

    # train test split with first 80% as training data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=False)
    time_train = time[:len(X_train)]
    time_test = time[len(X_train):]
    time_train_dt = time_dt[:len(X_train)]
    time_test_dt = time_dt[len(X_train):]

    # fit a linear model to the data with OLS
    res = sm.OLS(y_train, X_train).fit()
    coeffs = res.params
    y_ols_train = np.dot(X_train, coeffs)
    y_ols_test = np.dot(X_test, coeffs)

    # compute the average absolute error
    abs_err_ols_train = np.abs(y_train - y_ols_train).mean()
    print(f"Avg. Absolute Error (OLS) Train: {abs_err_ols_train:.3f} m^2")
    abs_err_ols_test = np.abs(y_test - y_ols_test).mean()
    print(f"Avg. Absolute Error (OLS) Test: {abs_err_ols_test:.3f} m^2")


    # standardize the input data
    Xp_train = X_train[:, :-1]
    Xp_train = (Xp_train - Xp_train.mean(0)) / Xp_train.std(0)
    Xp_test = X_test[:, :-1]
    Xp_test = (Xp_test - Xp_test.mean(0)) / Xp_test.std(0)

    # create multi-layer perceptron regressor
    mlp = MLPRegressor(hidden_layer_sizes=(30,8,4), activation='relu', solver='adam', max_iter=50, verbose=True, alpha=0.1)
    y_mlp_train = mlp.fit(Xp_train, y_train).predict(Xp_train)
    abs_err_mlp_train = np.abs(y_train - y_mlp_train).mean()
    print(f"Avg. Absolute Error (MLP) train: {abs_err_mlp_train:.3f} m^2")
    y_mlp_test = mlp.predict(Xp_test)
    abs_err_mlp_test = np.abs(y_test - y_mlp_test).mean()
    print(f"Avg. Absolute Error (MLP) test: {abs_err_mlp_test:.3f} m^2")


    # fit data with decision tree regressor
    dt = RandomForestRegressor(max_depth=3, min_samples_split=2, min_samples_leaf=2, random_state=0)
    y_dt_train = dt.fit(X_train, y_train).predict(X_train)
    abs_err_dt_train = np.abs(y_train - y_dt_train).mean()
    print(f"Avg. Absolute Error (RF) train: {abs_err_dt_train:.3f} m^2")
    y_dt_test = dt.predict(X_test)
    abs_err_dt_test = np.abs(y_test - y_dt_test).mean()
    print(f"Avg. Absolute Error (RF) test: {abs_err_dt_test:.3f} m^2")


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

    utime_train = np.unique(time_train)
    utime_train_dt = np.unique(time_train_dt)
    mean_ols_train = np.zeros(len(utime_train)) # binned data OLS
    std_ols_train = np.zeros(len(utime_train))
    mean_mlp_train = np.zeros(len(utime_train)) # binned data MLP
    std_mlp_train = np.zeros(len(utime_train))
    mean_dt_train = np.zeros(len(utime_train)) # binned data DT
    std_dt_train = np.zeros(len(utime_train))

    # loop over each quarter and compute the mean and std
    for i, t in enumerate(utime_train):
        mask = time_train == t
        mean_ols_train[i] = np.mean(y_ols_train[mask])
        std_ols_train[i] = np.std(y_ols_train[mask])
        mean_mlp_train[i] = np.mean(y_mlp_train[mask])
        std_mlp_train[i] = np.std(y_mlp_train[mask])
        mean_dt_train[i] = np.mean(y_dt_train[mask])
        std_dt_train[i] = np.std(y_dt_train[mask])

    utime_test = np.unique(time_test)
    utime_test_dt = np.unique(time_test_dt)
    mean_ols_test = np.zeros(len(utime_test)) # binned data OLS
    std_ols_test = np.zeros(len(utime_test))
    mean_mlp_test = np.zeros(len(utime_test)) # binned data MLP
    std_mlp_test = np.zeros(len(utime_test))
    mean_dt_test = np.zeros(len(utime_test)) # binned data DT
    std_dt_test = np.zeros(len(utime_test))

    # loop over each quarter and compute the mean and std
    for i, t in enumerate(utime_test):
        mask = time_test == t
        mean_ols_test[i] = np.mean(y_ols_test[mask])
        std_ols_test[i] = np.std(y_ols_test[mask])
        mean_mlp_test[i] = np.mean(y_mlp_test[mask])
        std_mlp_test[i] = np.std(y_mlp_test[mask])
        mean_dt_test[i] = np.mean(y_dt_test[mask])
        std_dt_test[i] = np.std(y_dt_test[mask])


    # plot the data
    fig, ax = plt.subplots(3, 1, figsize=(11, 10))

    ax[0].errorbar(utime_dt, bmean, yerr=bstd, fmt='o', color='black', label='binned data')
    ax[0].errorbar(utime_train_dt, mean_ols_train, yerr=std_ols_train, fmt='.', ls='none', color='blue', label=f'train (avg. err: {abs_err_ols_train:.1f} m^2)')
    ax[0].errorbar(utime_train_dt, mean_ols_train, yerr=std_ols_train, fmt='.', ls='-',alpha=0.33, color='blue')
    ax[0].errorbar(utime_test_dt, mean_ols_test, yerr=std_ols_test, fmt='o', ls='none', color='red', label=f'test (avg. err: {abs_err_ols_test:.1f} m^2)')
    ax[0].errorbar(utime_test_dt, mean_ols_test, yerr=std_ols_test, fmt='.', ls='--',alpha=0.33, color='red')
    ax[0].legend(loc='upper left')
    #ax[0].set_xlabel("Time")
    ax[0].set_ylabel("Kelp Area [m^2]")
    ax[0].set_title("Ordinary Least Squares (OLS) Regression")
    ax[0].grid(True,ls='--',alpha=0.5)
    ax[0].set_ylim([0,666])

    # MLP timeseries
    ax[1].errorbar(utime_dt, bmean, yerr=bstd, fmt='o', color='black', label='binned data')
    ax[1].errorbar(utime_train_dt, mean_mlp_train, yerr=std_mlp_train, fmt='.', ls='none', color='blue', label=f'train (avg. err: {abs_err_mlp_train:.1f} m^2)')
    ax[1].errorbar(utime_train_dt, mean_mlp_train, yerr=std_mlp_train, fmt='.', ls='-',alpha=0.33, color='blue')
    ax[1].errorbar(utime_test_dt, mean_mlp_test, yerr=std_mlp_test, fmt='o', ls='none', color='red', label=f'test (avg. err: {abs_err_mlp_test:.1f} m^2)')
    ax[1].errorbar(utime_test_dt, mean_mlp_test, yerr=std_mlp_test, fmt='.', ls='--',alpha=0.33, color='red')
    ax[1].legend(loc='upper left')
    #ax[1].set_xlabel("Time")
    ax[1].set_ylabel("Kelp Area [m^2]")
    ax[1].set_title("Multi-Layer Perceptron (MLP) Regression")
    ax[1].grid(True,ls='--',alpha=0.5)
    ax[1].set_ylim([0,666])


    # DT timeseries
    ax[2].errorbar(utime_dt, bmean, yerr=bstd, fmt='o', color='black', label='binned data')
    ax[2].errorbar(utime_train_dt, mean_dt_train, yerr=std_dt_train, fmt='.', ls='none', color='blue', label=f'train (avg. err: {abs_err_dt_train:.1f} m^2)')
    ax[2].errorbar(utime_train_dt, mean_dt_train, yerr=std_dt_train, fmt='.', ls='-',alpha=0.33, color='blue')
    ax[2].errorbar(utime_test_dt, mean_dt_test, yerr=std_dt_test, fmt='o', ls='none', color='red', label=f'test (avg. err: {abs_err_dt_test:.1f} m^2)')
    ax[2].errorbar(utime_test_dt, mean_dt_test, yerr=std_dt_test, fmt='.', ls='--',alpha=0.33, color='red')

    ax[2].legend(loc='upper left')
    ax[2].set_xlabel("Time")
    ax[2].set_ylabel("Kelp Area [m^2]")
    ax[2].set_title("Random Forest (RF) Regression")
    ax[2].grid(True,ls='--',alpha=0.5)
    ax[2].set_ylim([0,666])

    #DT Text
#     train_representation = tree.export_text()
#     print(train_representation)
#     test_representation = tree.export_text(y_dt_test)
#     print(test_representation)
    
    #DT tree
#     fig = plt.figure(figsize=(25,20))
#     _ = tree.plot_tree(rf, feature_names=features, filled=True)
    
    plt.tight_layout()
    plt.savefig(args.file_path.replace('.pkl', '_regressors.png'))
    plt.show()
