import numpy as np
import joblib
import argparse
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # argparse for input filepath
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file_path', type=str, 
                        help='path to input metrics file', 
                        default="Data/kelp_metrics_31_36.pkl")
    args = parser.parse_args()

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

    # Define the parameter grid
    param_grid = {
        'hidden_layer_sizes': [(32,8), (32,), (32,8,4), (32,8,8), (32,32), (5,5,5), (8,8,8)], 
        'activation': ['relu', 'tanh', 'logistic'],
        'solver': ['sgd', 'adam'],
        'learning_rate': ['constant','adaptive'],
        'alpha': [0.0001, 0.001, 0.01, 0.1], 
    }

    # Create MLP regressor  
    mlp = MLPRegressor(max_iter=20)

    # Set up GridSearchCV
    clf = GridSearchCV(mlp, param_grid, cv=8, scoring='neg_mean_squared_error', verbose=1)

    # Fit the model
    clf.fit(X, y)

    # Print out best parameters  
    print(clf.best_params_)