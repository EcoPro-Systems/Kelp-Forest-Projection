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
                        default="Data/kelp_metrics_31_36.pkl")
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

    # average data into yearly bins
    """
    data['time'] = array(['2016-08-15T00:00:00.000000000', '2016-11-15T00:00:00.000000000',
       '2017-11-15T00:00:00.000000000', ...,
       '2016-08-15T00:00:00.000000000', '2019-05-15T00:00:00.000000000',
       '2019-08-15T00:00:00.000000000'], dtype='datetime64[ns]')
    """

    # get the unique years
    years = np.unique(data['time'].astype('datetime64[Y]'))
    yearly_kelp = np.zeros(len(years))
    yearly_std = np.zeros(len(years))
    yearly_sst = np.zeros(len(years))
    yearly_sst_std = np.zeros(len(years))

    # loop over each year and compute the mean
    for i, y in enumerate(years):
        mask = data['time'].astype('datetime64[Y]') == y
        yearly_kelp[i] = np.mean(data['kelp'][mask])
        yearly_std[i] = np.std(data['kelp'][mask])
        yearly_sst[i] = np.mean(data['temp'][mask])
        yearly_sst_std[i] = np.std(data['temp'][mask])

    # get unique times and bin data
    utime = np.unique(time)
    utime_dt = np.unique(time_dt)
    bmean = np.zeros(len(utime)) # binned data
    bstd = np.zeros(len(utime))
    
    # loop over each quarter and compute the mean and std
    for i, t in enumerate(utime):
        mask = time == t
        bmean[i] = np.mean(data['kelp'][mask])
        bstd[i] = np.std(data['kelp'][mask])

    # measure a yearly trend line with OLS
    time_yearly = np.arange(len(years))
    X = np.array([time_yearly, np.ones(len(years))]).T
    res = sm.OLS(yearly_kelp, X).fit()
    coeffs = res.params
    y_ols = np.dot(X, coeffs)

    # measure yearly trend line for SST
    res = sm.OLS(yearly_sst, X).fit()
    coeffs_sst = res.params
    y_sst = np.dot(X, coeffs_sst)

    # plot the data
    fig, ax = plt.subplots(2, 1, figsize=(11, 8))
    ax[0].errorbar(years.astype('datetime64[Y]')+np.timedelta64(6,'M'),
                yearly_kelp, yerr=yearly_std, fmt='o', ls='-', color='black', label='Yearly Mean')
    ax[0].plot(years.astype('datetime64[Y]')+np.timedelta64(6,'M'), y_ols, ls='--', color='green', label=f'OLS fit (slope: {coeffs[0]:.3f} m^2/year)')
    ax[0].errorbar(utime_dt, bmean, yerr=bstd, fmt='o', color='red',alpha=0.25, label='Quarterly Mean')
    ax[0].set_xlabel("Time")
    ax[0].set_ylabel("Kelp Area [m^2]")
    ax[0].set_title("Kelp Area vs. Time (avg. over 31-36N, 115-130W)")
    ax[0].grid(True,ls='--',alpha=0.5)
    ax[0].set_ylim([0,666])
    ax[0].legend(loc='upper left')

    ax[1].errorbar(years.astype('datetime64[Y]')+np.timedelta64(6,'M'),
                   yearly_sst-273.15, yerr=yearly_sst_std, fmt='o', ls='-', color='black', label='Yearly Mean')
    ax[1].plot(years.astype('datetime64[Y]')+np.timedelta64(6,'M'), y_sst-273.15, ls='--', color='green', label=f'OLS fit (slope: {coeffs_sst[0]:.3f} C/year)')
    ax[1].set_xlabel("Time")
    ax[1].set_ylabel("Sea Surface Temperature [C]")
    ax[1].set_title("SST vs. Time (avg. over 31-36N, 115-130W)")
    ax[1].grid(True,ls='--',alpha=0.5)
    ax[1].legend(loc='upper left')

    plt.tight_layout()
    plt.savefig(args.file_path.replace('.pkl', '_annual.png'))
    plt.show()

    import numpy as np 
    import scipy.stats as stats

    x = yearly_sst-273.25
    y = yearly_kelp

    # Pearson's correlation
    corr, pval = stats.pearsonr(x, y)

    # Interpret the p-value
    alpha = 0.05
    if pval < alpha:
        # create simple description
        print('The correlation is significant %.3f, (pearsonr)' % (pval))

    # Kendall's tau
    tau, pval = stats.kendalltau(x, y) 
    if pval < alpha:
        # create simple description
        print('The correlation is significant %.3f, (kendalltau)' % (pval))


    # Calculate the Spearman rank correlation
    corr, pvalue = stats.spearmanr(x, y)
    if pval < alpha:
        # create simple description
        print('The correlation is significant %.3f, (spearmanr)' % (pval))

    # Mann-Kendall
    tau, pval = stats.mstats.kendalltau(y)
    if pval < alpha:
        # create simple description
        print('The correlation is significant %.3f, (mstats.kendalltau)' % (pval))

    # Linear regression
    slope, intercept, rval, pval, stderr = stats.linregress(x, y)
    if pval < alpha:
        # create simple description
        print('The correlation is significant %.3f, (linregress)' % (pval))

    # ANOVA
    fval, pval = stats.f_oneway(x, y) 
    if pval < alpha:
        # create simple description
        print('The correlation is significant %.3f, (f_oneway)' % (pval))
