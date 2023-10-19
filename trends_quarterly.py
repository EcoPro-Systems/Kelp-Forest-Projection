import pickle
import argparse
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import stats

def correlation_tests(x, y):
    # measure the significance of the correlation
    correlations = {}

    # Pearson's correlation
    corr, pval = stats.pearsonr(x, y)
    correlations['pearsonr'] = {'corr': corr, 'pval': pval}

    # Kendall's tau
    tau, pval = stats.kendalltau(x, y)
    correlations['kendalltau'] = {'tau': tau, 'pval': pval}

    # Calculate the Spearman rank correlation
    corr, pval = stats.spearmanr(x, y)
    correlations['spearmanr'] = {'corr': corr, 'pval': pval}

    # Mann-Kendall
    tau, pval = stats.mstats.kendalltau(x, y)
    correlations['mann.kendall'] = {'tau': tau, 'pval': pval}

    # Linear regression
    slope, intercept, rval, pval, stderr = stats.linregress(x, y)
    correlations['linregress'] = {'slope': slope, 'intercept': intercept, 'rval': rval, 'pval': pval, 'stderr': stderr}

    # # ANOVA
    # fval, pval = stats.f_oneway(x, y) 
    # correlations['f_oneway'] = {'fval': fval, 'pval': pval}

    # # Kruskal-Wallis
    # hval, pval = stats.kruskal(x, y)
    # correlations['kruskal'] = {'hval': hval, 'pval': pval}

    # # Mann-Whitney U
    # uval, pval = stats.mannwhitneyu(x, y)
    # correlations['mannwhitneyu'] = {'uval': uval, 'pval': pval}

    # # Kolmogorov-Smirnov
    # dval, pval = stats.ks_2samp(x, y)
    # correlations['ks_2samp'] = {'dval': dval, 'pval': pval}

    return correlations

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
    time = time.astype(int) # number of days since min date
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

    # get unique times and bin data (quarterly)
    utime = np.unique(time)
    utime_dt = np.unique(time_dt)
    quarterly_kelp = np.zeros(len(utime)) # binned data
    quarterly_kelp_std = np.zeros(len(utime))
    quarterly_sst = np.zeros(len(utime))
    quarterly_sst_std = np.zeros(len(utime))
    
    # loop over each quarter and compute the mean and std
    for i, t in enumerate(utime):
        mask = time == t
        quarterly_kelp[i] = np.mean(data['kelp'][mask])
        quarterly_kelp_std[i] = np.std(data['kelp'][mask])
        quarterly_sst[i] = np.mean(data['temp'][mask])
        quarterly_sst_std[i] = np.std(data['temp'][mask])

    # float presentation of time
    starting_year = int(f"{time_dt.min().astype('datetime64[Y]')}")
    quarterly_time = starting_year + utime/365.

    # measure a seasonal trend line with OLS
    X = np.array([quarterly_time, np.ones(len(quarterly_time))]).T
    res = sm.OLS(quarterly_kelp, X).fit()
    coeffs = res.params
    y_kelp = np.dot(X, coeffs)

    # measure yearly trend line for SST
    res = sm.OLS(quarterly_sst, X).fit()
    coeffs_sst = res.params
    y_sst = np.dot(X, coeffs_sst)

    # measure yearly trend between sst and kelp
    X = np.array([quarterly_sst, np.ones(len(quarterly_sst))]).T
    res = sm.OLS(quarterly_kelp, X).fit()
    coeffs_sst_kelp = res.params
    y_sst_kelp = np.dot(X, coeffs_sst_kelp)

    # plot the data
    fig, ax = plt.subplots(1, 3, figsize=(12, 5))
    ax[0].errorbar(quarterly_time, quarterly_kelp, 
                   yerr=quarterly_kelp_std, fmt='o', ls='-', color='black', label='Quarterly Mean')
    ax[0].plot(quarterly_time, y_kelp, ls='-', color='red', label=f'OLS fit (slope: {coeffs[0]:.3f} m^2/year)')
    #ax[0].errorbar(utime_dt, bmean, yerr=bstd, fmt='o', color='red',alpha=0.25, label='Quarterly Mean')
    ax[0].set_xlabel("Year")
    ax[0].tick_params(axis='x', rotation=45)
    ax[0].set_ylabel("Kelp Area [m^2]")
    ax[0].grid(True,ls='--',alpha=0.5)
    ax[0].set_ylim([0,500])
    ax[0].legend(loc='best')

    ax[1].set_title("Annual Trends (avg. over 31-36N, 115-130W)")
    ax[1].errorbar(quarterly_time, quarterly_sst-273.15, 
                   yerr=quarterly_sst_std, fmt='o', ls='-', color='black', label='Quarterly Mean')
    ax[1].plot(quarterly_time, y_sst-273.15, ls='-', color='red', label=f'OLS fit (slope: {coeffs_sst[0]:.3f} C/year)')
    ax[1].set_xlabel("Year")
    # rotate tick labels 45 deg
    ax[1].tick_params(axis='x', rotation=45)
    ax[1].set_ylabel("Sea Surface Temperature [C]")
    #ax[1].set_title("SST vs. Time (avg. over 31-36N, 115-130W)")
    ax[1].grid(True,ls='--',alpha=0.5)
    ax[1].legend(loc='best')

    # create plot for sst vs kelp
    ax[2].plot(quarterly_sst-273.15, y_sst_kelp, ls='-', color='red', label=f'OLS fit (slope: {coeffs_sst_kelp[0]:.3f} m^2/C)')
    ax[2].scatter(quarterly_sst-273.15, quarterly_kelp, color='black', label='Quarterly Mean')
    ax[2].set_xlabel("Sea Surface Temperature [C]")
    ax[2].set_ylabel("Kelp Area [m^2]")
    ax[2].grid(True,ls='--',alpha=0.5)
    ax[2].legend(loc='best')

    plt.tight_layout()
    plt.savefig(args.file_path.replace('.pkl', '_quarterly.png'))

    # return p-vals for each correlation test
    alpha=0.05
    correlation_stats = {
        'SST vs. Kelp': correlation_tests(x = quarterly_sst-273.25, y = quarterly_kelp),
        'Time vs. SST': correlation_tests(x = quarterly_time, y = quarterly_sst-273.25),
        'Time vs. Kelp': correlation_tests(x = quarterly_time, y = quarterly_kelp),
    }

    # print out the results
    for key in correlation_stats:
        print(f"{key} Correlation tests for {args.file_path}")

        passed_metrics = 0
        # check for significance of trend
        for skey in correlation_stats[key]:
            # check for significance of trend
            if correlation_stats[key][skey]['pval'] < alpha:
                print(f"{key} is significant: {correlation_stats[key][skey]['pval']:.3f} for {skey}")
                passed_metrics += 1
            else:
                print(f"{key} is not significant: {correlation_stats[key][skey]['pval']:.3f} for {skey}")
        
        print(f"{passed_metrics} out of {len(correlation_stats[key])} metrics passed\n")

    plt.show()

    # stats for raw trend
    # stats_time_kelp = correlation_tests(x = time, y = data['kelp'])
    # stats_time_sst = correlation_tests(x = time, y = data['temp'])