import joblib
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
                        default="Data/kelp_metrics_sim_27_30.pkl")
    args = parser.parse_args()

    file_path = args.file_path.replace('.pkl', '')
    region = f"{file_path.split('_')[3]}-{file_path.split('_')[4]}N"

    # load data from disk
    with open(args.file_path, 'rb') as f:
        data = joblib.load(f)

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
    quarterly_sst = np.zeros(len(utime))
    quarterly_sst_std = np.zeros(len(utime))
    
    # loop over each quarter and compute the mean and std
    for i, t in enumerate(utime):
        mask = (time == t) & (~np.isnan(data['temp_lag']))
        quarterly_sst[i] = np.nanmean(data['temp_lag'][mask])
        quarterly_sst_std[i] = np.std(data['temp_lag'][mask])

    # remove first quarter for lag nan
    quarterly_sst = quarterly_sst[1:]
    quarterly_sst_std = quarterly_sst_std[1:]

    # float presentation of time
    starting_year = int(f"{time_dt.min().astype('datetime64[Y]')}")
    quarterly_time = starting_year + utime/365.
    quarterly_time = quarterly_time[1:]

    # measure a seasonal trend line with OLS
    X = np.array([quarterly_time, np.ones(len(quarterly_time))]).T
    # measure yearly trend line for SST
    res = sm.OLS(quarterly_sst, X).fit()
    coeffs_sst = res.params
    y_sst = np.dot(X, coeffs_sst)
    # print slope +- error
    print(f"Slope of trend line: {coeffs_sst[0]:.2f} +- {res.bse[0]:.2f} C/year")
    # monte carlo to find year at which temp reaches 23.47 +- 2.11C
    qtimes = []
    for i in range(10000):
        # sample from normal distribution
        sst = np.random.normal(loc=23.47, scale=2.11)
        # calculate time at which sst is equal to sst
        qtime = (sst - coeffs_sst[1])/coeffs_sst[0]
        qtimes.append(qtime)
    print(f"Time at which SST reaches 23.47 +- 2.11C: {np.mean(qtimes):.2f} +- {np.std(qtimes):.2f} years")

    # measure yearly trend between sst and kelp
    X = np.array([quarterly_sst, np.ones(len(quarterly_sst))]).T

    # plot the data
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))

    ax.set_title(f"Annual Trends (avg. over {region})")
    ax.errorbar(quarterly_time, quarterly_sst, 
                   yerr=quarterly_sst_std, fmt='.', ls='-', color='black', label='Quarterly Mean')
    ax.plot(quarterly_time, y_sst, ls='-', color='red', label=f'OLS fit (slope: {coeffs_sst[0]:.3f} C/year)')
    ax.set_xlabel("Year")
    # rotate tick labels 45 deg
    ax.tick_params(axis='x', rotation=45)
    ax.set_ylabel("Sea Surface Temperature [C]")
    #ax[1].set_title("SST vs. Time (avg. over 31-36N, 115-130W)")
    ax.grid(True,ls='--',alpha=0.5)
    ax.legend(loc='best')

    plt.tight_layout()
    plt.savefig(args.file_path.replace('.pkl', '_quarterly_lag.png'))

    # return p-vals for each correlation test
    alpha=0.05
    correlation_stats = {
        'Time vs. SST': correlation_tests(x = quarterly_time, y = quarterly_sst-273.25),
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