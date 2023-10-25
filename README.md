# Kelp Forest Monitoring

An ecological forecasting model for monitoring the biomass availability in kelp forests on the California coast. 

![](Figures/kelp_forest.jpg)

## Installation + Setup

1) Clone repo + setup a conda environment

```
conda create -n kelp python=3.10
conda activate kelp
conda install ipython jupyter pandas matplotlib scipy scikit-learn
conda install -c conda-forge xarray dask netCDF4 bottleneck
pip install tqdm statsmodels astropy
```

2) Download larges files from git lfs: 
    - `git lfs install`
    - `git lfs pull`

2) Download interpolated SST values
    - Go to https://hub.jpl-cmda.org
    - Navigate to `shared/notebooks/Kelp_Biomass/`
    - Download [kelp_interpolated_data.pkl (~3GB)](https://hub.jpl-cmda.org/user/kpearson/files/shared/notebooks/Kelp_Biomass/kelp_interpolated_data.pkl)

The interpolated data comes from `create_interpolated_sst.py` and interpolates the monthly SST data onto a quarterly grid to match the Kelp data. The interpolated data is saved as a pickle file and is used in `kelp_metrics.py` to calculate the lag temperatures and derivatives for a given latitude range.

3) Create some metrics/features, set the `lower_lat` and `upper_lat` variables in the script to change where the metrics are calculated
    - `python kelp_metrics.py`

The metrics script will create a new file called `kelp_metrics.pkl` which contains features for our regression algorithm.

4) Train a regression model (includes OLS, MLP and RF)
    - `python train_regressors.py`

## Analysis Scripts

The python scripts can be run locally and the jupyter notebooks are meant to be run on the [CMDA server](https://hub.jpl-cmda.org).

| Script Name | Description | 
| ----------- | ----------- |
| `create_interpolated_sst.py`, `create_interpolated_sst.ipynb` | Interpolate the monthly SST data onto the same grid as the kelp data and create a new file called: `kelp_interpolated_data.pkl` |
| `kelp_metrics.py`, `Kelp_Metrics.ipynb` | Calculate various metrics like lag temps and derivatives for each kelp location then save the data to a new file called: `kelp_metrics.pkl`. These metrics are used as features for our regression algorithm. |
| `plot_timeseries.py` | Create time series plots for temperature and abundance using `kelp_metrics.pkl`, averages over the entire region. 
| ![](Data/kelp_metrics_31_36_sst_timeseries.png) | ![](Data/kelp_metrics_31_36_kelp_timeseries.png) |
| `plot_histogram_sst.py` `plot_histogram_kelp.py`   | Create seasonal histograms for change in abundance using `kelp_metrics.pkl`, averages over the entire region. |
| ![](Data/kelp_metrics_31_36_histogram_sst.png) | ![](Data/kelp_metrics_31_36_histogram.png) |
| `plot_lag_correlation.py` `plot_lag_correlation_change.py` | Create lag correlation plots for temperature and abundance using `kelp_metrics.pkl`, averages over the entire region. |
| ![](Data/kelp_metrics_31_36_lag_correlation.png) | ![](Data/kelp_metrics_31_36_lag_correlation_change.png) |
| `trends_annual.py` | Calculate the annual trends for kelp abundance and temperature using `kelp_metrics.pkl`. Also, measures significance of trends with various pval estimates (e.g. pearsonr, Mann-Kendall, Kendall-Tau, ANOVA, Spearmanr, etc.) |
| ![]() | ![](Data/kelp_metrics_31_36_annual.png) |
| `trends_quarterly.py` | Calculate the seasonal trends (quarterly) for kelp abundance and temperature using `kelp_metrics.pkl`. Also, measures significance of trends with various pval estimates (e.g. pearsonr, Mann-Kendall, Kendall-Tau, ANOVA, Spearmanr, etc.) |
| Trend with previous quarter's temperature ![](Data/kelp_metrics_31_36_quarterly_lag.png) | Trend with current quarter's temperature ![](Data/kelp_metrics_31_36_quarterly.png) | 
| `regressors_optimize.py` | Hyperparameter optimization for regression algorithms using scikit-learn | 
| `regressors_train.py` | Train various regression models to predict the abundance of Kelp using ordinary least-squares, multi-layer perceptron and random forest with features from `kelp_metrics.pkl`. |
| ![]() | ![](Data/kelp_metrics_31_36_regressors.png) |
| `regressors_predict.py` | Coming soon... |

# Tests for statistical significance

| Annual P-Vals | SST vs. Kelp | Time vs. SST | Time vs. Kelp |
|--------------|--------------|--------------|--------------|
| [Pearsonr](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html)    | 0.149        | 0.001        | 0.053        |
| [Kendalltau](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kendalltau.html)  | 0.164        | 0.000        | 0.080        |
| [Spearmanr](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html)   | 0.198        | 0.000        | 0.058        |
| [Mann.Kendall](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mstats.kendalltau.html) | 0.164        | 0.000        | 0.080        |
| [Linregress](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html)  | 0.149        | 0.001        | 0.053        | 

Signficant p-vals are less than ~0.05-0.1, where smaller values are more significant. These values are measured in the `trends_annual.py` script.

| Quarterly P-vals| SST vs. Kelp | Time vs. SST | Time vs. Kelp |
|-----------------|--------------|--------------|---------------|
| Pearsonr        | 0.000        | 0.033        | 0.166         |
| Kendalltau      | 0.009        | 0.051        | 0.213         |
| Spearmanr       | 0.003        | 0.041        | 0.202         | 
| Mann.Kendall    | 0.009        | 0.051        | 0.213         |
| Linregress      | 0.000        | 0.033        | 0.166         |


One thing these tables don't show is the direction of the correction but from the quarterly plots above we see an inverse trend between Kelp and SST from a quarter before.

| Quarterly P-vals | SST_lag vs. Kelp | Time vs. SST_lag | Time vs. Kelp |
|------------------|------------------|------------------|--------------|
| Pearsonr         | 0.000            | 0.094            | 0.295        |
| Kendalltau       | 0.000            | 0.092            | 0.346        |
| Spearmanr        | 0.000            | 0.065            | 0.333        |
| Mann.Kendall     | 0.000            | 0.092            | 0.346        |
| Linregress       | 0.000            | 0.094            | 0.295        |

### Pearson's correlation:
- Measures the linear relationship between two continuous variables 
- Produces a correlation coefficient (r) from -1 to +1
   - +1 is a perfect positive linear relationship
   - -1 is a perfect negative linear relationship 
- Also provides a p-value to test if the correlation is statistically significant

### Kendall's tau:
- Nonparametric test that looks at concordant and discordant pairs
- Concordant pairs are when both values increase or decrease together 
- Discordant pairs are when one value increases while the other decreases
- Tau coefficient indicates the net correlation of the concordant/discordant pairs
- Tau close to +1/-1 indicates a strong monotonic trend

### Mann-Kendall: 
- Nonparametric test specifically for monotonic upward or downward trends over time
- Compares each data point to all subsequent points 
- Positive differences indicate an upward trend, negative differences indicate downward 
- Statistical test on these differences determines if the trend is significant


### Spearman's Rank Correlation:
- Nonparametric measure of correlation between two variables
- Assesses monotonic relationship rather than linear 
- Converts data values to ranks before calculating coefficient
- Coefficient (r) ranges from -1 to +1
- Positive r means variables increase together 
- Negative r means one increases as other decreases
- r near zero means little to no association
- r of +1/-1 indicates perfect monotonic relationship
- Tests significance of r using a hypothesis test
- Produces a p-value to determine if correlation is significant
- Useful for non-normal distributions or nonlinear relationships
- Simple to calculate and interpret
- Less sensitive to outliers compared to Pearson correlation

### Linear Regression analysis:
- Fits a straight line model to the data 
- Tests if slope coefficient is significantly different than zero
- Slope significantly greater than zero indicates increasing trend
- Slope significantly less than zero indicates decreasing trend

# Predictions with Machine Learning

We test three different regression algorithms to predict the abundance of kelp: ordinary least-squares, multi-layer perceptron and random forest. The regressors are trained using the features from `kelp_metrics.pkl` and the target variable is the abundance of kelp. The regressors are trained using the `regressors_train.py` script.

| Variable | Correlation Coefficient | Mutual Information | Feature Importance |
| -------- | ----------------------- | ------------------ | ------------------ |
| Time             |  -0.032 | 0.155 | 0.059 |
| Elevation [m]    | 0.040   | 0.013 | 0.000 |
| Sunlight [day]   | 0.304   | 0.184 | 0.739 |
| Latitude         | 0.116   | 0.098 | 0.000 |
| Longitude        | -0.164  | 0.103 | 0.004 |
| Temperature      | 0.005   | 0.391 | 0.000 |
| Temperature Lag  | -0.313  | 0.422 | 0.198 |

Even though the parameters individually may be correlated to the amount of kelp, the random forest regression model suggests only the amount of sunlight, temperature from the previous quarter and time are important for making a prediction.


# Datasets

We use various data sets including kelp biomass, sea surface temperature, and digital elevation models.

## Kelp Biomass

[Kelp Watch](https://kelpwatch.org/) is an online platform that provides access to satellite data on kelp canopy dynamics along the west coast of North America. Developed through a collaboration between researchers and conservation groups, Kelp Watch uses Landsat imagery to quantify seasonal giant kelp and bull kelp canopy area in 30x30m regions spanning from Baja California, Mexico to Oregon since 1984. The interactive web interface allows users to visualize, analyze, and download the kelp canopy data to support research and inform management decisions. Key applications include assessing long-term trends, impacts of disturbances like marine heatwaves, and local kelp forest dynamics. Overall, Kelp Watch makes complex satellite data more accessible to better understand and manage these valuable kelp forest ecosystems. The Kelp Watch project monitors ~500,000 locations along the west coast.

Data URL: https://sbclter.msi.ucsb.edu/data/catalog/package/?package=knb-lter-sbc.74

![](Figures/kelp_west_coast.png)

`kelp_area_m^2` - The total emergent kelp canopy area in square meters within the selected geometry. Cells with no numerical value correspond to instances when the scene was either obstructed by clouds and/or no clear observation of the area was available and no measurement was obtained. The nan's and zeros should be filtered out in correlation estimates.


| Quarter | Season        | Months                  | Date              |
| ------- | ------------- | ----------------------- | ----------------- |
| Q1      | winter        | January – March         | 02-15T00:00:00.00 |
| Q2      | spring        | April – June            | 05-15T00:00:00.00 |
| Q3      | summer        | July – September        | 08-15T00:00:00.00 |
| Q4      | fall          | October – December      | 11-15T00:00:00.00 |


## Sea Surface Temperature

We use the [JPL MUR SST](https://podaac.jpl.nasa.gov/dataset/MUR-JPL-L4-GLOB-v4.1) data set to get the sea surface temperature data. The data for this project are monthly averages of the SST at 0.01 deg resolution. For more information see [Kalmus et al. 2022](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2021EF002608)

![](Figures/temperature_map.png)


## Digital Elevation Models

GEBCO - https://www.gebco.net/data_and_products/gridded_bathymetry_data/ (sub-ice topo/bathy; 15 arc-second resolution)

NOAA - https://www.ncei.noaa.gov/products/coastal-relief-model (Southern California Version 2; 1 arc-second resolution)


## To Do
- Get future temperature data + make predictions using best regressor
- Update CMDA server
