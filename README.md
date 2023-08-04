# Kelp Forest Monitoring

An ecological forecasting model for monitoring the biomass availability in kelp forests on the California coast. 

## Installation

Set up your python environment

`conda env create -f environment.yml`

or 

```
conda create -n kelp python=3.10
conda activate kelp
conda install ipython jupyter pandas matplotlib scipy scikit-learn
conda install -c conda-forge xarray dask netCDF4 bottleneck
pip install tqdm
```


# Datasets

We use various data sets including kelp biomass, sea surface temperature, and digital elevation models.

## Kelp Biomass

[Kelp Watch](https://kelpwatch.org/) is an online platform that provides access to satellite data on kelp canopy dynamics along the west coast of North America. Developed through a collaboration between researchers and conservation groups, Kelp Watch uses Landsat imagery to quantify seasonal giant kelp and bull kelp canopy area within 10x10 km regions spanning from Baja California, Mexico to Oregon since 1984. The interactive web interface allows users to visualize, analyze, and download the kelp canopy data to support research and inform management decisions. Key applications include assessing long-term trends, impacts of disturbances like marine heatwaves, and local kelp forest dynamics. Overall, Kelp Watch makes complex satellite data more accessible to better understand and manage these valuable kelp forest ecosystems.

Data URL: https://sbclter.msi.ucsb.edu/data/catalog/package/?package=knb-lter-sbc.74

![](Figures/kelp_california.png)

`kelp_area_m2` - The total emergent kelp canopy area in square meters within the selected geometry. Cells with no numerical value correspond to instances when the scene was either obstructed by clouds and/or no clear observation of the area was available and no measurement was obtained.


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

## Codes

The python scripts can be run locally and the jupyter notebooks are meant to be run on the CMDA server.

| Code | Description |
| ---- | ----------- |
| `kelp_gridding.py`  `Grid_Temp_to_Kelp.ipynb` | Interpolate the monthly SST data onto the same grid as the kelp data and create a new file called: `kelp_interpolated_data.pkl` |
| `kelp_metrics.py` | Calculate the various metrics like lag temps and derivatives for each kelp location then save the data to a new file called: `kelp_metrics.pkl`. These metrics are ultimately used as features for our regression algorithm. |

TO DO
- create time series plots
- create correlation plots
- regression model + prediction



```json
kelp_interpolated_data = [
{'lat': 27.1033379433901,
 'long': -114.288749436661,
 'mur_time': array(['2015-10-15T21:00:00.000000000', '2020-05-15T21:00:00.000000000',
        '2003-04-15T09:00:00.000000000', '2013-12-15T21:00:00.000000000',
        '2005-02-14T09:00:00.000000000', '2004-03-15T21:00:00.000000000',
         ...
        '2004-02-14T21:00:00.000000000', '2012-12-15T21:00:00.000000000',
        '2014-10-15T21:00:00.000000000', '2013-11-15T09:00:00.000000000',
        '2005-01-15T21:00:00.000000000', '2003-03-15T21:00:00.000000000',
        '2020-04-15T09:00:00.000000000', '2013-02-14T09:00:00.000000000',]),
 'mur_temp': array([299.384  , 290.146  , 287.75   , 291.74698, 290.042  , 288.117  ,
        289.754  , 296.022  , 288.807  , 291.177  , 291.948  , 290.333  ,
        292.004  , 288.615  , 290.123  , 292.948  , 296.639  , 293.873  ,
        290.40298, 292.306  , 297.013  , 294.694  , 290.17798, 294.043  ,
        ...
        295.227  , 290.258  , 295.85498, 292.669  , 287.998  , 293.235  ,
        298.947  , 293.712  , 291.421  , 288.292  , 288.763  , 288.895  ,
        293.267  , 289.93298, 291.845  , 294.632  , 290.422  , 287.93   ,
        292.472  ], dtype=float32),
 'kelp_time': array(['1984-02-15T00:00:00.000000000', '1984-05-15T00:00:00.000000000',
        '1984-08-15T00:00:00.000000000', '1984-11-15T00:00:00.000000000',
        '1985-02-15T00:00:00.000000000', '1985-05-15T00:00:00.000000000',
        '1985-08-15T00:00:00.000000000', '1985-11-15T00:00:00.000000000',
        '1986-02-15T00:00:00.000000000', '1986-05-15T00:00:00.000000000',
          ...
        '2021-08-15T00:00:00.000000000', '2021-11-15T00:00:00.000000000',
        '2022-02-15T00:00:00.000000000', '2022-05-15T00:00:00.000000000',
        '2022-08-15T00:00:00.000000000', '2022-11-15T00:00:00.000000000'],
       dtype='datetime64[ns]'),
 'kelp_area': array([ nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,  nan,   0.,  nan,
         nan,  nan,   0.,   0.,  nan,  nan,  nan,  nan,  nan,  nan,   0.,
          0.,   0.,  nan,   0.,  nan,  nan,  nan,   0.,  nan,  nan,  nan,
          0.,   0.,  nan,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
          0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
          0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
          0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
        482., 194.,   0.,  68., 116., 132.,   0.,  17., 325., 233.,   0.,
          0., 367.,  64.,   0.,   0.,   0.,   0., 121., 662., 767.,  45.,
          0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
          0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
        315.,  32.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
          0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
          0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
          0.,   0.]),
 'kelp_temp': array([         nan,          nan,          nan,          nan,
                 nan,          nan,          nan,          nan,
                 nan,          nan,          nan,          nan,
                 nan,          nan,          nan,          nan,
        ...
        288.11056221, 290.11753495, 295.60852629, 293.21854901,
        289.15287285, 289.77734012, 295.68311827, 295.98688144,
        288.88905684, 290.02875319, 293.25067532, 293.71855876,]),
}, ...]
```

The nan's represent missing data either due to incompleteness in either data product for a certain time. The nan's and zeros should be filtered out in future correlation estimates.