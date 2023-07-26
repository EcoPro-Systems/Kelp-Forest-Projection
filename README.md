# KelpForestMonitoring
An ecological forecasting model for monitoring the biomass availability in kelp forests on the California coast.


## Installation

Set up your python environment

`conda env create -f environment.yml`

or 

```
conda create -n kelp python=3.9
conda activate kelp
conda install ipython jupyter
pip install pandas matplotlib scipy sklearn
```



## Kelp Biomass

Kelp data from: https://sbclter.msi.ucsb.edu/data/catalog/package/?package=knb-lter-sbc.74

![](data/kelp_california.png)



## Sea Surface Temperature
- [JPL MUR SST](https://podaac.jpl.nasa.gov/dataset/MUR-JPL-L4-GLOB-v4.1) > Data Access > Search Granules

You may need to register for an account to access the data. Then filter for the dates you want and select download and then direct download to acquire links for each netcdf file. A few download links from our study are below:

- [Download link](https://search.earthdata.nasa.gov/downloads/4436350485) for data from Jan. 2021 - Jan. 2022

- [Alt Link](https://cmr.earthdata.nasa.gov/virtual-directory/collections/C1996881146-POCLOUD/temporal), [AWS Link](https://registry.opendata.aws/mur/#usageexa)

![](data/temperature_map.png)

Alternatively, we use a down-sampled version of the temperature data which is averaged on a monthly time frame. *Insert reference to how the data was downsampled*

25-35 deg -> giant kelp

## Codes to add

- notebook to extract kelp metrics
- create time series plots
- create correlation plots
- regression model + prediction