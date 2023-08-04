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
- [JPL MUR SST](https://podaac.jpl.nasa.gov/dataset/MUR-JPL-L4-GLOB-v4.1) > Data Access > Search Granules

You may need to register for an account to access the data. Then filter for the dates you want and select download and then direct download to acquire links for each netcdf file. A few download links from our study are below:

- [Download link](https://search.earthdata.nasa.gov/downloads/4436350485) for data from Jan. 2021 - Jan. 2022

- [Alt Link](https://cmr.earthdata.nasa.gov/virtual-directory/collections/C1996881146-POCLOUD/temporal), [AWS Link](https://registry.opendata.aws/mur/#usageexa)

![](Figures/temperature_map.png)

Alternatively, we use a down-sampled version of the temperature data which is averaged on a monthly time frame. *Insert reference to how the data was downsampled*

25-35 deg -> giant kelp

## Digital Elevation Models

GEBCO - https://www.gebco.net/data_and_products/gridded_bathymetry_data/ (sub-ice topo/bathy; 15 arc-second resolution)

NOAA - https://www.ncei.noaa.gov/products/coastal-relief-model (Southern California Version 2; 1 arc-second resolution)

## Codes

The python scripts can be run locally and the jupyter notebooks are meant to be run on the CMDA server.

| Code | Description |
| ---- | ----------- |
| `kelp_gridding.py`  `Grid_Temp_to_Kelp.ipynb` | Interpolate the monthly SST data onto the same grid as the kelp data and create a new file called: `kelp_interpolated_data.pkl` |
| `kelp_metrics.py` | Calculate the various metrics like lag temps and derivatives for each kelp location then save the data to a new file called: `kelp_metrics.pkl` |

TO DO
- create time series plots
- create correlation plots
- regression model + prediction
