# script to read in monthly SST temperatures and interpolate them onto the same time grid as the kelp data
import argparse
import numpy as np
import xarray as xr
from tqdm import tqdm
from joblib import dump
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree

class NearestNeighbor():
    def __init__(self, lats, lons, data):
        self.lats = lats
        self.lons = lons
        self.data = data

        # Create a KDTree using latitudes and longitudes
        self.tree = cKDTree(list(zip(self.lats,self.lons)))

    def __call__(self, lat, lon, debug=False):

        # Find the index of the nearest point
        _, index = self.tree.query([lat, lon])

        # Get the sea surface temperature at the nearest point
        nearest_sst = self.data[:, index]

        if debug:
            # difference between lat/lon
            dlat = np.abs(self.lats[index] - lat)
            dlon = np.abs(self.lons[index] - lon)
            print(f"lat: {lat:.2f} lon: {lon:.2f} dlat: {dlat:.2f} dlon: {dlon:.2f}")
            # convert to km
            dlat = dlat * 111.32
            dlon = dlon * 111.32 * np.cos(np.deg2rad(lat))
            # add units to print
            print(f"lat: {lat:.2f} lon: {lon:.2f} dlat: {dlat:.2f} km dlon: {dlon:.2f} km")
            return nearest_sst, dlat, dlon
        else:
            return nearest_sst

if __name__ == "__main__":

    # cmd line args
    parser = argparse.ArgumentParser()
    # kelp file
    parser.add_argument('-k', '--kelp_file', type=str,
                        help='path to kelp biomass file',
                        default="Data/LandsatKelpBiomass_2023_Q3_withmetadata.nc")
    # climate scenario (ssp126, ssp585)
    parser.add_argument('-c', '--climate_scenario', type=str,
                        help='climate scenario (ssp126, ssp585)',
                        default="ssp126")
    # climate model (CanESM5/GFDL-ESM4)
    parser.add_argument('-m', '--climate_model', type=str,
                        help='climate model (CanESM5/GFDL-ESM4)',
                        default="CanESM5")
    # std vs BGL downscaling
    parser.add_argument('-d', '--downscaling', type=str,
                        help='downscaling method (std, BGL)',
                        default="BGL")
    args = parser.parse_args()

    # load the Kelp biomass
    kelp = xr.open_dataset(args.kelp_file)

    # create projected time from 1900-2100 with 3 month intervals, similar to kelp time
    projected_times = []
    for year in range(2000,2100):
        for month in [2,5,8,11]:
            projected_times.append(np.datetime64(f'{year}-{month:02d}-15'))
    projected_times = np.array(projected_times)

    # load the monthly SST data from climate simulation
    sim_data = xr.open_dataset(f"Data/tos_Omon_{args.climate_model}_{args.climate_scenario}_r1i1p1f1_gr_2002-2100.downscaled_{args.downscaling}.unique.nc", decode_times=False)

    # Dimensions:  (index: 339690, time: 972)
    # Coordinates:
    #   * index    (index) float64 1.0 2.0 3.0 4.0 ... 3.397e+05 3.397e+05 3.397e+05
    #   * time     (time) float64 1.051e+06 1.052e+06 1.053e+06 ... 1.759e+06 1.76e+06
    # Data variables:
    #     sst      (time, index) float32 ...
    #     lon      (index) float32 ...
    #     lat      (index) float32 27.01 27.1 27.1 ... 9.969e+36 9.969e+36 9.969e+36
    #     months   (time) float32 ...
    #   * time     (time) float64 1.051e+06 1.052e+06 1.053e+06 ... 1.759e+06 1.76e+06
    # Attributes:
    #     units:    hours since 1900-01-16T12:00:00

    # convert time to datetime64[ns]
    sim_times =  np.datetime64('1900-01-16T12:00:00') + np.array(sim_data.time.values, dtype='timedelta64[h]')

    # create nearest neighbor interpolation function
    sim_sst = NearestNeighbor(sim_data.lat.values, sim_data.lon.values, sim_data.sst.values)
    sim_data.close()

    print("reading in kelp data...")
    data = []
    # for each kelp location
    for i in tqdm(range(kelp.latitude.shape[0])):

        # check location is within the grid
        # if kelp.latitude.values[i] > sim_sst.lats.max() or \
        #     kelp.latitude.values[i] < sim_sst.lats.min() or \
        #     kelp.longitude.values[i] > sim_sst.lons.max() or \
        #     kelp.longitude.values[i] < sim_sst.lons.min():
        #     continue

        # create a dictionary to store data
        location_data = {
            'lat': kelp.latitude.values[i],
            'lon': kelp.longitude.values[i],
            'sim_time': [],
            'sim_temp': [], # Monthly averaged 0.01-degree MUR SST
            'kelp_area': kelp.area.values[:,i],
            'kelp_time': kelp.time.values,
            # kelp_temp - temperatures interpolated to kelp_time
            # kelp_temp_std - standard deviation of temperatures interpolated to kelp_time
        }

        data.append(location_data)

    kelp.close()

    # extract SST for each kelp location
    for i in tqdm(range(len(data))): # for each kelp location

        # nearest neighbor interpolation
        sim_temp = sim_sst(data[i]['lat'], data[i]['lon'])
        #temp = sim_sst.sel(lat=lat, lon=lon, method='nearest').sst.values

        # if all values are nan, skip
        if np.all(np.isnan(sim_temp)):
            continue

        # save data
        data[i]['sim_temp'].extend(sim_temp)
        data[i]['sim_time'].extend(sim_times)

    # remove empty entries
    data = [d for d in data if len(d['sim_temp']) > 0]

    print("interpolating SST data onto kelp time grid...")

    # Then for each kelp location interpolate SST onto same time grid as kelp data
    for ii in tqdm(range(len(data))): # for each kelp location
        # cast as numpy arrays
        data[ii]['sim_time'] = np.array(data[ii]['sim_time'])
        data[ii]['sim_temp'] = np.array(data[ii]['sim_temp'])

        # time since minimum time
        stime = (data[ii]['sim_time'] - data[ii]['sim_time'].min()).astype(float)

        # interpolate sim data to kelp time grid
        fn_temp = interp1d(stime, data[ii]['sim_temp'], kind='linear', fill_value=np.nan, bounds_error=False)

        # interpolate sim data to kelp time grid
        ktime = (data[ii]['kelp_time'] - data[ii]['sim_time'].min()).astype(float)
        data[ii]['kelp_temp'] = fn_temp(ktime)

        # interpolate mur data to projected time grid
        ptime = (projected_times - data[ii]['sim_time'].min()).astype(float)
        
        data[ii]['projected_time'] = projected_times
        data[ii]['projected_temp'] = fn_temp(ptime)

    # save data
    with open(f'Data/kelp_interpolated_sst_{args.climate_scenario}_{args.downscaling}.pkl', 'wb') as f:
        dump(data, f)


    # make plot
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots(1,1,figsize=(12,4))
    ax.plot(data[0]['sim_time'], data[0]['sim_temp'], '-', label='sim', alpha=0.5)
    #ax.plot(data[0]['kelp_time'], data[0]['kelp_temp'], '-', label='kelp')
    ax.plot(data[0]['projected_time'], data[0]['projected_temp'], '--', label='projected', alpha=0.5)
    ax.plot(data[-1]['sim_time'], data[-1]['sim_temp'], '-', label='sim', alpha=0.5)
    #ax.plot(data[0]['kelp_time'], data[0]['kelp_temp'], '-', label='kelp')
    ax.plot(data[-1]['projected_time'], data[-1]['projected_temp'], '--', label='projected', alpha=0.5)
    ax.set_xlabel('time')
    ax.set_ylabel('SST')
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'Data/kelp_timeseries_{args.climate_model}_{args.climate_scenario}_{args.downscaling}.png', dpi=300)
    plt.close()