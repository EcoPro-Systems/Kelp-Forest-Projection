# script to read in monthly SST temperatures and interpolate them onto the same time grid as the kelp data
import os
import glob
import pickle
import numpy as np
import xarray as xr
from tqdm import tqdm
from scipy.interpolate import interp1d

# load the Kelp biomass
kelp_file = "/home/jovyan/shared/data/ecopro/KelpForest/LandsatKelpBiomass_2022_Q4_withmetadata.nc"
kelp = xr.open_dataset(kelp_file)

# load the monthly SST data
mur_dir = "/home/jovyan/shared/data/ecopro/MUR/"
mur_files = glob.glob(os.path.join(mur_dir, "*MUR*.nc"))

data = []

print("reading in kelp data...")
# for each kelp location
for i in tqdm(range(kelp.latitude.shape[0])):

    # create a dictionary to store data
    location_data = {
        'lat': kelp.latitude.values[i],
        'long': kelp.longitude.values[i],
        'mur_time': [],
        'mur_temp': [], # Monthly averaged 0.01-degree MUR SST
        'mur_temp_std': [],
        'kelp_area': kelp.area.values[:,i],
        'kelp_time': kelp.time.values,
        # kelp_temp - temperatures interpolated to kelp_time
        # kelp_temp_std - standard deviation of temperatures interpolated to kelp_time
    }

    data.append(location_data)

print("interpolating SST data...")
# open files for sea surface temperatures
for mf in tqdm(mur_files): 
    ds = xr.open_dataset(mf)

    # extract SST for each kelp location
    for i in range(kelp.latitude.shape[0]): # for each kelp location

        # get temp at closest point
        #temp = ds['monthly_mean_sst'].sel(lat=kelp.latitude.values[i], lon=kelp.longitude.values[i], method='nearest')
        #temp_std = ds['monthly_std_sst'].sel(lat=kelp.latitude.values[i], lon=kelp.longitude.values[i], method='nearest')

        # linear interpolation
        temp = ds.interp(lat=kelp.latitude.values[i], lon=kelp.longitude.values[i]).monthly_mean_sst.values[0]
        temp_std = ds.interp(lat=kelp.latitude.values[i], lon=kelp.longitude.values[i]).monthly_std_sst.values[0]

        # save data
        data[i]['mur_temp'].append(temp)
        data[i]['mur_temp_std'].append(temp_std)
        data[i]['mur_time'].append(ds.time.values[0])

    ds.close()

print("interpolating SST data onto kelp time grid...")
# Then for each kelp location interpolate SST onto same time grid as kelp data
for ii in tqdm(range(0,kelp.latitude.shape[0])): # for each kelp location
    # cast as numpy arrays
    data[ii]['mur_time'] = np.array(data[ii]['mur_time'])
    data[ii]['mur_temp'] = np.array(data[ii]['mur_temp'])

    # interpolate mur data to kelp time grid
    fn_temp = interp1d(data[ii]['mur_time'].astype(float), data[ii]['mur_temp'], kind='linear', fill_value=np.nan, bounds_error=False)
    data[ii]['kelp_temp'] = fn_temp(data[ii]['kelp_time'].astype(float))

    # interpolate stdev of mur data to kelp time grid
    fn_temp_std = interp1d(data[ii]['mur_time'].astype(float), data[ii]['mur_temp_std'], kind='linear', fill_value=np.nan, bounds_error=False)
    data[ii]['kelp_temp_std'] = fn_temp_std(data[ii]['kelp_time'].astype(float))


# save data
with open(f'kelp_interpolated_data.pkl', 'wb') as f:
    pickle.dump(data, f)
