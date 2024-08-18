import os
import glob
import numpy as np
import xarray as xr
from tqdm import tqdm
from scipy.interpolate import interp1d
from joblib import dump

r_earth = 6371.0 # km

def process_location(lat, lon, kelp):
    # average kelp area within 0.01 degrees of the lat/lon

    location_data = {
        'lat': lat,
        'long': lon,
        'mur_time': [],
        'mur_temp': [],
        'mur_temp_std': [],
        'kelp_area': [], # m^2 per km^2
        'kelp_time': kelp.time.values,
    }

    mask = (kelp.latitude >= lat - 0.005) & (kelp.latitude < lat + 0.005) & \
           (kelp.longitude >= lon - 0.005) & (kelp.longitude < lon + 0.005)

    if not kelp.time.values.size:  # Check if 'kelp_time' is empty
        return None

    if not np.any(mask) or np.all(np.isnan(kelp.area.values[:, mask])):
        return None
    
    # calculate the area of the grid cell adjust for latitude
    scale_factor = r_earth * np.pi / 180.0  * np.abs(np.cos(np.deg2rad(lat))) # km per degree
    area = (scale_factor * 0.01 * 1000)**2 # meters squared
    location_data['kelp_area'] = np.nansum(kelp.area.values[:, mask],axis=1) # meters per area
    # normalize kelp area [m^2] to [km^2]
    location_data['kelp_area'] = location_data['kelp_area'] / area * 1000**2 # m^2 per km^2

    return location_data

def interpolate_data(data):
    data['mur_time'] = np.array(data['mur_time'])
    data['mur_temp'] = np.array(data['mur_temp'])

    fn_temp = interp1d(data['mur_time'].astype(float), data['mur_temp'], kind='linear', fill_value=np.nan, bounds_error=False)
    data['kelp_temp'] = fn_temp(data['kelp_time'].astype(float))

    fn_temp_std = interp1d(data['mur_time'].astype(float), data['mur_temp_std'], kind='linear', fill_value=np.nan, bounds_error=False)
    data['kelp_temp_std'] = fn_temp_std(data['kelp_time'].astype(float))
    return data

#kelp_file = "/home/jovyan/efs/data/KelpForest/LandsatKelpBiomass_2022_Q4_withmetadata.nc"
kelp_file = "Data/LandsatKelpBiomass_2023_Q3_withmetadata.nc"
kelp = xr.open_dataset(kelp_file)

#mur_dir = "/home/jovyan/efs/data/MUR"
mur_dir = "SST"
mur_files = glob.glob(os.path.join(mur_dir, "*MUR*.nc"))

print("Reading and processing data...")
data = []
ds = xr.open_dataset(mur_files[0])
lat_grid = ds.lat.values
lon_grid = ds.lon.values
ds.close()

# Find the minimum and maximum latitude and longitude values from the kelp dataset
min_lat, max_lat = kelp.latitude.min().values, kelp.latitude.max().values
min_lon, max_lon = kelp.longitude.min().values, kelp.longitude.max().values

# Filter the lat_grid and lon_grid arrays to include only the values within the kelp's latitude and longitude range
lat_mask = (lat_grid >= min_lat) & (lat_grid <= max_lat)
lon_mask = (lon_grid >= min_lon) & (lon_grid <= max_lon)
lat_grid = lat_grid[lat_mask]
lon_grid = lon_grid[lon_mask]

print("Extracting kelp data...")
for i in tqdm(enumerate(lat_grid), total=len(lat_grid)):
    lat = i[1]
    for j,lon in enumerate(lon_grid):
        location_data = process_location(lat, lon, kelp)
        if location_data is not None:
            data.append(location_data)

# loop over temperature files and add to data
print("Reading SST data...")
for file in tqdm(mur_files):
    ds = xr.open_dataset(file)
    for i, location in enumerate(data):
        lat, lon = location['lat'], location['long']
        location['mur_temp'].append(ds.sel(lat=lat, lon=lon).monthly_mean_sst.values)
        location['mur_temp_std'].append(ds.sel(lat=lat, lon=lon).monthly_std_sst.values)
        location['mur_time'].append(ds.time.values[0])
    ds.close()

# convert to numpy arrays
for location in data:
    location['mur_temp'] = np.array(location['mur_temp'])
    location['mur_temp_std'] = np.array(location['mur_temp_std'])
    location['mur_time'] = np.array(location['mur_time'])

# remove second axis on tmp and tmp_std
for location in data:
    location['mur_temp'] = np.squeeze(location['mur_temp'])
    location['mur_temp_std'] = np.squeeze(location['mur_temp_std'])

print("Interpolating SST data onto kelp time grid...")
data = [interpolate_data(d) for d in tqdm(data)]

with open(f'kelp_averaged_data.pkl', 'wb') as f:
    dump(data, f)


# create a plot of the map
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.set_extent([min_lon, max_lon, min_lat, max_lat])

ax.set_title(f"Kelp Locations", fontsize=20)

# plot kelp locations
for location in data:
    ax.plot(location['long'], location['lat'], color='limegreen', marker='.', markersize=3)

# draw the map
ax.coastlines()
ax.set_global()

# color terrain
ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.OCEAN, facecolor='lightblue')

# add labels on axes every 10 degrees
ax.set_xticks(np.arange(-180, 180, 2), crs=ccrs.PlateCarree())
ax.set_xticklabels([f'{x}°' for x in np.arange(-180, 180, 2)], fontsize=12, rotation=45)
ax.set_yticks(np.arange(-90, 90, 2), crs=ccrs.PlateCarree())
ax.set_yticklabels([f'{x}°' for x in np.arange(-90, 90, 2)], fontsize=12)

# zoom in on the bounding box
ax.set_extent([-128, -110, 22, 50], crs=ccrs.PlateCarree())
ax.grid(True, color='k', alpha=0.5, linestyle='--', linewidth=0.5)
plt.show()
