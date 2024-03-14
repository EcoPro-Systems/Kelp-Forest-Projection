import numpy as np
import xarray as xr
import cartopy.feature
import cartopy.crs as ccrs
import matplotlib.pyplot as plt


kelp_path = "Data/LandsatKelpBiomass_2023_Q3_withmetadata.nc"
temp_path = "Data/200305-JPL-L4-SSTfnd-MUR_monthly-GLOB-fv04.2.nc"

# load data
kelp = xr.open_dataset(kelp_path)
print(kelp)
temp = xr.open_dataset(temp_path)
print(temp)

# size of box
lat = 32
lon = -119
radius = 5

# find the closest point
dist = np.sqrt((kelp.longitude.values-lon)**2 + (kelp.latitude.values-lat)**2)
idx = np.argmin(dist)

# find how many points are within a 0.1 degree radius
user_idxs = np.where(dist<radius)[0]

if len(user_idxs) == 0:
    print("No kelp data found in this area")

lat_min = lat - radius
lat_max = lat + radius
long_min = lon - radius
long_max = lon + radius

# set up the figure
fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())

# draw the map
ax.coastlines()
ax.set_global()
ax2.coastlines()
ax2.set_global()

ax.set_title(f"Kelp Locations", fontsize=20)
ax2.set_title(f"Sea Surface Temperature from JPL MUR", fontsize=20)

# color terrain
ax.add_feature(cartopy.feature.LAND, facecolor='lightgray')
ax.add_feature(cartopy.feature.OCEAN, facecolor='lightblue')

# plot mean locations
im = ax.scatter(kelp.longitude.values,kelp.latitude.values, c=[0.15]*len(kelp.latitude), marker='.', label="Kelp Watch", transform=ccrs.PlateCarree(), cmap='rainbow',vmin=0,vmax=1)

cbar = plt.colorbar(im, ax=ax, shrink=0.75)
cbar.remove()

# add labels on axes every 10 degrees
ax.set_xticks(np.arange(-180, 180, 2), crs=ccrs.PlateCarree())
ax.set_xticklabels([f'{x}°' for x in np.arange(-180, 180, 2)], fontsize=12, rotation=45)
ax.set_yticks(np.arange(-90, 90, 2), crs=ccrs.PlateCarree())
ax.set_yticklabels([f'{x}°' for x in np.arange(-90, 90, 2)], fontsize=12)

ax.set_xlabel(f'Longitude [E]', fontsize=16)
ax.set_ylabel(f'Latitude [N]', fontsize=16)
ax.grid(True,color='k', alpha=0.5, linestyle='--', linewidth=0.5)

# zoom in on the bounding box
ax.set_extent([-128, -110, 20, 50], crs=ccrs.PlateCarree())

# draw bounding box around the region
ax.plot([long_min, long_min, long_max, long_max, long_min], [lat_min, lat_max, lat_max, lat_min, lat_min], color='r', label='ROI', transform=ccrs.PlateCarree())
ax.grid(True, ls='--')
# add legend
ax.legend(loc='upper right', fontsize=16)

# plot the temperature data
im = ax2.imshow(temp.monthly_mean_sst[0,:,:]-273.15, origin='lower', extent=[-180, 180, -90, 90], transform=ccrs.PlateCarree(), cmap='jet', interpolation='none', vmin=11, vmax=24)
ax2.plot([long_min, long_min, long_max, long_max, long_min], [lat_min, lat_max, lat_max, lat_min, lat_min], color='r', label='ROI', transform=ccrs.PlateCarree())

# add temperature label to cbar
cbar = plt.colorbar(im, ax=ax2, shrink=0.9, pad=0.05)
cbar.set_label('Temperature [C]', rotation=270, labelpad=15, fontsize=14)

# add labels on axes every 10 degrees
ax2.set_xticks(np.arange(-180, 180, 2), crs=ccrs.PlateCarree())
ax2.set_xticklabels([f'{x}°' for x in np.arange(-180, 180, 2)], fontsize=12, rotation=45)
ax2.set_yticks(np.arange(-90, 90, 2), crs=ccrs.PlateCarree())
ax2.set_yticklabels([f'{x}°' for x in np.arange(-90, 90, 2)], fontsize=12)
ax2.add_feature(cartopy.feature.LAND, facecolor='lightgray')

ax2.set_xlabel(f'Longitude [E]', fontsize=16)
ax2.set_ylabel(f'Latitude [N]', fontsize=16)
ax2.grid(True,color='k', alpha=0.5, linestyle='--', linewidth=0.5)

# zoom in on the bounding box
ax2.set_extent([-128, -110, 20, 50], crs=ccrs.PlateCarree())

plt.tight_layout()
plt.savefig('Data/kelp_temp_map.png')
plt.show()
plt.close()