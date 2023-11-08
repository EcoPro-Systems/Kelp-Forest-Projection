import pickle
import numpy as np
import xarray as xr
from tqdm import tqdm
from statsmodels.regression.linear_model import OLS
import os
import zipfile
import datetime

import astropy.units as u
from astropy.coordinates import EarthLocation, AltAz, get_sun
from astropy.time import Time
import numpy as np

def calculate_day_length(latitude, longitude, dates):
    #     # Example usage:
    #     latitude = 40.7128  # Latitude of New York City
    #     longitude = -74.0060  # Longitude of New York City
    #     date = np.datetime64("2023-09-01")  # Date in numpy.datetime64 format

    durations = np.zeros(len(dates))
    location = EarthLocation(lat=latitude*u.deg, lon=longitude*u.deg)

    for i,date in enumerate(dates):
        times = Time(date)
        
        # Calculate the altitude of the Sun at various times during the day
        times_span = times + np.linspace(-12, 12, 24*60)*u.hour
        altaz_frame = AltAz(obstime=times_span, location=location)
        sun_altitudes = get_sun(times_span).transform_to(altaz_frame).alt
        
        # Find sunrise and sunset times by looking for altitude crossings
        sunrise_idx = np.where(sun_altitudes > 0)[0][0]
        sunset_idx = np.where(sun_altitudes > 0)[0][-1]
        
        sunrise_time = times_span[sunrise_idx]
        sunset_time = times_span[sunset_idx]
        
        durations[i] = (sunset_time - sunrise_time).to(u.day).value
    
    return durations


# make function to extract kelp data and temperatures
def extract_kelp_metrics(data, bathymetry, sunlight, lowerBound, upperBound):
    """
    Extract various metrics like correlation coefficients and slopes from kelp data and differences in temperature

    Parameters
    ----------
    data : list
        list of dictionaries containing kelp data

    bathymetry : xarray
        bathymetry data

    lowerBound : float
        lower bound of latitude

    upperBound : float
        upper bound of latitude 
    """

    # initialize dictionary of data
    kelp_data = {
        'dkelp':[],          # one quarter difference in kelp area
        'dkelp_kelp':[],     # kelp area at same time as difference (average of neighboring quarters)
        'dtemp':[],          # one quarter difference in temperature
        'dtemp_temp':[],     # temperature at same time as difference (average of neighboring quarters)
        'dtemp_temp_lag':[], # temperature one quarter before 
        'dtemp_temp_lag2':[],# temperature two quarter before 
        'kelp':[],           # total kelp area
        'temp':[],           # temperature [K]
        'temp_lag':[],       # temperature one quarter before [K]
        'temp_lag2':[],      # temperature two quarters before [K]
        'sunlight':[],       # daylight duration [day]
        'time':[],           # time
        'dtime':[],          # time of difference
        'lat':[],            # latitude
        'lon':[],            # longitude
        'dlat':[],           # latitude corresponding to difference measurements
        'dlon':[],           # longitude corresponding to difference measurements
        'elevation':[],      # elevation [m]
        'delevation':[],     # elevation corresponding to difference measurements [m]
    }

    # loop over all locations
    for d in tqdm(data):

        # skip data that is not between the upper and lower bounds
        if (d['lat'] >= upperBound) or (d['lat'] < lowerBound): 
            continue 

        # filter out areas with no measurements and nans
        bad_mask = (d['kelp_area'] == 0) | (np.isnan(d['kelp_area'])) | (np.isnan(d['kelp_temp']))

        # compute temperature one quarter before
        d['kelp_temp_lag'] = np.roll(d['kelp_temp'],1)
        d['kelp_temp_lag'][0] = np.nan

        # compute temperature two quarters before
        d['kelp_temp_lag2'] = np.roll(d['kelp_temp'],2)
        d['kelp_temp_lag2'][0] = np.nan
        d['kelp_temp_lag2'][1] = np.nan

        # save data
        kelp_data['kelp'].extend(d['kelp_area'][~bad_mask])
        kelp_data['temp'].extend(d['kelp_temp'][~bad_mask])
        kelp_data['temp_lag'].extend(d['kelp_temp_lag'][~bad_mask])
        kelp_data['temp_lag2'].extend(d['kelp_temp_lag2'][~bad_mask])
        kelp_data['time'].extend(d['kelp_time'][~bad_mask])

        # save latitude and longitude
        kelp_data['lat'].extend(d['lat']*np.ones(len(d['kelp_area'][~bad_mask])))
        kelp_data['lon'].extend(d['long']*np.ones(len(d['kelp_area'][~bad_mask])))

        # too slow to sample daylight duration for every data point
        #daylight_duration = calculate_day_length(d['lat'], d['long'], d['kelp_time'][~bad_mask])
    
        # change year on every date to 2023 to match sunlight grid for interpolation
        kelp_time = d['kelp_time'][~bad_mask]
        kelp_time = np.array([np.datetime64(str(d)[:10]) for d in kelp_time])
        kelp_time = np.array([f'2023-{str(d)[5:]}' for d in kelp_time])
        kelp_time = np.array([np.datetime64(d) for d in kelp_time])
        kelp_time = kelp_time.astype('datetime64[ns]')
        
        # calculate daylight duration
        if len(kelp_time) > 0:
            daylight_duration = sunlight.interp(time=kelp_time, lat=d['lat'], lon=d['long']).daylight_duration.values
            kelp_data['sunlight'].extend(daylight_duration)

        # use linear interpolation
        elevation = bathymetry.interp(lat=d['lat'],lon=d['long']).elevation.values
        kelp_data['elevation'].extend(elevation*np.ones(len(d['kelp_area'][~bad_mask])))

        # properly estimate the difference data
        nanmask = d['kelp_area'] == 0 # mask out areas with no kelp
        nandata = d['kelp_area'].copy()
        nandata[nanmask] = np.nan # set areas with no kelp to nan
        nandiff = np.diff(nandata) # take quarterly difference
        nonnanmask_kelp = ~np.isnan(nandiff) # mask out nans to ensure difference between sequential quarters
        nandiff_temp = np.diff(d['kelp_temp']) # take quarterly difference of temperature
        nonnanmask_temp = ~np.isnan(nandiff_temp) # mask out nans to ensure difference between sequential quarters
        nonnanmask = nonnanmask_kelp & nonnanmask_temp # mask out nans in both kelp and temperature data

        kelp_data['dkelp'].extend(nandiff[nonnanmask]) # save kelp difference data
        kelp_data['dkelp_kelp'].extend((d['kelp_area'][1:][nonnanmask] + d['kelp_area'][:-1][nonnanmask])/2) # average kelp area at same time as difference

        # average temperature at same time as difference
        kelp_data['dtemp_temp'].extend((d['kelp_temp'][1:][nonnanmask] + d['kelp_temp'][:-1][nonnanmask])/2)
        kelp_data['dtemp_temp_lag'].extend((d['kelp_temp_lag'][1:][nonnanmask] + d['kelp_temp_lag'][:-1][nonnanmask])/2)
        kelp_data['dtemp_temp_lag2'].extend((d['kelp_temp_lag2'][1:][nonnanmask] + d['kelp_temp_lag2'][:-1][nonnanmask])/2)

        kelp_data['dtemp'].extend(nandiff_temp[nonnanmask]) # save temperature difference data

        # save time data for differences by averaging time of sequential quartersq
        times_prev = d['kelp_time'][1:][nonnanmask].astype('datetime64[ns]')
        times_next = d['kelp_time'][:-1][nonnanmask].astype('datetime64[ns]')
        delta = times_next - times_prev
        times_mid = times_prev + delta/2
        kelp_data['dtime'].extend(times_mid.astype(str))

        # save latitude and longitude
        kelp_data['dlat'].extend(d['lat']*np.ones(len(nandiff[nonnanmask])))
        kelp_data['dlon'].extend(d['long']*np.ones(len(nandiff[nonnanmask])))
        
        # add delevation
        kelp_data['delevation'].extend(elevation*np.ones(len(nandiff[nonnanmask])))

        # free up memory
        del nanmask, nandata, nandiff, nonnanmask, nandiff_temp, bad_mask

    # convert to numpy arrays
    for k in kelp_data.keys():
            # convert to numpy arrays
        for k in kelp_data.keys():
            try:
                kelp_data[k] = np.array(kelp_data[k])
            except:
                print("could not convert", k)
        kelp_data[k] = np.array(kelp_data[k])

    # convert time to datetime
    kelp_data['time'] = kelp_data['time'].astype('datetime64[ns]')
    kelp_data['dtime'] = kelp_data['dtime'].astype('datetime64[ns]')

    # measure correlation and trend line
    A = np.vstack([kelp_data['dtemp_temp']-273.15, np.ones(len(kelp_data['dtemp_temp']))]).T
    res = OLS(kelp_data['dkelp'], A).fit()
    m,b = res.params[0], res.params[1]

    # define characteristic temperature where change in kelp is 0
    # just solve for x above to get the characteristic temperature at each location
    kelp_data['temp_char'] = -b/m
    kelp_data['average_temp'] = np.mean(kelp_data['dtemp_temp'])-273

    # saving the slope + error
    kelp_data['slope_dkelp_temp_char'] = m
    kelp_data['slope_dkelp_temp_char_err'] = res.bse[0]
    
    return kelp_data

if __name__ == "__main__":
    # load data
    with open('Data/kelp_interpolated_data.pkl', 'rb') as f:
        data = pickle.load(f)

    # load bathymetry data
    #bathymetry = xr.open_dataset('Data/GEBCO_2022_sub_ice_topo.nc')
    #lower_lat = 37
    #upper_lat = 50

    # check if data file exists or unzip it
    if not os.path.exists('Data/crm_socal_1as_vers2.nc'):
        # if running for first time
        with zipfile.ZipFile('Data/crm_socal_1as_vers2.nc.zip', 'r') as zip_ref:
            zip_ref.extractall('Data/')

    # load bathymetry data
    bathymetry = xr.open_dataset('Data/crm_socal_1as_vers2.nc')
    bathymetry = bathymetry.rename({'Band1':'elevation'})

    # if using noaa dem: limit: ~31-36
    lower_lat = 33
    upper_lat = 36

    # create a grid for computing daylight
    sunlight_file = f'Data/sunlight_{lower_lat}_{upper_lat}.nc'
    if os.path.exists(sunlight_file):
        sunlight = xr.open_dataset(sunlight_file)
    else:
        print("Computing daylight duration...")

        # find all unique dates and lat/lon limit
        dates = data[0]['kelp_time']

        lats = []; lons = []
        # loop over all locations and extract lat/long
        for d in data:
            lats.append(d['lat'])
            lons.append(d['long'])
        # convert to numpy array
        lats = np.array(lats)
        lons = np.array(lons)

        # mask data based on lat limit
        mask = (lats >= lower_lat) & (lats < upper_lat)
        lats = lats[mask]
        lons = lons[mask]

        # create a grid of lat/long
        lat_list = np.linspace(lats.min(), lats.max(), 50)
        lon_list = np.linspace(lons.min(), lons.max(), 50)
        lat_grid, lon_grid = np.meshgrid(lat_list, lon_list)

        # ignore the year and find unique dates
        dates = np.unique(np.array([np.datetime64(str(d)[:10]) for d in dates]))
        # dates = array(['1984-02-15', '1984-05-15', '1984-08-15', '1984-11-15'])
        
        # ignore year and find unique dates as string
        dates_str = np.unique(np.array([str(d)[5:] for d in dates]))
        
        # add the year 2023 to each date
        dates_str = np.array([f'2023-{d}' for d in dates_str])

        # convert to datetime
        dates = np.array([np.datetime64(d) for d in dates_str])

        # create a grid of daylight duration
        sunlight = np.zeros((len(dates), len(lat_list), len(lon_list)))

        # loop over all dates
        for j in tqdm(range(len(lat_list))):
            for k in range(len(lon_list)):
                # calculate daylight duration
                sunlight[:,j,k] = calculate_day_length(lat_list[j], lon_list[k], dates)

        # convert to xarray so we can interpolate
        # fix this error: AttributeError: 'DataArray' object has no attribute 'daylight_duration'
        sunlight_xr = xr.DataArray(sunlight, dims=['time', 'lat', 'lon'], coords={'time':dates, 'lat':lat_list, 'lon':lon_list}, name='daylight_duration')
        
        # save grid to netcdf file
        sunlight_xr.to_netcdf(sunlight_file)

        # free up memory
        del lat_list, lon_list, lat_grid, lon_grid, dates, dates_str, sunlight, sunlight_xr

        # load file
        sunlight = xr.open_dataset(sunlight_file)

    # extract kelp metrics
    kelp_data = extract_kelp_metrics(data, bathymetry, sunlight, lower_lat, upper_lat)

    # save to pkl file
    with open(f"Data/kelp_metrics_{lower_lat}_{upper_lat}.pkl", "wb") as f:
        pickle.dump(kelp_data, f)
